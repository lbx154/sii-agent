"""Pure GRPO RL rollout for SII-Agent + slime.

This module is a self-contained, OPD-free / teacher-free replacement for
``training.slime_sii_rollout``. It exposes the two hooks slime expects:

* ``generate(args, sample, sampling_params, evaluation)`` — multi-turn ReAct
  rollout that uses the SII-Agent tool registry (``tools.dispatch`` and
  ``tools.tool_specs``) so the policy is trained against exactly the same
  tools the agent uses at evaluation time. Produces token-level loss masks
  for slime's GRPO trainer.
* ``shaped_reward_rm(args, sample, **kwargs)`` — pure rule-based reward
  combining task correctness with tool-use shaping. No teacher model, no
  remote RM server, no logprob distillation.

Configure via the slime launch flags::

    --custom-generate-function-path training.slime_sii_rl.generate
    --custom-rm-path training.slime_sii_rl.shaped_reward_rm
    --advantage-estimator grpo

Reward weights are tunable via environment variables (see
``REWARD_WEIGHTS`` below).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from argparse import Namespace
from typing import Any

from agent.scoring import score_answer
from tools import dispatch, tool_specs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool / prompt configuration
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "final_answer",
)
BROWSECOMP_ALLOWED_TOOLS = ("search", "final_answer")
VISUAL_ALLOWED_TOOLS = (
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "reverse_image_search",
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "final_answer",
)

# Mirrors the SII-Agent benchmark system prompt so RL behaviour matches eval.
SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what you still need.
2. Call exactly ONE available tool.
3. Read the tool result and decide next step.

Available tools for this run:
{tool_list}

Rules:
- Prefer wiki_search for encyclopedic entities; use web_search when wiki is insufficient.
- Use browser_open for one source URL and browser_open_many for several independent URLs.
- DO NOT repeat the same query / URL — refine instead.
- Never put uncertainty or "not found" narratives in `final_answer.answer`; only the concise answer.
- Use as few tool calls as possible while still verifying the answer span.
- You MUST call `final_answer` before the step budget is exhausted.
"""

BROWSECOMP_SYSTEM_PROMPT = """You are solving BrowseComp-Plus questions using a fixed local corpus.

Loop:
1. Call the available search tool with a focused query.
2. If a document-retrieval tool is available, fetch the most promising doc id.
3. Call `final_answer` once the answer is supported by retrieved docs.

Available tools for this run:
{tool_list}

Rules:
- Search tools return the top documents with docid, score, and a snippet.
- Build the answer only from BrowseComp-Plus corpus evidence.
- Cite supporting document ids in square brackets, e.g. [12345].
- Do not use open-web or Wikipedia tools.
- Keep final_answer concise and include citations when possible.
"""

VISUAL_SYSTEM_PROMPT = """You are solving visual factual QA questions with local images and retrieval tools.

Loop:
1. Inspect the image or generate visual search queries.
2. Search/browse to verify external facts when needed.
3. Call `final_answer` when the answer is supported.

Available tools for this run:
{tool_list}

Rules:
- The user prompt contains an `Image path:` line for local image input.
- Prefer `visual_web_search` first; otherwise `image_to_text` or `image_to_search_queries` before web/wiki search.
- Do not treat the first visual guess as proven. Verify named entities with retrieval evidence.
- Keep final_answer concise.
"""

# ---------------------------------------------------------------------------
# Reward shaping configuration (tunable via env vars)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Reward design notes:
# - The benchmark's failure modes (per project context) are (a) redundant searches
#   and (b) low confidence / no final_answer. Reward is shaped to directly attack
#   these without overwhelming the dominant task-correctness signal.
# - GRPO normalises within a prompt group, so what matters is *differences* between
#   rollouts. Magnitudes are kept small relative to task_correct=1.0 so correctness
#   stays the primary lever.
REWARD_WEIGHTS = {
    "task_correct": _env_float("SII_RL_REWARD_TASK", 1.0),         # primary signal
    "final_called_bonus": _env_float("SII_RL_REWARD_FINAL", 0.1),   # clean final_answer call
    "final_text_bonus": _env_float("SII_RL_REWARD_FINAL_TEXT", 0.0),  # no credit for text-only
    "tool_use_bonus": _env_float("SII_RL_REWARD_TOOL_USE", 0.05),  # >=1 non-final tool
    "no_tool_penalty": _env_float("SII_RL_REWARD_NO_TOOL", 0.05),  # hallucinated immediately
    "format_penalty": _env_float("SII_RL_REWARD_FORMAT", 0.3),     # ran out of steps
    "redundant_penalty": _env_float("SII_RL_REWARD_REDUNDANT", 0.15),  # core anti-redundancy
    "step_penalty": _env_float("SII_RL_REWARD_STEP", 0.04),        # per call over budget
    "step_budget": _env_int("SII_RL_REWARD_STEP_BUDGET", 5),        # first 5 calls free
    "reward_clip_low": _env_float("SII_RL_REWARD_CLIP_LOW", -0.5),
    "reward_clip_high": _env_float("SII_RL_REWARD_CLIP_HIGH", 1.5),
}

# ---------------------------------------------------------------------------
# Parsing helpers — accept native, Qwen XML, and JSON-list tool-call formats
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _coerce_parameter(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _parse_json_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if isinstance(name, str) and isinstance(args, dict):
        return name, args
    return None


def _parse_tool_call(response: str) -> tuple[str, dict[str, Any]] | None:
    match = _TOOL_CALL_RE.search(response)
    if not match:
        return _parse_json_tool_call(response.strip())

    inner = match.group(1).strip()
    parsed = _parse_json_tool_call(inner)
    if parsed is not None:
        return parsed

    function_match = _XML_FUNCTION_RE.search(inner)
    if not function_match:
        return None
    name = function_match.group(1).strip()
    body = function_match.group(2)
    args = {
        param_match.group(1).strip(): _coerce_parameter(param_match.group(2))
        for param_match in _XML_PARAM_RE.finditer(body)
    }
    return name, args


# ---------------------------------------------------------------------------
# Sample helpers
# ---------------------------------------------------------------------------

def _metadata(sample: Any) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _task_name(sample: Any) -> str:
    metadata = _metadata(sample)
    return str(metadata.get("task") or metadata.get("dataset") or "").lower()


def _default_allowed_tools(sample: Any) -> tuple[str, ...]:
    task = _task_name(sample)
    if task in {"browsecomp", "browsecomp-plus", "browsecomp_plus"}:
        return BROWSECOMP_ALLOWED_TOOLS
    if task in {"mmsearch", "simplevqa", "visual"}:
        return VISUAL_ALLOWED_TOOLS
    return DEFAULT_ALLOWED_TOOLS


def _allowed_tools(sample: Any) -> tuple[str, ...]:
    metadata = _metadata(sample)
    configured = metadata.get("allowed_tools") or os.getenv("SII_SLIME_ALLOWED_TOOLS")
    if isinstance(configured, str):
        names = [name.strip() for name in configured.split(",") if name.strip()]
    elif isinstance(configured, list):
        names = [str(name).strip() for name in configured if str(name).strip()]
    else:
        names = list(_default_allowed_tools(sample))
    if "final_answer" not in names:
        names.append("final_answer")
    return tuple(dict.fromkeys(names))


def _system_prompt(sample: Any, allowed_tools: tuple[str, ...]) -> str:
    metadata = _metadata(sample)
    if isinstance(metadata.get("system_prompt"), str) and metadata["system_prompt"].strip():
        template = metadata["system_prompt"].strip()
    elif "search" in allowed_tools or _task_name(sample) in {"browsecomp", "browsecomp-plus", "browsecomp_plus"}:
        template = BROWSECOMP_SYSTEM_PROMPT
    elif _task_name(sample) in {"mmsearch", "simplevqa", "visual"} or "visual_web_search" in allowed_tools:
        template = VISUAL_SYSTEM_PROMPT
    else:
        template = SYSTEM_PROMPT
    return template.format(tool_list=", ".join(f"`{name}`" for name in allowed_tools))


# ---------------------------------------------------------------------------
# Tokenizer / loss-mask helpers
# ---------------------------------------------------------------------------

def _chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _token_delta(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return (new_tokens, loss_mask) for the most recent message.

    For an assistant turn the new tokens count as model output (mask=1); for
    tool/user observations they are context (mask=0).

    Qwen-3.5's chat template re-renders earlier tokens depending on what comes
    later (e.g. tool blocks injected/expanded after a tool call), so we cannot
    rely on a strict string-prefix invariant. Instead we tokenize both and take
    the common-prefix token diff; anything that doesn't match between prev and
    curr at the prefix is treated as "lost" context (mask=0) so that the
    assistant output we care about (the suffix produced by this turn) is still
    isolated correctly. Loss-mask only ever applies to the truly new suffix.
    """
    curr_text = _chat_template(tokenizer, messages, tools, add_generation_prompt=False)
    if messages[-1]["role"] == "assistant":
        prev_text = _chat_template(tokenizer, messages[:-1], tools, add_generation_prompt=True)
        mask_value = 1
    else:
        prev_text = _chat_template(tokenizer, messages[:-1], tools, add_generation_prompt=False)
        mask_value = 0
    curr_ids = _encode(tokenizer, curr_text)
    prev_ids = _encode(tokenizer, prev_text)

    # Find longest common prefix length in token space.
    common = 0
    cap = min(len(curr_ids), len(prev_ids))
    while common < cap and curr_ids[common] == prev_ids[common]:
        common += 1

    new_tokens = curr_ids[common:]
    # If the previous tokenization extended *beyond* the common prefix (because
    # later context re-rendered earlier blocks differently), those tokens are
    # silently dropped from history; new_tokens still contains the full suffix
    # of the current rendering, which is what the trainer will see. We mark all
    # of new_tokens with the message-role-derived mask.
    return new_tokens, [mask_value] * len(new_tokens)


def _safe_json(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch through the agent framework tool registry (off the event loop)."""
    return await asyncio.to_thread(dispatch, name, arguments)


def _build_messages(question: str, system_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _strip_stop_text(text: str) -> str:
    return text.removesuffix("<|im_end|>").strip()


def _score(final_answer: str | None, label: str | None) -> dict[str, Any]:
    result = score_answer(final_answer, label)
    correct = result.get("correct")
    return {
        "exact": result.get("exact"),
        "f1": result.get("f1"),
        "correct": correct,
        "task_reward": 1.0 if correct is True else 0.0,
    }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

async def generate(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Multi-turn ReAct rollout against the agent's tool registry.

    Generates tokens through slime's SGLang router (so trainer sees real
    on-policy samples) and dispatches tool calls through ``tools.dispatch``
    so behaviour matches the SII-Agent at evaluation time.
    """
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    tokenizer = state.tokenizer
    allowed = _allowed_tools(sample)
    tools = tool_specs(allowed)
    metadata = _metadata(sample)
    raw_question = metadata.get("question")
    if isinstance(raw_question, str) and raw_question.strip():
        question = raw_question
    elif isinstance(sample.prompt, str):
        question = sample.prompt
    elif isinstance(sample.prompt, list):
        last_user = next(
            (m for m in reversed(sample.prompt) if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        if last_user is not None and isinstance(last_user.get("content"), str):
            question = last_user["content"]
        else:
            question = json.dumps(sample.prompt, ensure_ascii=False)
    else:
        question = json.dumps(sample.prompt, ensure_ascii=False)
    messages = _build_messages(question, _system_prompt(sample, allowed))
    prompt_text = _chat_template(tokenizer, messages, tools, add_generation_prompt=True)
    prompt_token_ids = _encode(tokenizer, prompt_text)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_steps = _env_int("SII_RL_MAX_STEPS", 6)
    max_turn_tokens = _env_int("SII_RL_MAX_TURN_TOKENS", 1024)
    max_observation_chars = _env_int("SII_RL_MAX_OBSERVATION_CHARS", 4000)
    max_repeats = _env_int("SII_RL_MAX_REPEATS", 2)
    max_total_response_len = int(sampling_params.get("max_new_tokens") or 8192)
    started = time.time()

    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    assistant_texts: list[str] = []
    tool_counts: dict[str, int] = {}
    seen_calls: dict[str, int] = {}
    redundant_call_count = 0
    final_answer: str | None = None
    stop_reason = "max_steps"
    parse_failures = 0

    try:
        for step in range(max_steps):
            if step == max_steps - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": "This is the final step. Call final_answer now with your best concise answer.",
                    }
                )
                delta, mask = _token_delta(tokenizer, messages, tools)
                response_token_ids.extend(delta)
                loss_mask.extend(mask)

            text_input = _chat_template(tokenizer, messages, tools, add_generation_prompt=True)
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_new_tokens"] = min(
                int(turn_sampling_params.get("max_new_tokens") or max_turn_tokens),
                max_turn_tokens,
            )
            payload = {
                "text": text_input,
                "sampling_params": turn_sampling_params,
            }
            output = await post(url, payload)
            finish_reason = output.get("meta_info", {}).get("finish_reason", {})
            if finish_reason.get("type") == "abort":
                stop_reason = "abort"
                sample.status = Sample.Status.ABORTED
                break

            response = _strip_stop_text(str(output.get("text") or ""))
            messages.append({"role": "assistant", "content": response})
            assistant_texts.append(response)
            delta, mask = _token_delta(tokenizer, messages, tools)
            response_token_ids.extend(delta)
            loss_mask.extend(mask)
            if len(response_token_ids) > max_total_response_len:
                stop_reason = "truncated"
                sample.status = Sample.Status.TRUNCATED
                break

            parsed_call = _parse_tool_call(response)
            if parsed_call is None:
                parse_failures += 1
                if step == max_steps - 1 and response:
                    final_answer = response
                    stop_reason = "final_text"
                    sample.status = Sample.Status.COMPLETED
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "You did not call a function. Call one available function or final_answer.",
                    }
                )
                delta, mask = _token_delta(tokenizer, messages, tools)
                response_token_ids.extend(delta)
                loss_mask.extend(mask)
                continue

            name, call_args = parsed_call
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if name not in allowed:
                tool_result = f"ERROR: tool '{name}' is not allowed. Allowed tools: {list(allowed)}"
            elif name == "final_answer":
                final_answer = str(call_args.get("answer") or "").strip()
                if final_answer:
                    stop_reason = "final"
                    sample.status = Sample.Status.COMPLETED
                    break
                tool_result = "ERROR: final_answer requires a non-empty 'answer'."
            else:
                signature = f"{name}|{_safe_json(call_args)}"
                seen_calls[signature] = seen_calls.get(signature, 0) + 1
                if seen_calls[signature] > max_repeats:
                    redundant_call_count += 1
                    tool_result = (
                        f"NOTICE: You already called {name} with these args. "
                        "Use a different query or call final_answer."
                    )
                else:
                    tool_result = await _dispatch_tool(name, call_args)

            messages.append({"role": "tool", "name": name, "content": tool_result[:max_observation_chars]})
            delta, mask = _token_delta(tokenizer, messages, tools)
            response_token_ids.extend(delta)
            loss_mask.extend(mask)
            if len(response_token_ids) > max_total_response_len:
                stop_reason = "truncated"
                sample.status = Sample.Status.TRUNCATED
                break
        else:
            sample.status = Sample.Status.TRUNCATED
    except Exception as exc:  # noqa: BLE001
        logger.exception("SII RL rollout failed for sample index=%s", sample.index)
        stop_reason = f"error:{type(exc).__name__}"
        sample.status = Sample.Status.FAILED

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED if final_answer is None else Sample.Status.COMPLETED

    tokens = prompt_token_ids + response_token_ids
    if len(loss_mask) != len(response_token_ids):
        raise ValueError(f"loss_mask length {len(loss_mask)} != response_length {len(response_token_ids)}")

    scoring = _score(final_answer, sample.label)
    non_final_tool_calls = sum(count for name, count in tool_counts.items() if name != "final_answer")
    metadata = dict(sample.metadata or {})
    metadata.update(
        {
            "allowed_tools": list(allowed),
            "elapsed": time.time() - started,
            "final_answer": final_answer,
            "stop_reason": stop_reason,
            "task_reward": scoring["task_reward"],
            "score": scoring,
            "tool_call_counts": tool_counts,
            "non_final_tool_calls": non_final_tool_calls,
            "redundant_call_count": redundant_call_count,
            "parse_failures": parse_failures,
            "messages": messages,
        }
    )

    sample.prompt = prompt_text
    sample.tokens = tokens
    sample.response = "\n".join(text for text in assistant_texts if text)
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.metadata = metadata
    sample.reward = None
    return sample


# ---------------------------------------------------------------------------
# Reward (pure GRPO — no teacher, no remote RM)
# ---------------------------------------------------------------------------

def _compute_reward(metadata: dict[str, Any]) -> tuple[float, dict[str, float]]:
    weights = REWARD_WEIGHTS
    task_reward = float(metadata.get("task_reward") or 0.0)
    non_final_tool_calls = int(metadata.get("non_final_tool_calls") or 0)
    redundant_call_count = int(metadata.get("redundant_call_count") or 0)
    stop_reason = str(metadata.get("stop_reason") or "")

    components: dict[str, float] = {"task": task_reward * weights["task_correct"]}

    # Tool usage: rewards exploring at least one tool, penalises hallucinating
    # an answer without ever calling a tool.
    if non_final_tool_calls >= 1:
        components["tool_use"] = weights["tool_use_bonus"]
    else:
        components["tool_use"] = -weights["no_tool_penalty"]

    # Format: heavy penalty for never producing a final answer; small reward
    # for cleanly invoking `final_answer`; minor reward for a text-only final.
    if stop_reason == "final":
        components["format"] = weights["final_called_bonus"]
    elif stop_reason == "final_text":
        components["format"] = weights["final_text_bonus"]
    elif stop_reason in {"max_steps", "truncated", "abort"} or not metadata.get("final_answer"):
        components["format"] = -weights["format_penalty"]
    else:
        components["format"] = 0.0

    # Anti-redundancy: this is the main behavioural target of the run.
    components["redundant"] = -weights["redundant_penalty"] * redundant_call_count

    # Step efficiency: only punish when the model overshoots the budget.
    excess_steps = max(0, non_final_tool_calls - int(weights["step_budget"]))
    components["step"] = -weights["step_penalty"] * excess_steps

    total = sum(components.values())
    clipped = max(weights["reward_clip_low"], min(weights["reward_clip_high"], total))
    components["raw_total"] = total
    components["clipped_total"] = clipped
    return clipped, components


async def shaped_reward_rm(args: Namespace, sample: Any, **kwargs: Any) -> float:
    """Pure rule-based reward for GRPO.

    Combines task correctness with light tool-use shaping. The reward
    components are also stored in ``sample.metadata['reward_components']`` for
    later diagnostics.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward, components = _compute_reward(metadata)
    metadata["reward_components"] = components
    metadata["reward_total"] = reward
    sample.metadata = metadata
    return reward


async def task_reward_rm(args: Namespace, sample: Any, **kwargs: Any) -> float:
    """Strict task-correctness reward (no shaping). Useful for ablations."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return float(metadata.get("task_reward") or 0.0)
