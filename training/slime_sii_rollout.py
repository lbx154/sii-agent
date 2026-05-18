"""Custom slime rollout hooks for SII ReAct OPD."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from argparse import Namespace
from typing import Any

import torch

from agent.scoring import score_answer
from tools import dispatch, tool_specs

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_TOOLS = ("wiki_search", "web_search", "browser_open", "browser_open_many", "final_answer")
BROWSECOMP_ALLOWED_TOOLS = ("search", "final_answer")

SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what evidence is still needed.
2. Call exactly one available function, or call final_answer when ready.
3. Read the tool result and decide the next step.

Available tools for this run:
{tool_list}

Rules:
- Prefer wiki_search for encyclopedic questions and web_search when wiki_search is insufficient.
- Use browser_open for one source URL and browser_open_many for several independent source URLs.
- Do not repeat the same tool arguments; refine the query instead.
- Keep final_answer concise. Submit final_answer before the step budget is exhausted.
"""

BROWSECOMP_SYSTEM_PROMPT = """You are solving BrowseComp-Plus questions using a fixed local corpus.

Loop:
1. Search the fixed corpus with search.
2. Call final_answer when the answer is supported.

Available tools for this run:
{tool_list}

Rules:
- search returns the top 5 documents with docid, score, and a 512-token snippet.
- Build the answer only from BrowseComp-Plus corpus evidence.
- Cite supporting document ids in square brackets, e.g. [12345].
- Do not use open-web or Wikipedia tools for BrowseComp-Plus.
- Keep final_answer concise and include citations in the answer text when possible.
"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return metadata


def _task_name(sample: Any) -> str:
    metadata = _metadata(sample)
    return str(metadata.get("task") or metadata.get("dataset") or "").lower()


def _default_allowed_tools(sample: Any) -> tuple[str, ...]:
    task = _task_name(sample)
    if task in {"browsecomp", "browsecomp-plus", "browsecomp_plus"}:
        return BROWSECOMP_ALLOWED_TOOLS
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
    else:
        template = SYSTEM_PROMPT
    return template.format(tool_list=", ".join(f"`{name}`" for name in allowed_tools))


def _chat_template(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
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
    curr = _chat_template(tokenizer, messages, tools, add_generation_prompt=False)
    if messages[-1]["role"] == "assistant":
        prev = _chat_template(tokenizer, messages[:-1], tools, add_generation_prompt=True)
        mask_value = 1
    else:
        prev = _chat_template(tokenizer, messages[:-1], tools, add_generation_prompt=False)
        mask_value = 0
    if not curr.startswith(prev):
        raise ValueError(
            f"chat template delta mismatch for role={messages[-1]['role']}: "
            f"prev_len={len(prev)} curr_len={len(curr)}"
        )
    new_tokens = _encode(tokenizer, curr[len(prev) :])
    return new_tokens, [mask_value] * len(new_tokens)


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


def _safe_json(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
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


async def generate(args: Namespace, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> Any:
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    tokenizer = state.tokenizer
    allowed = _allowed_tools(sample)
    tools = tool_specs(allowed)
    question = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
    messages = _build_messages(question, _system_prompt(sample, allowed))
    prompt_text = _chat_template(tokenizer, messages, tools, add_generation_prompt=True)
    prompt_token_ids = _encode(tokenizer, prompt_text)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    max_steps = int(os.getenv("SII_SLIME_MAX_STEPS", "5"))
    max_turn_tokens = int(os.getenv("SII_SLIME_MAX_TURN_TOKENS", "1024"))
    max_observation_chars = int(os.getenv("SII_SLIME_MAX_OBSERVATION_CHARS", "4000"))
    max_repeats = int(os.getenv("SII_SLIME_MAX_REPEATS", "2"))
    max_total_response_len = int(sampling_params.get("max_new_tokens") or 8192)
    started = time.time()

    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    assistant_texts: list[str] = []
    tool_counts: dict[str, int] = {}
    seen_calls: dict[str, int] = {}
    final_answer: str | None = None
    stop_reason = "max_steps"

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
                "return_logprob": True,
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
        logger.exception("SII slime rollout failed for sample index=%s", sample.index)
        stop_reason = f"error:{type(exc).__name__}"
        sample.status = Sample.Status.FAILED

    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.TRUNCATED if final_answer is None else Sample.Status.COMPLETED

    tokens = prompt_token_ids + response_token_ids
    if len(loss_mask) != len(response_token_ids):
        raise ValueError(f"loss_mask length {len(loss_mask)} != response_length {len(response_token_ids)}")

    scoring = _score(final_answer, sample.label)
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


async def teacher_logprob_rm(args: Namespace, sample: Any, **kwargs):
    from slime.rollout.on_policy_distillation import reward_func as teacher_reward_func

    if isinstance(sample, list):
        return await asyncio.gather(*[teacher_logprob_rm(args, item, **kwargs) for item in sample])
    return await teacher_reward_func(args, sample, **kwargs)


def post_process_rewards(args: Namespace, samples: list[Any], **kwargs):
    task_rewards: list[float] = []
    for sample in samples:
        reward = sample.get_reward_value(args)
        response_length = sample.response_length
        input_logprobs = reward["meta_info"]["input_token_logprobs"][1:]
        if len(input_logprobs) < response_length:
            raise ValueError(
                f"teacher logprobs shorter than response for sample={sample.index}: "
                f"{len(input_logprobs)} < {response_length}"
            )
        teacher_log_probs = torch.tensor(
            [item[0] for item in input_logprobs[-response_length:]],
            dtype=torch.float32,
        )
        if len(teacher_log_probs) != response_length:
            raise ValueError(
                f"teacher_log_probs length {len(teacher_log_probs)} != response_length {response_length}"
            )
        sample.teacher_log_probs = teacher_log_probs
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if os.getenv("SII_SLIME_USE_TASK_REWARD", "1").lower() in {"1", "true", "yes"}:
            task_rewards.append(float(metadata.get("task_reward") or 0.0))
        else:
            task_rewards.append(0.0)
    return task_rewards, task_rewards
