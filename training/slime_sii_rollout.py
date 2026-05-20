"""Custom slime rollout hooks for SII ReAct OPD."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import time
from argparse import Namespace
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import torch
import aiohttp

from agent.scoring import score_answer
from tools import dispatch, tool_specs

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_TOOLS = ("wiki_search", "web_search", "browser_open", "browser_open_many", "final_answer")
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
VLM_TOOLS = frozenset({"visual_web_search", "image_to_text", "image_to_search_queries"})
WIKI_TOOLS = frozenset({"wiki_search", "wiki_page"})

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

VISUAL_SYSTEM_PROMPT = """You are solving visual factual QA questions with local images and retrieval tools.

Loop:
1. Inspect the image or generate visual search queries.
2. Search/browse to verify external facts when needed.
3. Call final_answer when the answer is supported.

Available tools for this run:
{tool_list}

Rules:
- The user prompt contains an `Image path:` line for local image input.
- Prefer visual_web_search first when available; otherwise use image_to_text or image_to_search_queries before web/wiki search.
- Do not treat the first visual guess as proven. Verify named entities, dates, locations, and external facts with retrieval evidence.
- Keep final_answer concise.
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
_SEARCH_TOOL_NAMES = frozenset(
    {
        "search",
        "web_search",
        "wiki_search",
        "browsecomp_search",
        "visual_web_search",
        "reverse_image_search",
    }
)
_QUERY_TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿĀ-žḀ-ỿ]+", re.UNICODE)
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return metadata


def _task_name(sample: Any) -> str:
    metadata = _metadata(sample)
    return str(metadata.get("task") or metadata.get("dataset") or "").lower()


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def _vision_endpoint_configured(base_url: str | None = None) -> bool:
    if _truthy_env("SII_FORCE_ENABLE_VISION_TOOLS", "0"):
        return True
    base_url = (base_url or os.getenv("VISION_BASE_URL", "")).strip()
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return True
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wiki_index_available() -> bool:
    index_path = os.getenv("WIKI25_INDEX_PATH", "data/wiki25/wiki25_fts.sqlite").strip()
    return bool(index_path) and os.path.exists(index_path)


def _configure_rollout_vision_endpoint(args: Namespace) -> None:
    if not _truthy_env("SII_VISION_USE_ROLLOUT_ROUTER", "1"):
        return
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if not router_ip or not router_port:
        return
    base_url = f"http://{router_ip}:{router_port}/v1"
    os.environ["VISION_BASE_URL"] = base_url
    os.environ.setdefault("VISION_BACKEND", "vllm")
    os.environ.setdefault("VISION_API_KEY", os.getenv("VLLM_API_KEY", "EMPTY"))
    os.environ["VISION_MODEL"] = (
        os.getenv("SII_VISION_ROLLOUT_MODEL")
        or os.getenv("VISION_MODEL")
        or os.getenv("VLLM_MODEL")
        or "Qwen3.5-9B"
    )
    _vision_endpoint_configured.cache_clear()


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
    if (
        _truthy_env("SII_DISABLE_VISION_TOOLS_WHEN_UNAVAILABLE", "1")
        and any(name in VLM_TOOLS for name in names)
        and not _vision_endpoint_configured()
    ):
        names = [name for name in names if name not in VLM_TOOLS]
    if (
        _truthy_env("SII_DISABLE_WIKI_TOOLS_WHEN_UNAVAILABLE", "1")
        and any(name in WIKI_TOOLS for name in names)
        and not _wiki_index_available()
    ):
        names = [name for name in names if name not in WIKI_TOOLS]
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


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


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
        role = messages[-1]["role"]
        prefix_len = _common_prefix_len(prev, curr)
        if role == "assistant" or prefix_len == 0:
            raise ValueError(
                f"chat template delta mismatch for role={role}: "
                f"prev_len={len(prev)} curr_len={len(curr)} common_prefix_len={prefix_len}"
            )
        logger.warning(
            "chat template non-prefix delta for role=%s; using common-prefix fallback "
            "prev_len=%s curr_len=%s common_prefix_len=%s",
            role,
            len(prev),
            len(curr),
            prefix_len,
        )
        new_text = curr[prefix_len:]
    else:
        new_text = curr[len(prev) :]
    new_tokens = _encode(tokenizer, new_text)
    return new_tokens, [mask_value] * len(new_tokens)


def _append_delta(
    response_token_ids: list[int],
    loss_mask: list[int],
    delta: list[int],
    mask: list[int],
    max_total_response_len: int,
) -> bool:
    if len(delta) != len(mask):
        raise ValueError(f"delta/mask length mismatch: {len(delta)} != {len(mask)}")
    remaining = max_total_response_len - len(response_token_ids)
    if remaining <= 0:
        return True
    if len(delta) > remaining:
        response_token_ids.extend(delta[:remaining])
        loss_mask.extend(mask[:remaining])
        return True
    response_token_ids.extend(delta)
    loss_mask.extend(mask)
    return len(response_token_ids) >= max_total_response_len


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


def _search_query_for_call(name: str, arguments: dict[str, Any]) -> str | None:
    if name not in _SEARCH_TOOL_NAMES:
        return None
    query = arguments.get("query") or arguments.get("q")
    if query is None and name == "reverse_image_search":
        query = arguments.get("source")
    if query is None:
        return None
    query = " ".join(str(query).split())
    return query or None


def _query_token_set(query: str) -> frozenset[str]:
    tokens = {
        token.strip("'’.-_").lower()
        for token in _QUERY_TOKEN_RE.findall(query)
    }
    return frozenset(
        token
        for token in tokens
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    )


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    return await asyncio.to_thread(dispatch, name, arguments)


def _build_messages(question: str, system_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _sample_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role not in {"user", ""}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                text_parts = [
                    str(item.get("text") or "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_parts:
                    parts.append("\n".join(part for part in text_parts if part))
        if parts:
            return "\n\n".join(parts)
    return json.dumps(prompt, ensure_ascii=False)


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

    _configure_rollout_vision_endpoint(args)
    state = GenerateState(args)
    tokenizer = state.tokenizer
    allowed = _allowed_tools(sample)
    tools = tool_specs(allowed)
    question = _sample_prompt_text(sample.prompt)
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
    seen_searches: list[tuple[str, str, frozenset[str]]] = []
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
                if _append_delta(response_token_ids, loss_mask, delta, mask, max_total_response_len):
                    stop_reason = "truncated"
                    sample.status = Sample.Status.TRUNCATED
                    break

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
            if _append_delta(response_token_ids, loss_mask, delta, mask, max_total_response_len):
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
                if _append_delta(response_token_ids, loss_mask, delta, mask, max_total_response_len):
                    stop_reason = "truncated"
                    sample.status = Sample.Status.TRUNCATED
                    break
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
                    search_query = _search_query_for_call(name, call_args)
                    if search_query:
                        search_tokens = _query_token_set(search_query)
                        threshold = _float_env("SII_SEARCH_SIMILARITY_THRESHOLD", 0.82)
                        similar_limit = _int_env("SII_SEARCH_MAX_SIMILAR_REPEATS", 1)
                        similar = [
                            previous
                            for previous in seen_searches
                            if _similarity(search_tokens, previous[2]) >= threshold
                        ]
                        max_search_calls_per_tool = _int_env("SII_SEARCH_MAX_CALLS_PER_TOOL", 8)
                        max_search_calls_total = _int_env("SII_SEARCH_MAX_CALLS_TOTAL", 16)
                        total_search_calls = sum(tool_counts.get(tool_name, 0) for tool_name in _SEARCH_TOOL_NAMES)
                        if len(similar) >= similar_limit:
                            tool_result = (
                                f"NOTICE: This {name} query is too similar to an earlier query "
                                f"made via {similar[-1][0]} ({similar[-1][1]!r}). Synthesize from existing evidence, try a genuinely "
                                "different query, or call final_answer."
                            )
                        elif max_search_calls_per_tool > 0 and tool_counts.get(name, 0) > max_search_calls_per_tool:
                            tool_result = (
                                f"NOTICE: You have already used {name} {tool_counts[name]} times. "
                                "Stop searching with this tool; synthesize from evidence or call final_answer."
                            )
                        elif max_search_calls_total > 0 and total_search_calls > max_search_calls_total:
                            tool_result = (
                                f"NOTICE: You have already made {total_search_calls} search calls. "
                                "Stop searching; synthesize from evidence and call final_answer."
                            )
                        else:
                            seen_searches.append((name, search_query, search_tokens))
                            tool_result = await _dispatch_tool(name, call_args)
                    else:
                        tool_result = await _dispatch_tool(name, call_args)

            messages.append({"role": "tool", "name": name, "content": tool_result[:max_observation_chars]})
            delta, mask = _token_delta(tokenizer, messages, tools)
            if _append_delta(response_token_ids, loss_mask, delta, mask, max_total_response_len):
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
    sample.reward = scoring["task_reward"] if evaluation else None
    return sample


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _teacher_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=_float_env("SII_TEACHER_RM_TOTAL_TIMEOUT", 1800.0),
        connect=_float_env("SII_TEACHER_RM_CONNECT_TIMEOUT", 300.0),
        sock_connect=_float_env("SII_TEACHER_RM_CONNECT_TIMEOUT", 300.0),
        sock_read=_float_env("SII_TEACHER_RM_READ_TIMEOUT", 1800.0),
    )


def _task_reward(sample: Any) -> float:
    if os.getenv("SII_SLIME_USE_TASK_REWARD", "1").lower() not in {"1", "true", "yes"}:
        return 0.0
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return float(metadata.get("task_reward") or 0.0)


def _extract_teacher_log_probs(sample: Any, reward: dict[str, Any]) -> torch.Tensor:
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
    return teacher_log_probs


async def _teacher_logprob_one(args: Namespace, sample: Any, session: aiohttp.ClientSession) -> Any:
    from slime.rollout.on_policy_distillation import reward_func as teacher_reward_func
    from slime.utils.processing_utils import encode_image_for_rollout_engine

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    retries = max(0, _int_env("SII_TEACHER_RM_RETRIES", 5))
    for attempt in range(retries + 1):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                reward = await resp.json()
                sample.teacher_log_probs = _extract_teacher_log_probs(sample, reward)
                return _task_reward(sample)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt >= retries:
                logger.exception("teacher logprob request failed after %s attempts", attempt + 1)
                raise
            delay = min(2.0 * (attempt + 1), 30.0)
            logger.warning(
                "teacher logprob request failed (%s/%s): %s; retrying in %.1fs",
                attempt + 1,
                retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception:
            with contextlib.suppress(Exception):
                reward = await teacher_reward_func(args, sample)
                sample.teacher_log_probs = _extract_teacher_log_probs(sample, reward)
                return _task_reward(sample)
            raise

    raise RuntimeError("unreachable teacher logprob retry loop exit")


async def teacher_logprob_rm(args: Namespace, sample: Any, **kwargs):
    if isinstance(sample, list):
        concurrency = max(1, _int_env("SII_TEACHER_RM_CONCURRENCY", 2))
        semaphore = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession(timeout=_teacher_timeout()) as session:
            async def bounded(item: Any) -> Any:
                async with semaphore:
                    return await _teacher_logprob_one(args, item, session)

            return await asyncio.gather(*[bounded(item) for item in sample])

    async with aiohttp.ClientSession(timeout=_teacher_timeout()) as session:
        return await _teacher_logprob_one(args, sample, session)


def post_process_rewards(args: Namespace, samples: list[Any], **kwargs):
    task_rewards: list[float] = []
    for sample in samples:
        reward = sample.get_reward_value(args)
        if sample.teacher_log_probs is None:
            if not isinstance(reward, dict):
                raise ValueError(f"missing teacher_log_probs for sample={sample.index}")
            sample.teacher_log_probs = _extract_teacher_log_probs(sample, reward)
        task_reward = _task_reward(sample)
        sample.reward = task_reward
        task_rewards.append(task_reward)
    return task_rewards, task_rewards
