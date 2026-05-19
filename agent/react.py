"""Minimal ReAct loop with OpenAI-style tool calling, governed by a Harness."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .llm import chat
from tools import TOOL_REGISTRY, tool_specs, dispatch
from harness.controller import HarnessConfig, HarnessResult, StepGuard
from memory.short_term import ShortTermMemory

BENCHMARK_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "final_answer",
)
VISUAL_TOOLS = (
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "browser_open",
    "browser_open_many",
    "final_answer",
)
RICH_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "final_answer",
)
MEMORY_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "memory_search",
    "memory_stats",
    "memory_list",
    "memory_get",
    "memory_create",
    "memory_update",
    "memory_delete",
    "final_answer",
)
WRITABLE_MEMORY_TOOLS = {"memory_create", "memory_update", "memory_delete"}
TRAIN_ONLY_TOOLS = WRITABLE_MEMORY_TOOLS
TOOL_PROFILES = {
    "benchmark": BENCHMARK_TOOLS,
    "default": BENCHMARK_TOOLS,
    "visual": VISUAL_TOOLS,
    "rich": RICH_TOOLS,
    "full": RICH_TOOLS,
    "memory": MEMORY_TOOLS,
    "self_retrieval": MEMORY_TOOLS,
}
DEFAULT_TOOLS = BENCHMARK_TOOLS
_BROWSER_EXTRACT_TOOLS = {"browser_open", "browser_open_many", "browser_text", "browser_click", "browser_type"}
_RESEARCH_TOOLS = {
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "reverse_image_search",
}


SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what you still need.
2. Call exactly ONE available tool.
3. Read the tool result and decide next step.

Available tools for this run:
{tool_list}

Rules:
{tool_rules}
- Cross-check at least one source for non-trivial claims. If the user provides a `Provided context` section, treat that context as the primary source and do not search only to cross-check facts already present there.
- DO NOT repeat the same query / URL — refine instead.
- For distinctive phrases or unusual clue wording, run an early exact quoted-phrase search instead of only broad paraphrases.
- Answer in the same language as the user question when possible; for Chinese questions, prefer the common Chinese name/phrase over an English alias.
- Prefer 1-3 evidence tool calls. Training memory CRUD calls are allowed when the rules require them.
- When you have a plausible answer and have completed required memory actions, call `final_answer` with a concise answer.
- You MUST call `final_answer` before the step budget is exhausted.
- If stuck, simplify the query or pivot to a related search term.
"""


def _shell_tool_enabled() -> bool:
    return os.getenv("SII_AGENT_ENABLE_SHELL_TOOL", "").strip().lower() in {"1", "true", "yes"}


def _runtime_mode() -> str:
    mode = os.getenv("SII_AGENT_RUNTIME_MODE", "train").strip().lower()
    return mode if mode in {"train", "test"} else "train"


def _profile_tools() -> tuple[str, ...]:
    profile = os.getenv("SII_AGENT_TOOL_PROFILE", "benchmark").strip().lower()
    if profile == "all":
        tools = tuple(TOOL_REGISTRY)
    else:
        tools = TOOL_PROFILES.get(profile, BENCHMARK_TOOLS)
    tools = tuple(name for name in tools if name != "verify")
    if _runtime_mode() == "test":
        tools = tuple(name for name in tools if name not in TRAIN_ONLY_TOOLS)
    if _shell_tool_enabled() and profile in {"self_retrieval", "memory", "rich", "full", "all"} and "bash_exec" not in tools:
        tools = (*tools, "bash_exec")
    if not _shell_tool_enabled():
        tools = tuple(name for name in tools if name != "bash_exec")
    return tools


def _tool_rules(allowed_tools: tuple[str, ...]) -> str:
    tools = set(allowed_tools)
    rules: list[str] = []
    if "memory_search" in tools:
        rules.append(
            "- Memory is an active training resource. Early in a train-mode task, generate 2-6 focused query phrases yourself (entities, relation words, task pattern, likely failure mode), then call `memory_search` with both the full question and `queries`. It returns compressed actionable guidance plus compact record ids; use memory_get only when you need full content for update/delete."
        )
    if "memory_search" in tools and _runtime_mode() == "test":
        rules.append("- Runtime mode is `test`: memory is read-only; retrieve useful records but do not create, update, or delete memory.")
    if "memory_create" in tools:
        rules.append(
            "- You are in train mode: improving memory is part of the task, not optional bookkeeping. "
            "Do not wait for the final step: before `final_answer`, call `memory_create` when this question teaches a reusable lesson or procedural skill. "
            "If `memory_search` exposed stale, wrong, duplicate, or harmful memory, call `memory_update` or `memory_delete` on that concrete id. "
            "For internal training-check rejections, do not create failure/reflection memory immediately; revise the answer first, and the harness will persist recovery memory only if the corrected answer verifies. "
            "Skip CRUD when the retrieved memory is good and the current run teaches nothing reusable."
        )
    if "web_search" in tools:
        rules.append("- Use `web_search` for live facts through the configured search-proxy or direct Serper fallback; it returns top snippets, not full pages.")
    if "wiki_search" in tools:
        rules.append("- Use `wiki_search` and `wiki_page` for offline encyclopedic facts when available.")
    if "browser_open" in tools:
        rules.append("- Use `browser_open` to read one promising URL; it reads the full page and returns query-focused excerpts when possible.")
    if "browser_open_many" in tools:
        rules.append("- Use `browser_open_many` to read several independent source URLs concurrently.")
    if "visual_web_search" in tools:
        rules.append(
            "- For visual factual questions with an image path or image URL, prefer `visual_web_search` first when available. It gathers evidence but does not produce the final answer."
        )
    if "image_to_search_queries" in tools or "image_to_text" in tools or "reverse_image_search" in tools:
        rules.append("- For image questions, combine visible/OCR clues with search evidence; do not treat the first visual entity guess as proven.")
    if "reverse_image_search" in tools:
        rules.append("- When calling `reverse_image_search`, pass the user question as `query` so it can fall back to text search if image upload/lens fails.")
    if "browser_click" in tools or "browser_type" in tools:
        rules.append("- Use browser interaction tools only when a page needs interaction after opening.")
    if "bash_exec" in tools:
        rules.append("- Use `bash_exec` only for explicit command-line inspection; prefer specialized search/memory/browser tools.")
    return "\n".join(rules) if rules else "- Use the available tools only when they directly help answer the question."


def _system_prompt(allowed_tools: tuple[str, ...], extra_system: str | None) -> str:
    sys = SYSTEM_PROMPT.format(
        tool_list=", ".join(f"`{name}`" for name in allowed_tools),
        tool_rules=_tool_rules(allowed_tools),
    )
    if "memory_search" in set(allowed_tools):
        overall = _overall_memory_guidance()
        if overall:
            sys += f"\n\n[Overall Memory Guidance]\n{overall}"
    if extra_system:
        sys += f"\n\n[Memory & Reflection]\n{extra_system}"
    return sys


def _clip_prompt_text(text: object, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def _overall_memory_guidance() -> str:
    if os.getenv("SII_MEMORY_OVERALL_IN_PROMPT", "1").strip().lower() in {"0", "false", "no"}:
        return ""
    root = Path(os.getenv("SII_AGENT_MEMORY_ROOT", os.getenv("MEMORY_ROOT", "logs/memory")))
    path = root / "overall.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    try:
        configured_limit = int(os.getenv("SII_MEMORY_OVERALL_PROMPT_MAX_CHARS", "1600"))
    except ValueError:
        configured_limit = 1600
    max_chars = max(400, min(2400, configured_limit))
    lines = [
        "Use this as process guidance only; it is not evidence for the current answer.",
        _clip_prompt_text(data.get("overall"), max_chars),
    ]
    categories = data.get("categories")
    if isinstance(categories, list):
        for item in categories[:6]:
            if not isinstance(item, dict):
                continue
            pattern = _clip_prompt_text(item.get("pattern"), 80)
            guidance = _clip_prompt_text(item.get("guidance"), 220)
            if pattern and guidance:
                lines.append(f"- {pattern}: {guidance}")
    avoid = data.get("avoid")
    if isinstance(avoid, list) and avoid:
        lines.append("Avoid: " + "; ".join(_clip_prompt_text(item, 140) for item in avoid[:4] if item))
    rendered = "\n".join(line for line in lines if line).strip()
    return _clip_prompt_text(rendered, max_chars)


def _internal_gold_verify_enabled(expected: str | None) -> bool:
    if _runtime_mode() != "train" or expected is None:
        return False
    return os.getenv("SII_AGENT_ENABLE_GOLD_VERIFY", "1").strip().lower() not in {"0", "false", "no"}


def _browser_extract_query(question: str) -> str:
    text = str(question or "").strip()
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    text = re.split(
        r"\n\s*\n|\s+Use the relevance-ranked\b|\s+Use the relevant\b|\s+Provided context\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(text.split())[:1000]


def _with_browser_extract_query(name: str, args: dict[str, Any], question: str) -> dict[str, Any]:
    if name not in _BROWSER_EXTRACT_TOOLS or str(args.get("extract_query") or "").strip():
        return args
    extract_query = _browser_extract_query(question)
    if not extract_query:
        return args
    updated = dict(args)
    updated["extract_query"] = extract_query
    return updated


def _tool_count(counts: dict[str, int], names: set[str]) -> int:
    return sum(int(counts.get(name) or 0) for name in names)


def _safe_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _coerce_parameter_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    parsed = _safe_json(text)
    if parsed is not None:
        return parsed
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def _payload_to_tool_call(payload: Any, call_id: str) -> Any | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    name = str(payload.get("name") or function.get("name") or "").strip()
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters")
    if arguments is None:
        arguments = function.get("arguments")
    if not name or arguments is None:
        return None
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        arguments_text = json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments_text),
        type="function",
        _synthetic=True,
    )


def _parse_content_tool_calls(content: str) -> list[Any]:
    calls: list[Any] = []
    seen_spans: list[tuple[int, int]] = []
    parsed_content = _safe_json((content or "").strip())
    if isinstance(parsed_content, dict):
        parsed_items = parsed_content.get("tool_calls") if isinstance(parsed_content.get("tool_calls"), list) else [parsed_content]
    elif isinstance(parsed_content, list):
        parsed_items = parsed_content
    else:
        parsed_items = []
    for payload in parsed_items:
        call = _payload_to_tool_call(payload, f"content-tool-call-{len(calls)}")
        if call is not None:
            calls.append(call)
    for i, match in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)):
        payload = _safe_json(match.group(1))
        call = _payload_to_tool_call(payload, f"content-tool-call-{len(calls)}")
        if call is None:
            continue
        calls.append(call)
        seen_spans.append(match.span())
    qwen_pattern = re.compile(
        r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    for match in qwen_pattern.finditer(content or ""):
        if any(start <= match.start() < end for start, end in seen_spans):
            continue
        name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for param in re.finditer(r"<parameter=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
            args[param.group(1).strip()] = _coerce_parameter_value(param.group(2))
        calls.append(
            SimpleNamespace(
                id=f"content-tool-call-{len(calls)}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                type="function",
                _synthetic=True,
            )
        )
    return calls


def _split_inline_reasoning(content: str) -> tuple[str, str]:
    text = content or ""
    reasoning_parts = [
        match.group(1).strip()
        for match in re.finditer(r"<think\b[^>]*>(.*?)</think>", text, flags=re.I | re.S)
        if match.group(1).strip()
    ]
    visible = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S).strip()
    return visible, "\n\n".join(reasoning_parts)


def _dump_tool_call(tc: Any) -> dict:
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def _message_reasoning(msg: Any) -> str:
    for name in ("reasoning_content", "reasoning"):
        value = getattr(msg, name, None)
        if value:
            return str(value)
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        for name in ("reasoning_content", "reasoning"):
            value = extra.get(name)
            if value:
                return str(value)
    if hasattr(msg, "model_dump"):
        try:
            data = msg.model_dump()
        except Exception:  # noqa: BLE001
            data = {}
        if isinstance(data, dict):
            for name in ("reasoning_content", "reasoning"):
                value = data.get(name)
                if value:
                    return str(value)
    return ""


def _forced_final_tool_choice() -> dict:
    return {"type": "function", "function": {"name": "final_answer"}}


def _apply_final_tool_call(res: HarnessResult, tc: Any) -> bool:
    if tc.function.name != "final_answer":
        return False
    args = _safe_json(tc.function.arguments or "{}")
    if args is None:
        return False
    answer, _ = _split_inline_reasoning(str(args.get("answer", "")))
    answer = answer.strip()
    if not answer:
        return False
    res.final_answer = answer
    rationale, _ = _split_inline_reasoning(str(args.get("rationale", "")))
    res.rationale = rationale
    res.stop_reason = "final"
    return True


def _messages_with_short_memory(
    messages: list[dict],
    memory: ShortTermMemory | None,
    active_tools: tuple[str, ...],
) -> list[dict]:
    if memory is None:
        return messages
    rendered = memory.render_for_prompt()
    if not rendered:
        return messages
    memory_msg = {"role": "user", "content": rendered}
    if active_tools == ("final_answer",) and messages:
        return messages[:-1] + [memory_msg, messages[-1]]
    return messages + [memory_msg]


def run_react(
    question: str,
    cfg: HarnessConfig | None = None,
    extra_system: str | None = None,
    expected: str | None = None,
    task: str | None = None,
) -> HarnessResult:
    cfg = cfg or HarnessConfig()
    guard = StepGuard(cfg)
    res = HarnessResult()
    short_memory = ShortTermMemory(question, max_chars=cfg.short_memory_max_chars) if cfg.use_short_memory else None

    allowed_tools = cfg.allowed_tools or _profile_tools()
    allowed_tools = tuple(name for name in allowed_tools if name != "verify")
    if _runtime_mode() == "test":
        allowed_tools = tuple(name for name in allowed_tools if name not in TRAIN_ONLY_TOOLS)
    internal_verify_enabled = _internal_gold_verify_enabled(expected)
    verify_token = None
    if internal_verify_enabled:
        from tools.verify import set_verify_context

        verify_token = set_verify_context(question=question, expected=expected, task=task)

    def reset_verify() -> None:
        nonlocal verify_token
        if verify_token is not None:
            from tools.verify import reset_verify_context

            reset_verify_context(verify_token)
            verify_token = None

    sys = _system_prompt(allowed_tools, extra_system)
    messages: list[dict] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question},
    ]
    has_verified = not internal_verify_enabled
    budget_messages_sent: set[str] = set()

    def append_budget_message_once(key: str, content: str) -> None:
        if key in budget_messages_sent:
            return
        budget_messages_sent.add(key)
        messages.append({"role": "user", "content": content})
        res.trajectory.append({"role": "system", "name": "budget_notice", "content": content})

    def record_verify_result(tool_result: str) -> dict | None:
        nonlocal has_verified
        parsed = _safe_json(tool_result)
        if isinstance(parsed, dict) and isinstance(parsed.get("correct"), bool):
            if parsed["correct"]:
                has_verified = True
        return parsed

    def internal_verify(answer: str, rationale: str = "") -> tuple[str, dict | None]:
        from tools.verify import verify

        verify_result = verify(answer, rationale)
        parsed_verify = record_verify_result(verify_result)
        verify_row = dict(parsed_verify) if isinstance(parsed_verify, dict) else {"raw": verify_result}
        verify_row.setdefault("answer", answer)
        verify_row.setdefault("rationale", rationale)
        res.internal_verify_results.append(verify_row)
        res.tool_call_counts["internal_verify"] = res.tool_call_counts.get("internal_verify", 0) + 1
        res.trajectory.append(
            {
                "role": "system",
                "name": "internal_verify",
                "args": {"answer": answer, "rationale": rationale},
                "content": verify_result,
            }
        )
        return verify_result, parsed_verify

    for step in range(cfg.max_steps):
        active_tools = allowed_tools
        pending_train_actions = internal_verify_enabled and not has_verified
        research_calls = _tool_count(res.tool_call_counts, _RESEARCH_TOOLS)
        web_search_calls = int(res.tool_call_counts.get("web_search") or 0)
        if not pending_train_actions:
            if cfg.max_research_tool_calls > 0 and research_calls >= cfg.max_research_tool_calls:
                active_tools = ("final_answer",)
                append_budget_message_once(
                    "research_cap",
                    (
                        f"You have used the evidence tool budget ({research_calls}/{cfg.max_research_tool_calls}). "
                        "Stop searching and call `final_answer` now with the best concise answer supported by the "
                        "evidence already gathered."
                    ),
                )
            else:
                if cfg.max_web_search_calls > 0 and web_search_calls >= cfg.max_web_search_calls:
                    if "web_search" in active_tools:
                        active_tools = tuple(name for name in active_tools if name != "web_search")
                    append_budget_message_once(
                        "web_search_cap",
                        (
                            f"You have already used {web_search_calls} web_search calls. Stop broad web searching; "
                            "use a targeted page/wiki/image check only if essential, otherwise call `final_answer`."
                        ),
                    )
                if cfg.synthesize_after_tool_calls > 0 and research_calls >= cfg.synthesize_after_tool_calls:
                    append_budget_message_once(
                        "synthesize_notice",
                        (
                            f"You have used {research_calls} evidence tool calls. Converge now: identify the strongest "
                            "candidate, run at most one targeted verification if needed, then call `final_answer`."
                        ),
                    )
        if step == cfg.max_steps - 1 and not pending_train_actions:
            active_tools = ("final_answer",)
            messages.append(
                {
                    "role": "user",
                    "content": "This is the final step. Submit your best concise answer now with `final_answer`.",
                }
            )
        elif step == cfg.max_steps - 1 and pending_train_actions:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is the final training step. If no candidate has been checked yet, call `final_answer` "
                        "with your current best answer so the training harness can verify it internally. If a prior "
                        "answer was rejected, use that feedback to revise the answer and call `final_answer` again."
                    ),
                }
            )
        specs = tool_specs(active_tools)
        time_left = guard.time_left()
        if time_left <= 0:
            res.stop_reason = "timeout"
            break
        if time_left < cfg.min_llm_call_seconds:
            res.stop_reason = "timeout"
            break
        try:
            tool_choice: str | dict = "auto"
            if active_tools == ("final_answer",):
                tool_choice = _forced_final_tool_choice()
            call_messages = _messages_with_short_memory(messages, short_memory, active_tools)
            resp = chat(
                call_messages,
                tools=specs,
                max_tokens=cfg.max_llm_tokens,
                timeout=min(cfg.max_llm_call_seconds, time_left),
                tool_choice=tool_choice,
            )
        except Exception as e:  # noqa: BLE001
            res.stop_reason = f"error: {type(e).__name__}: {e}"
            break

        choice = resp.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
        res.finish_reasons[finish_reason] = res.finish_reasons.get(finish_reason, 0) + 1
        msg = choice.message
        visible_content, inline_reasoning = _split_inline_reasoning(msg.content or "")
        parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(visible_content)
        effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
        dumped_tool_calls = [_dump_tool_call(tc) for tc in effective_tool_calls]
        tool_calls_to_run = effective_tool_calls
        skipped_parallel_tool_calls: list[Any] = []
        if cfg.max_parallel_tool_calls > 0 and len(effective_tool_calls) > cfg.max_parallel_tool_calls:
            tool_calls_to_run = effective_tool_calls[: cfg.max_parallel_tool_calls]
            skipped_parallel_tool_calls = effective_tool_calls[cfg.max_parallel_tool_calls :]
        assistant_content = "" if parsed_content_tool_calls else visible_content
        reasoning_content = "\n\n".join(
            part for part in (_message_reasoning(msg).strip(), inline_reasoning) if part
        )
        assistant_message = {"role": "assistant", "content": assistant_content}
        if effective_tool_calls:
            assistant_message["tool_calls"] = dumped_tool_calls
        messages.append(assistant_message)
        res.steps = step + 1
        assistant_event = {"role": "assistant", "content": visible_content, "tool_calls": dumped_tool_calls}
        if reasoning_content:
            assistant_event["reasoning_content"] = reasoning_content
        res.trajectory.append(assistant_event)

        if not effective_tool_calls:
            content = visible_content.strip()
            if active_tools == ("final_answer",) and content:
                if internal_verify_enabled and not has_verified:
                    verify_result, parsed_verify = internal_verify(content, "text finalization on forced final step")
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "INTERNAL_GOLD_VERIFY_RESULT_BEFORE_FINAL_ANSWER:\n"
                                    f"{verify_result}\n\n"
                                    "The candidate was not accepted. Reflect on the failure, then call `final_answer` "
                                    "with a corrected answer. Do not create lesson/skill memory yet; memory is written "
                                    "only if the corrected answer verifies as correct."
                                ),
                            }
                        )
                        continue
                res.final_answer = content
                res.rationale = "text finalization on forced final step"
                res.stop_reason = "final"
                if short_memory is not None:
                    res.short_memory_stats = short_memory.stats()
                res.elapsed = guard.elapsed()
                reset_verify()
                return res
            if guard.note_no_tool():
                res.stop_reason = "loop:no_tool"
                break
            if short_memory is not None:
                short_memory.observe_no_tool(content)
            messages.append({"role": "user", "content": "You did not call a tool. Either call a tool or call `final_answer`."})
            continue
        guard.reset_no_tool()

        for tc in tool_calls_to_run:
            name = tc.function.name
            args = _safe_json(tc.function.arguments or "{}")
            res.tool_calls += 1
            res.tool_call_counts[name] = res.tool_call_counts.get(name, 0) + 1

            if name not in active_tools:
                tool_result = (
                    f"ERROR: tool '{name}' is not allowed in this run. "
                    f"Allowed tools: {list(active_tools)}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                res.trajectory.append({"role": "tool", "name": name, "args": {}, "content": tool_result})
                continue

            if args is None:
                tool_result = (
                    "ERROR: invalid JSON arguments. Retry with valid JSON for this tool; "
                    "if submitting the final answer, keep it concise."
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                res.trajectory.append({"role": "tool", "name": name, "args": {}, "content": tool_result})
                continue
            args = _with_browser_extract_query(name, args, question)

            if name == "final_answer":
                if internal_verify_enabled and not has_verified:
                    answer = str(args.get("answer", "")).strip()
                    if not answer:
                        tool_result = "ERROR: final_answer requires a non-empty 'answer'. Retry with a concise answer."
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )
                        res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result})
                        continue
                    rationale = str(args.get("rationale") or "")
                    verify_result, parsed_verify = internal_verify(answer, rationale)
                    if isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True:
                        if not _apply_final_tool_call(res, tc):
                            tool_result = "ERROR: final_answer requires a non-empty 'answer'. Retry with a concise answer."
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                }
                            )
                            res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result})
                            continue
                        res.elapsed = guard.elapsed()
                        if short_memory is not None:
                            res.short_memory_stats = short_memory.stats()
                        reset_verify()
                        return res
                    tool_result = (
                        "INTERNAL_GOLD_VERIFY_RESULT_BEFORE_FINAL_ANSWER:\n"
                        f"{verify_result}\n\n"
                        "The candidate was not accepted. If correct=false, reflect on the failure, then call "
                        "`final_answer` with a corrected answer. Do not create lesson/skill memory yet; memory is "
                        "written only if the corrected answer verifies as correct."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
                    continue
                if not _apply_final_tool_call(res, tc):
                    tool_result = "ERROR: final_answer requires a non-empty 'answer'. Retry with a concise answer."
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
                    res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result})
                    continue
                res.elapsed = guard.elapsed()
                if short_memory is not None:
                    res.short_memory_stats = short_memory.stats()
                reset_verify()
                return res

            if guard.note_tool(name, json.dumps(args, sort_keys=True)):
                tool_result = (
                    f"NOTICE: You already called {name} with these args multiple times. "
                    f"Try a different query, a different tool, or call final_answer."
                )
            else:
                tool_result = dispatch(name, args)
                if short_memory is not None:
                    short_memory.observe_tool(name, args, tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )
            res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result})

        for tc in skipped_parallel_tool_calls:
            name = tc.function.name
            tool_result = (
                "NOTICE: extra parallel tool call skipped. This agent executes only one tool call per turn; "
                "read the executed tool result, then call one next tool or `final_answer`."
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )
            skipped_args = _safe_json(tc.function.arguments or "{}") or {}
            res.trajectory.append({"role": "tool", "name": name, "args": skipped_args, "content": tool_result})

        if guard.time_left() <= 0:
            res.stop_reason = "timeout"
            break

    if not res.final_answer and cfg.finalize_on_stop and guard.time_left() >= cfg.min_llm_call_seconds:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The tool budget is exhausted. Based only on the gathered evidence, "
                    "call `final_answer` now with the best concise answer. Return only the answer phrase/name "
                    "and optional citations/docids; do not add analysis or uncertainty caveats."
                ),
            }
        )
        try:
            call_messages = _messages_with_short_memory(messages, short_memory, ("final_answer",))
            resp = chat(
                call_messages,
                tools=tool_specs(("final_answer",)),
                tool_choice=_forced_final_tool_choice(),
                max_tokens=cfg.max_llm_tokens,
                timeout=min(cfg.max_llm_call_seconds, guard.time_left()),
            )
            choice = resp.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
            res.finish_reasons[f"forced_{finish_reason}"] = (
                res.finish_reasons.get(f"forced_{finish_reason}", 0) + 1
            )
            msg = choice.message
            visible_content, inline_reasoning = _split_inline_reasoning(msg.content or "")
            parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(visible_content)
            effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
            dumped_tool_calls = [_dump_tool_call(tc) for tc in effective_tool_calls]
            reasoning_content = "\n\n".join(
                part for part in (_message_reasoning(msg).strip(), inline_reasoning) if part
            )
            assistant_event = {"role": "assistant", "content": visible_content, "tool_calls": dumped_tool_calls}
            if reasoning_content:
                assistant_event["reasoning_content"] = reasoning_content
            res.trajectory.append(assistant_event)
            for tc in effective_tool_calls:
                if tc.function.name != "final_answer":
                    continue
                args = _safe_json(tc.function.arguments or "{}")
                if args is None:
                    continue
                answer, _ = _split_inline_reasoning(str(args.get("answer", "")))
                answer = answer.strip()
                if internal_verify_enabled and not has_verified and answer:
                    verify_result, parsed_verify = internal_verify(answer, str(args.get("rationale") or ""))
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        # No interaction budget remains here. Keep the rejected final answer for scoring;
                        # runner-level gold reflection will force lesson/skill persistence.
                        res.final_answer = answer
                        rationale, _ = _split_inline_reasoning(str(args.get("rationale") or ""))
                        res.rationale = rationale
                        res.stop_reason = "final"
                        res.tool_calls += 1
                        res.tool_call_counts["final_answer"] = res.tool_call_counts.get("final_answer", 0) + 1
                        break
                if _apply_final_tool_call(res, tc):
                    res.tool_calls += 1
                    res.tool_call_counts["final_answer"] = res.tool_call_counts.get("final_answer", 0) + 1
                    break
            content = visible_content.strip()
            if content and not res.final_answer:
                if internal_verify_enabled and not has_verified:
                    verify_result, parsed_verify = internal_verify(content, "forced finalization after tool budget")
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        # No interaction budget remains here. Keep the rejected final answer for scoring;
                        # runner-level gold reflection will force lesson/skill persistence.
                        res.final_answer = content
                        res.rationale = "forced finalization after tool budget"
                        res.stop_reason = "final"
                res.final_answer = content
                res.rationale = "forced finalization after tool budget"
                res.stop_reason = "final"
        except Exception as e:  # noqa: BLE001
            res.stop_reason = f"error: {type(e).__name__}: {e}"

    if not res.stop_reason:
        res.stop_reason = "max_steps"
    if short_memory is not None:
        res.short_memory_stats = short_memory.stats()
    res.elapsed = guard.elapsed()
    reset_verify()
    return res
