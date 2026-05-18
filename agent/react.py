"""Minimal ReAct loop with OpenAI-style tool calling, governed by a Harness."""
from __future__ import annotations
import json
import os
import re
from types import SimpleNamespace
from typing import Any

from .llm import chat
from tools import TOOL_REGISTRY, tool_specs, dispatch
from harness.controller import HarnessConfig, HarnessResult, StepGuard
from memory.short_term import ShortTermMemory

BENCHMARK_TOOLS = ("web_search", "wiki_search", "wiki_page", "browse", "browse_many", "final_answer")
VISUAL_TOOLS = (
    "visual_web_search",
    "image_to_text",
    "web_search",
    "wiki_search",
    "wiki_page",
    "browse",
    "browse_many",
    "image_search",
    "final_answer",
)
RICH_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browse",
    "browse_many",
    "image_search",
    "visual_web_search",
    "image_to_text",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "final_answer",
)
TOOL_PROFILES = {
    "benchmark": BENCHMARK_TOOLS,
    "default": BENCHMARK_TOOLS,
    "visual": VISUAL_TOOLS,
    "rich": RICH_TOOLS,
    "full": RICH_TOOLS,
}
DEFAULT_TOOLS = BENCHMARK_TOOLS


SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what you still need.
2. Call exactly ONE available tool.
3. Read the tool result and decide next step.

Available tools for this run:
{tool_list}

Rules:
- Use `web_search` for live facts; use `wiki_search` for encyclopedic facts or when web search is slow.
- Use `browse` to read one URL and `browse_many` to read several independent URLs concurrently.
- For visual factual questions with an image path or image URL, prefer `visual_web_search` first when available. It generates multiple visual/OCR hypotheses and verifies them against search evidence.
- Do not treat the first visual entity guess as proven. Compare candidates, use OCR/visible clues, and reject candidates that do not answer the exact question.
- Use `image_to_text` for focused OCR/caption follow-up; `image_search` is text-to-image search, not reverse image search.
- Use `browser_open` / `browser_open_many` / `browser_click` / `browser_type` only when a page needs JavaScript rendering or interaction.
- Cross-check at least one source for non-trivial claims. If the user provides a `Provided context` section, treat that context as the primary source and do not search only to cross-check facts already present there.
- DO NOT repeat the same query / URL — refine instead.
- Answer in the same language as the user question when possible; for Chinese questions, prefer the common Chinese name/phrase over an English alias.
- Prefer 1-3 tool calls. When you have a plausible answer, call `final_answer` with a concise answer.
- You MUST call `final_answer` before the step budget is exhausted.
- If stuck, simplify the query or pivot to a related search term.
"""


def _profile_tools() -> tuple[str, ...]:
    profile = os.getenv("SII_AGENT_TOOL_PROFILE", "benchmark").strip().lower()
    if profile == "all":
        return tuple(TOOL_REGISTRY)
    return TOOL_PROFILES.get(profile, BENCHMARK_TOOLS)


def _system_prompt(allowed_tools: tuple[str, ...], extra_system: str | None) -> str:
    sys = SYSTEM_PROMPT.format(tool_list=", ".join(f"`{name}`" for name in allowed_tools))
    if extra_system:
        sys += f"\n\n[Memory & Reflection]\n{extra_system}"
    return sys


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
    answer = str(args.get("answer", "")).strip()
    if not answer:
        return False
    res.final_answer = answer
    res.rationale = args.get("rationale", "")
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
) -> HarnessResult:
    cfg = cfg or HarnessConfig()
    guard = StepGuard(cfg)
    res = HarnessResult()
    short_memory = ShortTermMemory(question, max_chars=cfg.short_memory_max_chars) if cfg.use_short_memory else None

    allowed_tools = cfg.allowed_tools or _profile_tools()
    sys = _system_prompt(allowed_tools, extra_system)
    messages: list[dict] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question},
    ]

    for step in range(cfg.max_steps):
        active_tools = allowed_tools
        if step == cfg.max_steps - 1:
            active_tools = ("final_answer",)
            messages.append(
                {
                    "role": "user",
                    "content": "This is the final step. Submit your best concise answer now with `final_answer`.",
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
        parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(msg.content or "")
        effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
        dumped_tool_calls = [_dump_tool_call(tc) for tc in effective_tool_calls]
        assistant_content = "" if parsed_content_tool_calls else (msg.content or "")
        reasoning_content = _message_reasoning(msg)
        assistant_message = {"role": "assistant", "content": assistant_content}
        if effective_tool_calls:
            assistant_message["tool_calls"] = dumped_tool_calls
        messages.append(assistant_message)
        res.steps = step + 1
        assistant_event = {"role": "assistant", "content": msg.content or "", "tool_calls": dumped_tool_calls}
        if reasoning_content:
            assistant_event["reasoning_content"] = reasoning_content
        res.trajectory.append(assistant_event)

        if not effective_tool_calls:
            content = (msg.content or "").strip()
            if active_tools == ("final_answer",) and content:
                res.final_answer = content
                res.rationale = "text finalization on forced final step"
                res.stop_reason = "final"
                if short_memory is not None:
                    res.short_memory_stats = short_memory.stats()
                res.elapsed = guard.elapsed()
                return res
            if guard.note_no_tool():
                res.stop_reason = "loop:no_tool"
                break
            if short_memory is not None:
                short_memory.observe_no_tool(content)
            messages.append({"role": "user", "content": "You did not call a tool. Either call a tool or call `final_answer`."})
            continue
        guard.reset_no_tool()

        for tc in effective_tool_calls:
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

            if name == "final_answer":
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
                    "content": tool_result[:8000],
                }
            )
            res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result[:3000]})

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
            parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(msg.content or "")
            effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
            dumped_tool_calls = [_dump_tool_call(tc) for tc in effective_tool_calls]
            res.trajectory.append(
                {"role": "assistant", "content": msg.content or "", "tool_calls": dumped_tool_calls}
            )
            for tc in effective_tool_calls:
                if _apply_final_tool_call(res, tc):
                    res.tool_calls += 1
                    res.tool_call_counts["final_answer"] = res.tool_call_counts.get("final_answer", 0) + 1
                    break
            content = (msg.content or "").strip()
            if content and not res.final_answer:
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
    return res
