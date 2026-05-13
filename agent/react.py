"""Minimal ReAct loop with OpenAI-style tool calling, governed by a Harness."""
from __future__ import annotations
import json
from typing import Any

from .llm import chat
from tools import tool_specs, dispatch
from harness.controller import HarnessConfig, HarnessResult, StepGuard

DEFAULT_TOOLS = ("web_search", "wiki_search", "browse", "final_answer")


SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what you still need.
2. Call ONE tool (web_search / wiki_search / browse / final_answer).
3. Read the tool result and decide next step.

Rules:
- Use `web_search` for live facts; use `wiki_search` for encyclopedic facts or when web search is slow.
- Use `browse` to read a specific URL from search results.
- Cross-check at least one source for non-trivial claims.
- DO NOT repeat the same query / URL — refine instead.
- Prefer 1-3 tool calls. When you have a plausible answer, call `final_answer` with a concise answer.
- You MUST call `final_answer` before the step budget is exhausted.
- If stuck, simplify the query or pivot to a related search term.
"""


def _safe_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


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


def run_react(
    question: str,
    cfg: HarnessConfig | None = None,
    extra_system: str | None = None,
) -> HarnessResult:
    cfg = cfg or HarnessConfig()
    guard = StepGuard(cfg)
    res = HarnessResult()

    sys = SYSTEM_PROMPT + (f"\n\n[Memory & Reflection]\n{extra_system}" if extra_system else "")
    messages: list[dict] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question},
    ]
    allowed_tools = cfg.allowed_tools or DEFAULT_TOOLS

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
            resp = chat(
                messages,
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
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            }
        )
        res.steps = step + 1
        res.trajectory.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            if guard.note_no_tool():
                res.stop_reason = "loop:no_tool"
                break
            messages.append({"role": "user", "content": "You did not call a tool. Either call a tool or call `final_answer`."})
            continue
        guard.reset_no_tool()

        for tc in msg.tool_calls:
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
                return res

            if guard.note_tool(name, json.dumps(args, sort_keys=True)):
                tool_result = (
                    f"NOTICE: You already called {name} with these args multiple times. "
                    f"Try a different query, a different tool, or call final_answer."
                )
            else:
                tool_result = dispatch(name, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result[:8000],
                }
            )
            res.trajectory.append({"role": "tool", "name": name, "args": args, "content": tool_result[:1000]})

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
            resp = chat(
                messages,
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
            dumped_tool_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]
            res.trajectory.append(
                {"role": "assistant", "content": msg.content or "", "tool_calls": dumped_tool_calls}
            )
            for tc in msg.tool_calls or []:
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
    res.elapsed = guard.elapsed()
    return res
