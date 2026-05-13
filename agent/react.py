"""Minimal ReAct loop with OpenAI-style tool calling, governed by a Harness."""
from __future__ import annotations
import json
from typing import Any

from .llm import chat
from tools import tool_specs, dispatch
from harness.controller import HarnessConfig, HarnessResult, StepGuard


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
- When confident, call `final_answer` with a concise answer.
- If stuck, simplify the query or pivot to a related search term.
"""


def _safe_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


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
    specs = tool_specs()

    for step in range(cfg.max_steps):
        if guard.time_left() <= 0:
            res.stop_reason = "timeout"
            break
        try:
            resp = chat(messages, tools=specs)
        except Exception as e:  # noqa: BLE001
            res.stop_reason = f"error: {type(e).__name__}: {e}"
            break

        msg = resp.choices[0].message
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

            if name == "final_answer":
                res.final_answer = args.get("answer", "").strip()
                res.rationale = args.get("rationale", "")
                res.stop_reason = "final"
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

    if not res.stop_reason:
        res.stop_reason = "max_steps"
    res.elapsed = guard.elapsed()
    return res
