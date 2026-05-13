"""Reflection module — analyze a failed trajectory and produce an actionable lesson."""
from __future__ import annotations
import json
from .llm import chat
from harness.controller import HarnessResult


REFLECT_SYS = """You are a meta-reasoner reviewing a failed agent trajectory.

Output STRICT JSON with keys:
{
  "failure_mode": "...",          // e.g., "wrong_tool", "bad_query", "hallucinated_answer", "loop", "timeout"
  "root_cause": "...",            // 1-2 sentences
  "corrective_strategy": "...",   // concrete next-time action
  "reusable_lesson": "..."        // <=200 chars, generic enough to help similar future tasks
}
No markdown, no prose outside the JSON.
"""


def _summarize_traj(traj: list[dict], max_steps: int = 12) -> str:
    parts = []
    for i, ev in enumerate(traj[-max_steps:], 1):
        if ev.get("role") == "assistant":
            tcs = ev.get("tool_calls") or []
            calls = ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs) or "(no tool)"
            parts.append(f"#{i} ASSISTANT [calls: {calls}] {ev.get('content','')[:200]}")
        else:
            parts.append(f"#{i} TOOL[{ev.get('name')}]({ev.get('args')}) -> {str(ev.get('content',''))[:200]}")
    return "\n".join(parts)


def reflect(question: str, expected: str | None, result: HarnessResult) -> dict:
    user = (
        f"QUESTION: {question}\n"
        f"EXPECTED (gold, if any): {expected}\n"
        f"AGENT_FINAL_ANSWER: {result.final_answer}\n"
        f"STOP_REASON: {result.stop_reason}\n"
        f"STEPS: {result.steps}, TOOL_CALLS: {result.tool_calls}, ELAPSED: {result.elapsed:.1f}s\n\n"
        f"TRAJECTORY:\n{_summarize_traj(result.trajectory)}"
    )
    resp = chat(
        [{"role": "system", "content": REFLECT_SYS}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"failure_mode": "parse_error", "root_cause": raw[:200],
                "corrective_strategy": "", "reusable_lesson": ""}
