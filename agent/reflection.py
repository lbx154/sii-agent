"""Reflection module — review a trajectory and produce an actionable retry plan."""
from __future__ import annotations
import json
import re
from .llm import chat
from harness.controller import HarnessResult


REFLECT_SYS = """You are a meta-reasoner reviewing a retrieval-agent trajectory.

Judge only from the question, the agent answer, and the shown trajectory evidence.
Do not assume access to a hidden gold answer. Do not recommend withholding, privacy refusal,
or "do not answer" behavior unless the user explicitly asked for that.

Decide whether a short retry is useful. A retry is useful only if the answer is missing,
is a refusal/uncertainty response, or is visibly unsupported/contradicted by retrieved evidence.
If the answer is concise and well supported, set needs_retry to false.

Output STRICT JSON with keys:
{
  "needs_retry": true/false,
  "confidence": "high|medium|low",
  "failure_mode": "...",          // e.g., "wrong_tool", "bad_query", "hallucinated_answer", "loop", "timeout"
  "root_cause": "...",            // 1-2 sentences
  "corrective_strategy": "...",   // concrete next-time action, optimized for few searches
  "reusable_lesson": "..."        // <=200 chars, generic enough to help similar future tasks
}
No markdown, no prose outside the JSON.
"""


_TITLE_RE = re.compile(r"---\s*title:\s*(.*?)\s+date:", re.S)


def _json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _clip(text: object, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + " ..."


def _doc_title(snippet: str) -> str:
    match = _TITLE_RE.search(snippet or "")
    if match:
        return " ".join(match.group(1).split())
    return ""


def _summarize_tool_content(name: str, args: object, content: str, max_chars: int) -> str:
    parsed = _json_loads(content)
    if name == "search" and isinstance(parsed, list):
        lines = []
        for idx, item in enumerate(parsed[:5], 1):
            if not isinstance(item, dict):
                lines.append(f"  {idx}. {_clip(item, 220)}")
                continue
            snippet = str(item.get("snippet") or item.get("text") or "")
            title = str(item.get("title") or _doc_title(snippet))
            head = f"  {idx}. docid={item.get('docid', '')} score={item.get('score', '')}"
            if title:
                head += f" title={title}"
            lines.append(head)
            if snippet:
                lines.append(f"     snippet={_clip(snippet, 520)}")
        rendered = "\n".join(lines)
        return _clip(rendered, max_chars)
    return _clip(content, max_chars)


def _summarize_traj(
    traj: list[dict],
    max_chars: int = 24000,
    per_tool_chars: int = 2200,
    per_assistant_chars: int = 1600,
) -> str:
    parts: list[str] = []
    for i, ev in enumerate(traj, 1):
        if ev.get("role") == "assistant":
            tcs = ev.get("tool_calls") or []
            call_bits = []
            for tc in tcs:
                fn = tc.get("function", {})
                call_bits.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            calls = "; ".join(call_bits) or "(no tool)"
            content = _clip(ev.get("content", ""), per_assistant_chars)
            parts.append(f"#{i} ASSISTANT calls: {calls}\ncontent: {content}")
        else:
            name = str(ev.get("name") or "?")
            args = ev.get("args")
            content = _summarize_tool_content(name, args, str(ev.get("content", "")), per_tool_chars)
            parts.append(f"#{i} TOOL[{name}] args={args}\n{content}")
    rendered = "\n\n".join(parts)
    if len(rendered) <= max_chars:
        return rendered
    if len(parts) <= 4:
        return _clip(rendered, max_chars)
    head = parts[:2]
    tail: list[str] = []
    used = len("\n\n".join(head)) + 120
    for part in reversed(parts[2:]):
        part_len = len(part) + 2
        if used + part_len > max_chars:
            break
        tail.append(part)
        used += part_len
    tail.reverse()
    omitted = len(parts) - len(head) - len(tail)
    return "\n\n".join(head + [f"... omitted {omitted} middle trajectory events to fit token budget ..."] + tail)


def reflect(
    question: str,
    expected: str | None,
    result: HarnessResult,
    correct: bool | None = None,
    include_expected: bool = False,
) -> dict:
    metadata_lines = []
    if include_expected and expected is not None:
        metadata_lines.append(f"EXPECTED (gold): {expected}")
    if correct is False and include_expected:
        metadata_lines.append("EXTERNAL_EVAL: previous answer was judged incorrect")
    user = (
        f"QUESTION: {question}\n"
        + ("\n".join(metadata_lines) + "\n" if metadata_lines else "")
        + f"AGENT_FINAL_ANSWER: {result.final_answer}\n"
        f"STOP_REASON: {result.stop_reason}\n"
        f"STEPS: {result.steps}, TOOL_CALLS: {result.tool_calls}, ELAPSED: {result.elapsed:.1f}s\n\n"
        f"TRAJECTORY:\n{_summarize_traj(result.trajectory)}"
    )
    try:
        resp = chat(
            [{"role": "system", "content": REFLECT_SYS}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=20000,
            timeout=30000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as exc:  # noqa: BLE001
        return {
            "failure_mode": "reflection_error",
            "root_cause": f"reflection LLM call failed: {type(exc).__name__}: {exc}",
            "corrective_strategy": "Retry with a different query or tool path, avoid repeating failed calls, and submit the best-supported concise answer.",
            "reusable_lesson": "If reflection is unavailable, vary the search/tool path instead of repeating the failed trajectory.",
        }
    try:
        return json.loads(raw)
    except Exception:
        return {"failure_mode": "parse_error", "root_cause": raw[:200],
                "corrective_strategy": "", "reusable_lesson": ""}
