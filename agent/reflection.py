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

The ORIGINAL_PROBLEM_PROMPT is the authoritative source for the answer target and required
constraints. Treat the trajectory as noisy: failed candidates may contain accidental attributes
that are not part of the original problem. In root_cause/corrective_strategy, never introduce a
new search constraint, demographic filter, industry filter, gender, nationality, company type,
date, or entity class unless it is explicitly present in ORIGINAL_PROBLEM_PROMPT or directly
verified by tool evidence as a required clue. If a failed candidate has an extra attribute not in
the original prompt, call it a spurious candidate attribute and tell the retry to drop it.

Decide whether a short retry is useful. A retry is useful only if the answer is missing,
is a refusal/uncertainty response, or is visibly unsupported/contradicted by retrieved evidence.
If the final_answer tool arguments/rationale contain self-correction such as "wait", "misread",
"maternal/paternal relation is different", or "I need to search/find", treat that as a visible
self-contradiction and set needs_retry to true with failure_mode "self_contradictory_final".
If the answer is concise and well supported, set needs_retry to false.

Output STRICT JSON with keys:
{
  "needs_retry": true/false,
  "confidence": "high|medium|low",
  "failure_mode": "...",          // e.g., "wrong_tool", "bad_query", "hallucinated_answer", "loop", "timeout"
  "root_cause": "...",            // 1-2 sentences
  "corrective_strategy": "...",   // concrete next-time action, optimized for few searches; use only original-prompt constraints and verified evidence
  "reusable_lesson": "...",       // <=200 chars, generic enough to help similar future tasks
  "skill_update": null | {
    "title": "...",               // 3-8 words summarizing the actual procedure, e.g. "Verify relation before answering"
    "tags": ["..."],              // general task/tool/reasoning tags
    "trigger": "...",             // narrow question phrase where the update applies
    "step": "...",                // one concrete imperative step to add for future similar cases
    "verifier": "...",            // one check that would catch this failure without gold
    "bad_pattern": "..."          // one answer/tool pattern to avoid
  }
}
Set skill_update to null unless you can propose a narrow, procedural skill useful beyond this exact question.
For skill_update.title, do NOT use generic labels such as "Reflection skill", "Reflection update", "Context first",
or a dataset name. The title must summarize the concrete skill content using an imperative/action phrase.
Never create a skill from hidden gold; use only visible trajectory evidence.
No markdown, no prose outside the JSON.
"""


GOLD_REFLECT_SYS = """You are a training-time memory writer for a retrieval agent.

You are given a question, the agent trajectory, the agent's final answer, and a hidden reference answer.
Use the reference answer only as supervision to diagnose why the agent's process recovered after an
incorrect attempt. Do not write a one-off memory that merely stores the reference answer, a benchmark
name, dataset name, or the exact example.

The ORIGINAL_PROBLEM_PROMPT is the authoritative source for the answer target and required
constraints. Treat failed trajectory candidates as noisy: do not turn their accidental attributes
into future search constraints unless the original prompt or verified evidence makes them required.
In corrective_strategy and skill_update.step, explicitly preserve original-prompt constraints and
drop spurious candidate attributes.

You MUST produce a high-quality reusable lesson and a non-null procedural skill_update. Optimize
the memory for future tasks with a similar reasoning/tool failure.

Output STRICT JSON with keys:
{
  "needs_retry": true,
  "confidence": "high|medium|low",
  "failure_mode": "...",          // concrete failure such as "wrong_relation", "bad_query", "missed_context", "overtrusted_memory"
  "root_cause": "...",            // 1-2 sentences explaining the process failure, not just the correct answer
  "corrective_strategy": "...",   // concrete next-time action, optimized for few searches/tools; use only original-prompt constraints and verified evidence
  "reusable_lesson": "...",       // <=240 chars, general and actionable; do not copy the reference answer
  "skill_update": {
    "title": "...",               // 3-8 words, concrete procedure title, e.g. "Verify Relation Direction First"
    "tags": ["..."],              // general task/tool/reasoning tags, no benchmark/dataset names
    "trigger": "...",             // narrow pattern where this skill applies
    "step": "...",                // one imperative step that would prevent this failure
    "verifier": "...",            // one non-gold check that would catch this failure next time
    "bad_pattern": "..."          // one reasoning/tool pattern to avoid
  }
}
No generic titles like "Reflection skill" or "Context first". No markdown or prose outside JSON.
Never use the words "gold", "benchmark", or a dataset name in any output field.
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
            reasoning = _clip(ev.get("reasoning_content", ""), per_assistant_chars)
            if reasoning:
                parts.append(f"#{i} ASSISTANT calls: {calls}\nreasoning: {reasoning}\ncontent: {content}")
            else:
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
    force_memory: bool = False,
    original_prompt: str | None = None,
) -> dict:
    original_problem_prompt = str(original_prompt if original_prompt is not None else question or "").strip()
    if not original_problem_prompt:
        original_problem_prompt = str(question or "").strip()
    agent_user_prompt = str(question or "").strip()
    metadata_lines = []
    if include_expected and expected is not None:
        metadata_lines.append(f"REFERENCE_ANSWER_FOR_TRAINING: {expected}")
    if correct is False and include_expected:
        metadata_lines.append("EXTERNAL_EVAL: previous answer was judged incorrect")
    system_prompt = GOLD_REFLECT_SYS if include_expected and (correct is False or force_memory) else REFLECT_SYS
    user = (
        "ORIGINAL_PROBLEM_PROMPT:\n"
        f"{original_problem_prompt}\n\n"
        "FULL_AGENT_USER_PROMPT:\n"
        f"{agent_user_prompt}\n\n"
        "Reflection rule: corrective_strategy must be grounded in ORIGINAL_PROBLEM_PROMPT plus verified "
        "tool evidence. Do not add filters from failed candidates unless the original prompt requires them.\n"
        + ("\n".join(metadata_lines) + "\n" if metadata_lines else "")
        + f"AGENT_FINAL_ANSWER: {result.final_answer}\n"
        f"STOP_REASON: {result.stop_reason}\n"
        f"STEPS: {result.steps}, TOOL_CALLS: {result.tool_calls}, ELAPSED: {result.elapsed:.1f}s\n\n"
        f"TRAJECTORY:\n{_summarize_traj(result.trajectory)}"
    )
    try:
        resp = chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=20000,
            timeout=30000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or "{}"
        finish_reason = str(getattr(resp.choices[0], "finish_reason", "") or "")
    except Exception as exc:  # noqa: BLE001
        return {
            "failure_mode": "reflection_error",
            "root_cause": f"reflection LLM call failed: {type(exc).__name__}: {exc}",
            "corrective_strategy": "Retry with a different query or tool path, avoid repeating failed calls, and submit the best-supported concise answer.",
            "reusable_lesson": "If reflection is unavailable, vary the search/tool path instead of repeating the failed trajectory.",
        }
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed["_reflection_finish_reason"] = finish_reason
            parsed["_reflection_raw_length"] = len(raw)
        return parsed
    except Exception:
        return {
            "failure_mode": "parse_error",
            "root_cause": raw[:4000],
            "corrective_strategy": "",
            "reusable_lesson": "",
            "_reflection_finish_reason": finish_reason,
            "_reflection_raw_length": len(raw),
        }
