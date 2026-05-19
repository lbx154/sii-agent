"""Training-only gold-answer verification tool."""
from __future__ import annotations

import contextvars
import json
import os
import re
from typing import Any

from agent.llm import chat
from agent.scoring import score_answer


_VERIFY_CONTEXT: contextvars.ContextVar[dict[str, str | None]] = contextvars.ContextVar(
    "sii_verify_context",
    default={},
)


def _runtime_mode() -> str:
    mode = os.getenv("SII_AGENT_RUNTIME_MODE", "train").strip().lower()
    return mode if mode in {"train", "test"} else "train"


def set_verify_context(question: str, expected: str | None, task: str | None = None) -> contextvars.Token:
    return _VERIFY_CONTEXT.set({"question": question, "expected": expected, "task": task})


def reset_verify_context(token: contextvars.Token | None) -> None:
    if token is not None:
        _VERIFY_CONTEXT.reset(token)


def _json_obj(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


SEMANTIC_VERIFY_SYS = """You are a strict training verifier.

Compare a candidate answer to the gold answer for the same question. Mark correct=true only when
the candidate is semantically equivalent to the gold answer for the requested target. Accept common
aliases, equivalent names, and harmless formatting differences. Reject wrong entities, missing list
items, extra contradictory answers, wrong relation direction, and answers that only describe a path
without giving the requested value.

Output STRICT JSON:
{"correct": true/false, "reason": "one concise sentence"}
No markdown or prose outside JSON.
"""


def verify_answer(
    answer: str | None,
    expected: str | None,
    question: str | None = None,
    *,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    scores = score_answer(answer, expected)
    local_correct = bool(scores["correct"])
    result: dict[str, Any] = {
        "correct": local_correct,
        "exact": scores["exact"],
        "f1": scores["f1"],
        "semantic_equivalent": local_correct,
        "method": "local_exact_or_f1" if local_correct else "local_mismatch",
    }
    if expected is None:
        result.update({"correct": None, "semantic_equivalent": None, "method": "missing_gold"})
        return result
    if local_correct:
        result["reason"] = "The candidate matches the gold answer by exact/F1 scoring."
        return result
    if use_llm is None:
        use_llm = os.getenv("SII_VERIFY_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
    if not use_llm:
        result["reason"] = "The candidate did not match the gold answer by local scoring."
        return result

    user = (
        f"QUESTION:\n{question or ''}\n\n"
        f"GOLD_ANSWER:\n{expected or ''}\n\n"
        f"CANDIDATE_ANSWER:\n{answer or ''}\n\n"
        f"LOCAL_EXACT: {scores['exact']}\nLOCAL_F1: {scores['f1']}"
    )
    try:
        resp = chat(
            [{"role": "system", "content": SEMANTIC_VERIFY_SYS}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=512,
            timeout=float(os.getenv("SII_VERIFY_LLM_TIMEOUT", "120")),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = _json_obj(raw) or {}
        semantic_correct = parsed.get("correct")
        if isinstance(semantic_correct, bool):
            result["correct"] = semantic_correct
            result["semantic_equivalent"] = semantic_correct
            result["method"] = "llm_semantic"
            result["reason"] = str(parsed.get("reason") or "").strip()
        else:
            result["semantic_judge_error"] = f"missing boolean correct in judge output: {raw[:500]}"
            result["reason"] = "The candidate did not match locally, and semantic judge output was invalid."
    except Exception as exc:  # noqa: BLE001
        result["semantic_judge_error"] = f"{type(exc).__name__}: {exc}"
        result["reason"] = "The candidate did not match locally, and semantic verification failed."
    return result


def verify(answer: str, rationale: str = "") -> str:
    if _runtime_mode() != "train":
        return json.dumps(
            {"error": "verify is only available in train mode.", "runtime_mode": _runtime_mode()},
            ensure_ascii=False,
            indent=2,
        )
    context = _VERIFY_CONTEXT.get() or {}
    expected = context.get("expected")
    if expected is None:
        return json.dumps(
            {
                "error": "No hidden gold answer is configured for this example.",
                "runtime_mode": _runtime_mode(),
            },
            ensure_ascii=False,
            indent=2,
        )
    result = verify_answer(answer, expected, question=context.get("question"))
    result.update(
        {
            "runtime_mode": _runtime_mode(),
            "verified_answer": answer,
            "candidate_rationale": rationale,
            "gold_answer": expected,
            "usage_note": (
                "If correct=false, treat the gold answer as training supervision: reflect on the failure and "
                "submit a corrected concise final answer. Lesson/skill memory is written only after a corrected "
                "answer verifies as correct."
            ),
        }
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
