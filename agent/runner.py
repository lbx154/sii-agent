"""High-level runner that wires ReAct + Reflection + Memory.

Two modes:
  - run_baseline(q):  pure ReAct, no memory, no reflection.
  - run_evolved(q):   inject relevant lessons into prompt; on failure → reflect → store.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import os
import re

from .react import run_react
from .reflection import reflect
from .scoring import judge_answer
from harness.controller import HarnessConfig, HarnessResult
from memory.store import MemoryStore, Episode, Lesson


@dataclass
class RunOutcome:
    result: HarnessResult
    correct: bool | None
    reflection: dict | None = None


def _judge(predicted: str | None, expected: str | None) -> bool | None:
    return judge_answer(predicted, expected)


def _redact_expected(text: str, expected: str | None) -> str:
    if not expected or len(expected.strip()) < 3:
        return text
    return re.sub(re.escape(expected.strip()), "[answer redacted]", text, flags=re.IGNORECASE)


def _redact_expected_obj(value: object, expected: str | None) -> object:
    if isinstance(value, dict):
        return {str(k): _redact_expected_obj(v, expected) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_expected_obj(item, expected) for item in value]
    if value is None or isinstance(value, bool):
        return value
    return _redact_expected(str(value), expected)


def _useful_reflection(reflection: dict | None) -> bool:
    if not reflection:
        return False
    failure_mode = str(reflection.get("failure_mode", "") or "").strip().lower()
    if failure_mode in {
        "",
        "none",
        "n/a",
        "na",
        "null",
        "parse_error",
        "reflection_error",
        "supported_answer",
        "correct",
        "already_correct",
        "no_failure",
    }:
        return False
    lesson = str(reflection.get("reusable_lesson", "")).strip()
    strategy = str(reflection.get("corrective_strategy", "")).strip()
    if strategy.lower() in {"none", "none needed", "n/a", "na", "no strategy needed", "no retry needed"}:
        return False
    if not lesson or not strategy:
        return False
    if len(lesson) < 20 or len(lesson) > 400:
        return False

    text = f"{lesson}\n{strategy}".lower()
    bad_phrases = (
        "withhold",
        "withheld",
        "do not guess",
        "do not answer",
        "do not memorize",
        "privacy",
        "private",
        "sensitive",
        "cannot answer",
        "refuse",
        "refusal",
        "insufficient evidence",
        "state uncertainty",
        "acknowledge uncertainty",
        "search limitations",
        "tool may have limitations",
        "tool limitations",
        "known entity knowledge",
        "not available",
        "doesn't exist",
        "does not exist",
        "answer is correct",
        "already correct",
        "no retry",
        "no corrective action",
    )
    return not any(phrase in text for phrase in bad_phrases)


_LOW_CONFIDENCE_PATTERNS = (
    "cannot determine",
    "could not determine",
    "unable to determine",
    "unable to identify",
    "unable to find",
    "could not identify",
    "cannot find",
    "not enough information",
    "insufficient information",
    "no definitive",
    "not found",
    "i don't know",
    "i do not know",
    "cannot answer",
    "unable to answer",
    "withheld",
)


def _low_confidence_answer(answer: str | None) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    return any(pattern in lower for pattern in _LOW_CONFIDENCE_PATTERNS)


def _verbose_uncertain_answer(answer: str | None) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    uncertainty = (
        "based on the available",
        "best supported",
        "most likely",
        "i will submit",
        "i cannot confidently",
        "not confidently",
    )
    return len(answer) > 450 and any(phrase in lower for phrase in uncertainty)


def _self_contradictory_rationale(rationale: str | None) -> bool:
    if not rationale:
        return False
    lower = rationale.lower()
    return any(
        phrase in lower
        for phrase in (
            "wait,",
            "wait.",
            "i misread",
            "misread the question",
            "need to search",
            "i need to find",
            "maternal grandfather would be the father",
            "paternal grandfather would be the father",
        )
    )


def _postprocess_2wiki_answer(answer: str | None) -> str | None:
    if not answer:
        return answer
    text = answer.strip()
    text = re.sub(r"^(?:the answer is|answer:)\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"\s+", " ", text)

    # Conservative local-scoring cleanup: keep only a substring already present
    # in the model answer; never invent a new entity.
    text = re.sub(r"\s*\((?:now|also known as|formerly|aka)\b[^)]*\)\s*$", "", text, flags=re.IGNORECASE).strip()
    parts = [part.strip() for part in text.split(",")]
    if len(parts) >= 2:
        first = parts[0]
        tail = ", ".join(parts[1:]).lower()
        if tail == first.lower() or re.search(r"\b(prefecture|province|region midtjylland|cal\.|u\.s\.|u\.s\.a\.)\b", tail):
            text = first
    return text.rstrip(" .;:") or answer


def _allow_2wiki_postprocess() -> bool:
    return os.getenv("SII_2WIKI_DISABLE_POSTPROCESS", "").strip().lower() not in {"1", "true", "yes"}


def _needs_self_reflection(result: HarnessResult, cfg: HarnessConfig) -> bool:
    if result.stop_reason != "final":
        return True
    if _low_confidence_answer(result.final_answer):
        return True
    if _verbose_uncertain_answer(result.final_answer):
        return True
    if _self_contradictory_rationale(result.rationale):
        return True
    if result.steps >= max(5, int(cfg.max_steps * 0.75)):
        return True
    return False


def _allow_task_reflection(task: str | None) -> bool:
    if task != "2wiki":
        return True
    value = os.getenv("SII_2WIKI_ENABLE_REFLECTION")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes"}


def _use_skill_updates(task: str | None) -> bool:
    return task == "2wiki" and os.getenv("SII_2WIKI_ENABLE_SKILLS", "").strip().lower() in {"1", "true", "yes"}


def _reflection_requests_retry(reflection: dict | None, result: HarnessResult) -> bool:
    if not reflection:
        return False
    if reflection.get("needs_retry") is True:
        return True
    confidence = str(reflection.get("confidence", "")).strip().lower()
    failure_mode = str(reflection.get("failure_mode", "")).strip().lower()
    if confidence == "low" and failure_mode not in {"", "none", "supported_answer"}:
        return True
    if result.stop_reason != "final" or _low_confidence_answer(result.final_answer):
        return failure_mode not in {"parse_error", "reflection_error"}
    return False


def _retry_config(cfg: HarnessConfig) -> HarnessConfig:
    retry_steps = max(4, min(cfg.max_steps, 8))
    return replace(cfg, max_steps=retry_steps)


def _choose_without_gold(
    first: HarnessResult,
    retry: HarnessResult,
    reflection: dict | None,
) -> HarnessResult:
    if retry.stop_reason != "final" or not retry.final_answer:
        return first
    if first.stop_reason != "final" or not first.final_answer:
        return retry
    if _low_confidence_answer(first.final_answer) and not _low_confidence_answer(retry.final_answer):
        return retry
    if _verbose_uncertain_answer(first.final_answer) and len(retry.final_answer) < len(first.final_answer):
        return retry
    if reflection and reflection.get("needs_retry") is True:
        failure_mode = str(reflection.get("failure_mode", "")).strip().lower()
        if (
            failure_mode == "self_contradictory_final"
            and not _low_confidence_answer(retry.final_answer)
            and retry.final_answer != first.final_answer
        ):
            return retry
        first_len = len(first.final_answer or "")
        retry_len = len(retry.final_answer or "")
        if (
            not _low_confidence_answer(retry.final_answer)
            and retry_len <= max(180, first_len)
            and (first_len > 180 or first.steps >= 20 or _low_confidence_answer(first.final_answer))
        ):
            return retry
    return first


def run_baseline(question: str, expected: str | None = None,
                 cfg: HarnessConfig | None = None) -> RunOutcome:
    res = run_react(question, cfg=cfg)
    return RunOutcome(result=res, correct=_judge(res.final_answer, expected))


def run_evolved(question: str, expected: str | None = None,
                cfg: HarnessConfig | None = None,
                memory: MemoryStore | None = None,
                allow_retry: bool = True,
                allow_reflection: bool = True,
                lesson_context: str | None = None,
                use_gold_for_reflection: bool = False,
                task: str | None = None) -> RunOutcome:
    cfg = cfg or HarnessConfig()
    memory = memory or MemoryStore()
    extra = lesson_context if lesson_context is not None else memory.render_for_prompt(question, task=task)
    res = run_react(question, cfg=cfg, extra_system=extra or None)
    if task == "2wiki" and _allow_2wiki_postprocess():
        res.final_answer = _postprocess_2wiki_answer(res.final_answer)
    reflection = None

    should_reflect = allow_reflection and _allow_task_reflection(task) and (
        _needs_self_reflection(res, cfg) or use_gold_for_reflection
    )
    if should_reflect:
        reflection = reflect(
            question,
            expected,
            res,
            correct=None,
            include_expected=use_gold_for_reflection,
        )
        safe_reflection = _redact_expected_obj(reflection or {}, expected)
        if _useful_reflection(reflection):
            if _use_skill_updates(task) and not use_gold_for_reflection:
                memory.add_reflection_skill(task, question, safe_reflection if isinstance(safe_reflection, dict) else None)
            else:
                memory.add_lesson(Lesson(
                    ts=memory.now(),
                    question=question,
                    failure_mode=str(safe_reflection.get("failure_mode", "")),
                    root_cause=str(safe_reflection.get("root_cause", "")),
                    corrective_strategy=str(safe_reflection.get("corrective_strategy", "")),
                    reusable_lesson=str(safe_reflection.get("reusable_lesson", "")),
                    outcome="failure",
                ))
        if allow_retry and _reflection_requests_retry(reflection, res):
            extra2 = lesson_context if lesson_context is not None else memory.render_for_prompt(question, task=task)
            retry_hint = (
                (extra2 + "\n\n" if extra2 else "") +
                f"[Self-review of previous attempt]\n"
                f"failure_mode: {safe_reflection.get('failure_mode')}\n"
                f"root_cause: {safe_reflection.get('root_cause')}\n"
                f"corrective_strategy: {safe_reflection.get('corrective_strategy')}\n"
                "Retry efficiently: do not repeat previous queries; use at most 3 focused searches "
                "unless a specific missing clue requires one more. Prefer one concise exact answer. "
                "Do not refuse or withhold merely because evidence is incomplete; provide the best-supported answer."
            )
            retry_res = run_react(question, cfg=_retry_config(cfg), extra_system=retry_hint)
            if task == "2wiki" and _allow_2wiki_postprocess():
                retry_res.final_answer = _postprocess_2wiki_answer(retry_res.final_answer)
            res = _choose_without_gold(res, retry_res, reflection)

    correct = _judge(res.final_answer, expected)
    memory.add_episode(Episode(
        ts=memory.now(), question=question, answer=res.final_answer,
        correct=correct, steps=res.steps, tool_calls=res.tool_calls,
        stop_reason=res.stop_reason, tool_call_counts=res.tool_call_counts,
    ))
    return RunOutcome(result=res, correct=correct, reflection=reflection)
