"""High-level runner that wires ReAct + Reflection + Memory.

Two modes:
  - run_baseline(q):  pure ReAct, no memory, no reflection.
  - run_evolved(q):   inject relevant lessons into prompt; on failure → reflect → store.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
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


def _needs_self_reflection(result: HarnessResult, cfg: HarnessConfig) -> bool:
    if result.stop_reason != "final":
        return True
    if _low_confidence_answer(result.final_answer):
        return True
    if _verbose_uncertain_answer(result.final_answer):
        return True
    if result.steps >= max(5, int(cfg.max_steps * 0.75)):
        return True
    return False


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
                use_gold_for_reflection: bool = False) -> RunOutcome:
    cfg = cfg or HarnessConfig()
    memory = memory or MemoryStore()
    extra = lesson_context if lesson_context is not None else memory.render_for_prompt(question)
    res = run_react(question, cfg=cfg, extra_system=extra or None)
    reflection = None

    should_reflect = allow_reflection and (
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
        safe_reflection = {
            key: _redact_expected("" if value is None else str(value), expected)
            for key, value in (reflection or {}).items()
        }
        if _useful_reflection(reflection):
            memory.add_lesson(Lesson(
                ts=memory.now(),
                question=question,
                failure_mode=safe_reflection.get("failure_mode", ""),
                root_cause=safe_reflection.get("root_cause", ""),
                corrective_strategy=safe_reflection.get("corrective_strategy", ""),
                reusable_lesson=safe_reflection.get("reusable_lesson", ""),
                outcome="failure",
            ))
        if allow_retry and _reflection_requests_retry(reflection, res):
            extra2 = lesson_context if lesson_context is not None else memory.render_for_prompt(question)
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
            res = _choose_without_gold(res, retry_res, reflection)

    correct = _judge(res.final_answer, expected)
    memory.add_episode(Episode(
        ts=memory.now(), question=question, answer=res.final_answer,
        correct=correct, steps=res.steps, tool_calls=res.tool_calls,
        stop_reason=res.stop_reason, tool_call_counts=res.tool_call_counts,
    ))
    return RunOutcome(result=res, correct=correct, reflection=reflection)
