"""High-level runner that wires ReAct + Reflection + Memory.

Two modes:
  - run_baseline(q):  pure ReAct, no memory, no reflection.
  - run_evolved(q):   inject relevant lessons into prompt; on failure → reflect → store.
"""
from __future__ import annotations
from dataclasses import dataclass
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
    if str(reflection.get("failure_mode", "")).strip() == "parse_error":
        return False
    lesson = str(reflection.get("reusable_lesson", "")).strip()
    strategy = str(reflection.get("corrective_strategy", "")).strip()
    if not lesson or not strategy:
        return False
    if len(lesson) < 20 or len(lesson) > 400:
        return False

    text = f"{lesson}\n{strategy}".lower()
    bad_phrases = (
        "withhold",
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
    )
    return not any(phrase in text for phrase in bad_phrases)


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
    memory = memory or MemoryStore()
    extra = lesson_context if lesson_context is not None else memory.render_for_prompt(question)
    res = run_react(question, cfg=cfg, extra_system=extra or None)
    correct = _judge(res.final_answer, expected)
    reflection = None

    failed = (res.stop_reason != "final") or (correct is False)
    if failed and allow_reflection:
        reflection = reflect(
            question,
            expected,
            res,
            correct=correct,
            include_expected=use_gold_for_reflection,
        )
        safe_reflection = {
            key: _redact_expected(str(value), expected)
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
            ))
        if allow_retry:
            extra2 = lesson_context if lesson_context is not None else memory.render_for_prompt(question)
            retry_hint = (
                (extra2 + "\n\n" if extra2 else "") +
                f"[Self-reflection on previous attempt]\n"
                f"failure_mode: {safe_reflection.get('failure_mode')}\n"
                f"root_cause: {safe_reflection.get('root_cause')}\n"
                f"corrective_strategy: {safe_reflection.get('corrective_strategy')}\n"
                f"DO NOT repeat the previous mistake."
            )
            res = run_react(question, cfg=cfg, extra_system=retry_hint)
            correct = _judge(res.final_answer, expected)

    memory.add_episode(Episode(
        ts=memory.now(), question=question, answer=res.final_answer,
        correct=correct, steps=res.steps, tool_calls=res.tool_calls,
        stop_reason=res.stop_reason,
    ))
    return RunOutcome(result=res, correct=correct, reflection=reflection)
