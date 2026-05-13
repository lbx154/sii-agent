"""High-level runner that wires ReAct + Reflection + Memory.

Two modes:
  - run_baseline(q):  pure ReAct, no memory, no reflection.
  - run_evolved(q):   inject relevant lessons into prompt; on failure → reflect → store.
"""
from __future__ import annotations
from dataclasses import dataclass

from .react import run_react
from .reflection import reflect
from harness.controller import HarnessConfig, HarnessResult
from memory.store import MemoryStore, Episode, Lesson


@dataclass
class RunOutcome:
    result: HarnessResult
    correct: bool | None
    reflection: dict | None = None


def _judge(predicted: str | None, expected: str | None) -> bool | None:
    if expected is None:
        return None
    if not predicted:
        return False
    p = predicted.strip().lower()
    e = expected.strip().lower()
    return e in p or p in e


def run_baseline(question: str, expected: str | None = None,
                 cfg: HarnessConfig | None = None) -> RunOutcome:
    res = run_react(question, cfg=cfg)
    return RunOutcome(result=res, correct=_judge(res.final_answer, expected))


def run_evolved(question: str, expected: str | None = None,
                cfg: HarnessConfig | None = None,
                memory: MemoryStore | None = None,
                allow_retry: bool = True) -> RunOutcome:
    memory = memory or MemoryStore()
    extra = memory.render_for_prompt(question)
    res = run_react(question, cfg=cfg, extra_system=extra or None)
    correct = _judge(res.final_answer, expected)
    reflection = None

    failed = (res.stop_reason != "final") or (correct is False)
    if failed:
        reflection = reflect(question, expected, res)
        memory.add_lesson(Lesson(
            ts=memory.now(),
            question=question,
            failure_mode=reflection.get("failure_mode", ""),
            root_cause=reflection.get("root_cause", ""),
            corrective_strategy=reflection.get("corrective_strategy", ""),
            reusable_lesson=reflection.get("reusable_lesson", ""),
        ))
        if allow_retry:
            extra2 = memory.render_for_prompt(question)
            retry_hint = (
                (extra2 + "\n\n" if extra2 else "") +
                f"[Self-reflection on previous attempt]\n"
                f"failure_mode: {reflection.get('failure_mode')}\n"
                f"root_cause: {reflection.get('root_cause')}\n"
                f"corrective_strategy: {reflection.get('corrective_strategy')}\n"
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
