"""Harness — bounds the agent: max steps, wall time, repeated-call detection."""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HarnessConfig:
    max_steps: int = 8
    max_wall_seconds: float = 120.0
    max_repeats: int = 2          # same tool+args within window → flagged
    max_consecutive_no_tool: int = 2  # LLM keeps "thinking" without acting


@dataclass
class HarnessResult:
    final_answer: str | None = None
    rationale: str = ""
    steps: int = 0
    elapsed: float = 0.0
    stop_reason: str = ""         # "final" | "max_steps" | "timeout" | "loop" | "error"
    trajectory: list[dict] = field(default_factory=list)
    tool_calls: int = 0


class StepGuard:
    """Detects pathological loops in tool calls."""

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.t0 = time.time()
        self._sigs: dict[str, int] = {}
        self._consecutive_no_tool = 0

    def time_left(self) -> float:
        return self.cfg.max_wall_seconds - (time.time() - self.t0)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def note_no_tool(self) -> bool:
        self._consecutive_no_tool += 1
        return self._consecutive_no_tool >= self.cfg.max_consecutive_no_tool

    def reset_no_tool(self) -> None:
        self._consecutive_no_tool = 0

    def note_tool(self, name: str, arguments: Any) -> bool:
        sig = hashlib.sha1(f"{name}|{arguments}".encode()).hexdigest()
        self._sigs[sig] = self._sigs.get(sig, 0) + 1
        return self._sigs[sig] > self.cfg.max_repeats
