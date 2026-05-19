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
    max_llm_tokens: int = 1536
    max_llm_call_seconds: float = 60.0
    min_llm_call_seconds: float = 20.0
    allowed_tools: tuple[str, ...] | None = None
    finalize_on_stop: bool = True
    max_repeats: int = 2          # same tool+args within window → flagged
    max_consecutive_no_tool: int = 2  # LLM keeps "thinking" without acting
    use_short_memory: bool = False
    short_memory_max_chars: int = 2500
    max_parallel_tool_calls: int = 0  # 0 disables; otherwise execute at most this many per turn
    max_web_search_calls: int = 0  # 0 disables; otherwise remove web_search after this many calls
    max_research_tool_calls: int = 0  # 0 disables; otherwise force final after this many evidence calls
    synthesize_after_tool_calls: int = 0  # 0 disables; otherwise send a one-time convergence reminder


@dataclass
class HarnessResult:
    final_answer: str | None = None
    rationale: str = ""
    steps: int = 0
    elapsed: float = 0.0
    stop_reason: str = ""         # "final" | "max_steps" | "timeout" | "loop" | "error"
    trajectory: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    finish_reasons: dict[str, int] = field(default_factory=dict)
    short_memory_stats: dict[str, int] = field(default_factory=dict)
    internal_verify_results: list[dict[str, Any]] = field(default_factory=list)


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
