"""Long-term memory: episodic log + distilled lessons, file-backed JSONL.
Retrieval = simple keyword overlap (good enough for SimpleQA/2Wiki, swap for embeddings later).
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Episode:
    ts: float
    question: str
    answer: str | None
    correct: bool | None
    steps: int
    tool_calls: int
    stop_reason: str


@dataclass
class Lesson:
    ts: float
    question: str
    failure_mode: str
    root_cause: str
    corrective_strategy: str
    reusable_lesson: str


_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s or "") if len(t) > 1}


class MemoryStore:
    def __init__(self, root: str | os.PathLike = "logs/memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.root / "episodes.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"

    # ---------------- writers ----------------
    def add_episode(self, ep: Episode) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")

    def add_lesson(self, lesson: Lesson) -> None:
        with self.lessons_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(lesson), ensure_ascii=False) + "\n")

    # ---------------- readers ----------------
    def _read(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def all_lessons(self) -> list[dict]:
        return self._read(self.lessons_path)

    def retrieve_lessons(self, question: str, k: int = 3) -> list[dict]:
        qtok = _tokens(question)
        scored = []
        for l in self.all_lessons():
            ltok = _tokens(l.get("question", "")) | _tokens(l.get("reusable_lesson", ""))
            score = len(qtok & ltok)
            if score:
                scored.append((score, l))
        scored.sort(key=lambda x: -x[0])
        return [l for _, l in scored[:k]]

    # ---------------- helpers ----------------
    @staticmethod
    def now() -> float:
        return time.time()

    def render_for_prompt(self, question: str, k: int = 3) -> str:
        items = self.retrieve_lessons(question, k=k)
        if not items:
            return ""
        bullets = []
        for it in items:
            bullets.append(
                f"- (past failure: {it.get('failure_mode','?')}) "
                f"{it.get('reusable_lesson','').strip()}  "
                f"[fix: {it.get('corrective_strategy','').strip()[:160]}]"
            )
        return "Relevant lessons from past tasks:\n" + "\n".join(bullets)
