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
from threading import RLock


@dataclass
class Episode:
    ts: float
    question: str
    answer: str | None
    correct: bool | None
    steps: int
    tool_calls: int
    stop_reason: str
    tool_call_counts: dict[str, int] | None = None


@dataclass
class Lesson:
    ts: float
    question: str
    failure_mode: str
    root_cause: str
    corrective_strategy: str
    reusable_lesson: str
    outcome: str = "failure"
    score: float | None = None


_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n\n|$)", re.S)
_NOOP_VALUES = {"", "none", "n/a", "na", "null", "no strategy needed", "none needed"}
_YESNO_START = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could"}
_2WIKI_SEED_LESSONS: tuple[dict, ...] = (
    {
        "ts": 0.0,
        "question": "2WikiMultihopQA",
        "failure_mode": "two_hop_context_policy",
        "root_cause": "2Wiki context usually contains both hops, while external search often retrieves aliases or wrong entities.",
        "corrective_strategy": "First solve from the provided context: identify the question entity, find the intermediate entity, then answer the requested attribute. Search only when the relevant context blocks do not contain the needed fact.",
        "reusable_lesson": "For 2Wiki, prefer the provided context over search; do the two-hop chain inside context before calling any external tool.",
        "outcome": "seeded",
        "score": 1.0,
        "tags": ["2wiki", "compositional", "bridge", "context"],
    },
    {
        "ts": 0.0,
        "question": "2Wiki comparison questions",
        "failure_mode": "comparison_error",
        "root_cause": "Comparison questions require extracting one comparable value for each candidate before choosing the requested candidate.",
        "corrective_strategy": "For older/younger/first/earlier/same/different questions, write down both candidate facts from context, compare them, and return the candidate name or yes/no requested by the question.",
        "reusable_lesson": "For comparison questions, compare both candidate facts explicitly and answer the requested candidate, not the date or attribute.",
        "outcome": "seeded",
        "score": 1.0,
        "tags": ["2wiki", "comparison", "bridge_comparison", "yesno"],
    },
    {
        "ts": 0.0,
        "question": "2Wiki family relation questions",
        "failure_mode": "wrong_hop",
        "root_cause": "Family-chain questions are easy to answer with the wrong generation or spouse/parent relation.",
        "corrective_strategy": "For father/mother/child/spouse/grandparent/in-law questions, verify each relation in order and keep the intermediate entity separate from the final answer.",
        "reusable_lesson": "For family chains, verify hop 1 and hop 2 separately; do not swap parent, spouse, child, or grandparent roles.",
        "outcome": "seeded",
        "score": 1.0,
        "tags": ["2wiki", "family", "father", "mother", "wife", "husband", "child", "grandfather", "grandmother"],
    },
    {
        "ts": 0.0,
        "question": "2Wiki answer span formatting",
        "failure_mode": "format_violation",
        "root_cause": "Verbose answers and nationality/place aliases can be judged differently from the concise context span.",
        "corrective_strategy": "Return exactly the requested span: yes/no in lowercase, copy country/demonym wording from context, preserve date/place granularity, and no explanatory sentence.",
        "reusable_lesson": "Return only the concise answer span; for country/nationality questions copy the context wording instead of converting demonyms to countries.",
        "outcome": "seeded",
        "score": 1.0,
        "tags": ["2wiki", "format", "country", "nationality", "yesno", "date", "place"],
    },
)


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s or "") if len(t) > 1}


def _clean_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _NOOP_VALUES else text


def _original_question(question: str) -> str:
    match = _QUESTION_RE.search(question or "")
    return (match.group(1) if match else question or "").strip()


def _question_features(question: str) -> set[str]:
    q = _original_question(question).lower()
    tokens = _tokens(q)
    features = {"2wiki"} if "provided context" in (question or "").lower() or "2wiki" in (question or "").lower() else set()
    words = q.split()
    if words and words[0] in _YESNO_START:
        features.add("yesno")
    if "which" in tokens or any(term in q for term in ("came out first", "died first", "older", "younger", "established first")):
        features.add("comparison")
    if "country" in tokens or "nationality" in tokens:
        features.update({"country", "nationality"})
    if "where" in tokens or "place of birth" in q or "place of death" in q or "born" in tokens or "died" in tokens:
        features.add("place")
    if "when" in tokens or "date" in tokens:
        features.add("date")
    family_terms = {
        "father", "mother", "wife", "husband", "child", "grandfather", "grandmother",
        "paternal", "maternal", "in-law",
    }
    if tokens & family_terms:
        features.add("family")
        features.update(tokens & family_terms)
    domain_terms = {
        "director", "performer", "film", "song", "award", "school", "work", "birth",
        "death", "cause", "composer", "presenter",
    }
    features.update(tokens & domain_terms)
    return features


def _recency(ts: object) -> float:
    try:
        age_days = max(0.0, (time.time() - float(ts)) / 86400)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + age_days)


def _tool_summary(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "the minimum useful tool path"
    def count_value(item: tuple[object, object]) -> int:
        try:
            return int(item[1])
        except (TypeError, ValueError):
            return 0

    ordered = sorted(counts.items(), key=lambda item: (-count_value(item), str(item[0])))
    return ", ".join(f"{name}×{count}" for name, count in ordered[:4])


class MemoryStore:
    def __init__(self, root: str | os.PathLike = "logs/memory", read_only: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.root / "episodes.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"
        self.read_only = read_only
        self._lock = RLock()

    # ---------------- writers ----------------
    def add_episode(self, ep: Episode) -> None:
        if self.read_only:
            return
        with self._lock:
            with self.episodes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")

    def add_lesson(self, lesson: Lesson) -> None:
        if self.read_only:
            return
        with self._lock:
            with self.lessons_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(lesson), ensure_ascii=False) + "\n")

    # ---------------- readers ----------------
    def _read(self, path: Path) -> list[dict]:
        with self._lock:
            if not path.exists():
                return []
            return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def all_lessons(self) -> list[dict]:
        return self._read(self.lessons_path)

    def retrieve_lessons(self, question: str, k: int = 3, task: str | None = None) -> list[dict]:
        qtok = _tokens(question)
        qfeatures = _question_features(question)
        scored = []
        if task == "2wiki":
            for l in _2WIKI_SEED_LESSONS:
                tags = set(l.get("tags") or [])
                specific_overlap = len((qfeatures - {"2wiki"}) & (tags - {"2wiki"}))
                failure_mode = str(l.get("failure_mode", ""))
                if failure_mode == "two_hop_context_policy":
                    scored.append((30 + specific_overlap, 1.0, l))
                elif failure_mode == "format_violation":
                    scored.append((18 + specific_overlap, 1.0, l))
                elif specific_overlap:
                    scored.append((24 + 5 * specific_overlap, 1.0, l))
        for l in self.all_lessons():
            failure_mode = _clean_value(l.get("failure_mode")).lower()
            if task == "2wiki" and failure_mode in {"", "none", "n/a", "na", "supported_answer", "correct", "already_correct"}:
                continue
            ltok = (
                _tokens(l.get("question", ""))
                | _tokens(l.get("failure_mode", ""))
                | _tokens(l.get("root_cause", ""))
                | _tokens(l.get("corrective_strategy", ""))
                | _tokens(l.get("reusable_lesson", ""))
            )
            overlap = len(qtok & ltok)
            feature_overlap = len(qfeatures & _question_features(str(l.get("question", "")) + " " + str(l.get("reusable_lesson", ""))))
            score = overlap + 4 * feature_overlap
            if score >= 2:
                scored.append((score, _recency(l.get("ts")), l))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [l for _, _, l in scored[:k]]

    def all_episodes(self) -> list[dict]:
        return self._read(self.episodes_path)

    def retrieve_successes(self, question: str, k: int = 2) -> list[dict]:
        qtok = _tokens(question)
        scored = []
        for ep in self.all_episodes():
            if ep.get("correct") is not True:
                continue
            overlap = len(qtok & _tokens(ep.get("question", "")))
            if overlap >= 2:
                scored.append((overlap, _recency(ep.get("ts")), ep))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [ep for _, _, ep in scored[:k]]

    # ---------------- helpers ----------------
    @staticmethod
    def now() -> float:
        return time.time()

    def render_for_prompt(
        self,
        question: str,
        k: int = 3,
        include_successes: bool = True,
        task: str | None = None,
    ) -> str:
        lessons = self.retrieve_lessons(question, k=k, task=task)
        successes = self.retrieve_successes(question, k=max(1, min(2, k))) if include_successes else []
        if not lessons and not successes:
            return ""
        sections: list[str] = []
        if lessons:
            bullets = []
            for it in lessons:
                lesson = _clean_value(it.get("reusable_lesson"))
                strategy = _clean_value(it.get("corrective_strategy"))
                if not lesson:
                    continue
                bullets.append(
                    f"- (past {it.get('outcome','failure')}: {it.get('failure_mode','?')}) "
                    f"{lesson}  "
                    f"[fix: {strategy[:160]}]"
                )
            if bullets:
                heading = "2Wiki evolved memory (use only when relevant; user answer-format rules still win)" if task == "2wiki" else "Relevant lessons from past tasks"
                sections.append(heading + ":\n" + "\n".join(bullets))
        if successes:
            bullets = []
            for ep in successes:
                bullets.append(
                    "- Similar prior task succeeded; reuse its efficient pattern "
                    f"({ep.get('steps', '?')} steps, tools: {_tool_summary(ep.get('tool_call_counts'))}) "
                    "but derive the answer from current evidence."
                )
            sections.append("Successful strategy memories:\n" + "\n".join(bullets))
        return "\n\n".join(sections)
