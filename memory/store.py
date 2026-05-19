"""Long-term memory: episodic log + distilled lessons, file-backed JSONL.

Lesson retrieval defaults to a local dense LSA embedding index plus small
task-feature reranking, with BM25 retained as a deterministic fallback.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict, field
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


@dataclass
class Skill:
    ts: float
    id: str
    task: str
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    verifier: list[str] = field(default_factory=list)
    bad_patterns: list[str] = field(default_factory=list)
    source: str = "reflection"
    score: float | None = None


_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\n\n|$)", re.S)
_NOOP_VALUES = {"", "none", "n/a", "na", "null", "no strategy needed", "none needed"}
_YESNO_START = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could"}
_BAD_SKILL_PHRASES = (
    "withhold",
    "withheld",
    "do not answer",
    "cannot answer",
    "refuse",
    "privacy",
    "private",
    "sensitive",
    "state uncertainty",
    "acknowledge uncertainty",
    "answer is correct",
    "already correct",
    "no retry",
    "no corrective action",
)
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
_2WIKI_SEED_SKILLS: tuple[dict, ...] = (
    {
        "ts": 0.0,
        "id": "2wiki_context_first",
        "task": "2wiki",
        "title": "Context-first two-hop solving",
        "description": "Use the provided context as primary evidence and gate search.",
        "tags": ["2wiki", "context", "bridge", "search_gate"],
        "triggers": ["provided context", "context", "two-hop", "bridge"],
        "steps": [
            "Resolve the two-hop chain from the Provided context before searching.",
            "Search only when the needed hop is absent or contradicted in context.",
            "If searching, query the intermediate entity plus the requested attribute.",
        ],
        "verifier": [
            "Each hop is supported by context or a focused retrieved page.",
            "Search evidence does not override a clear context fact.",
        ],
        "bad_patterns": ["Searching broadly just to cross-check facts already present in context."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_answer_span",
        "task": "2wiki",
        "title": "Concise final answer span",
        "description": "Return the shortest answer span that satisfies the question.",
        "tags": ["2wiki", "format", "answer_span", "yesno", "date", "place"],
        "triggers": ["who", "what", "where", "when", "which", "is", "are", "was", "were"],
        "steps": [
            "Submit only the final answer span, with no rationale.",
            "For yes/no questions, answer lowercase yes or no.",
            "For place/date/entity answers, preserve the granularity requested by the question.",
        ],
        "verifier": ["The final answer is not a full sentence and does not include extra location/detail unless requested."],
        "bad_patterns": ["Returning a full address when the question expects a city-level place."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_family_chain",
        "task": "2wiki",
        "title": "Family-chain relation",
        "description": "Keep each family relation hop separate.",
        "tags": ["2wiki", "family", "father", "mother", "wife", "husband", "spouse", "grandfather", "grandmother", "in-law"],
        "triggers": [
            "father",
            "mother",
            "grandfather",
            "grandmother",
            "husband",
            "wife",
            "spouse",
            "child",
            "son",
            "daughter",
            "maternal",
            "paternal",
            "in-law",
            "father-in-law",
            "mother-in-law",
        ],
        "steps": [
            "Name the intermediate person first, then resolve the requested relation on that person.",
            "Do not replace the requested relation with spouse, parent, child, or sibling.",
        ],
        "verifier": ["Hop 1 and hop 2 are both explicitly supported before final answer."],
        "bad_patterns": ["Answering the mother/spouse when the question asks for a grandparent or in-law."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_birthplace_granularity",
        "task": "2wiki",
        "title": "Birth/death place granularity",
        "description": "Answer where-born/where-died questions at the requested granularity.",
        "tags": ["2wiki", "place", "birth", "death", "born", "died"],
        "triggers": ["born", "birthplace", "place of birth", "where was", "where were", "died", "place of death"],
        "steps": [
            "Find the target person's birth/death place, not their title, family seat, or residence.",
            "Prefer the city/town/country span asked for by the question over a longer address.",
        ],
        "verifier": ["The answer names the actual birthplace/deathplace, not an associated title or institution."],
        "bad_patterns": ["Guessing a duke's birthplace from the duchy title."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_worked_at_publication",
        "task": "2wiki",
        "title": "Worked-at/publication bridge",
        "description": "Resolve work-at and wrote-for questions to the publication or organization.",
        "tags": ["2wiki", "work", "publication", "magazine", "newspaper", "organization"],
        "triggers": ["worked at", "work at", "wrote for", "writer for", "editor of", "journalist", "publication", "magazine", "newspaper"],
        "steps": [
            "When asked where someone worked/wrote/edited, look for a publication or organization span.",
            "Do not answer that context is unspecified until checking nearby career sentences.",
        ],
        "verifier": ["The final answer is the workplace/publication/organization requested, not a role or location."],
        "bad_patterns": ["Answering 'not specified' when a career sentence names a magazine/newspaper."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_comparison",
        "task": "2wiki",
        "title": "Comparison decomposition",
        "description": "Extract comparable values before choosing.",
        "tags": ["2wiki", "comparison", "yesno", "date"],
        "triggers": ["which", "older", "younger", "earlier", "first", "same", "different", "larger", "higher", "came out first", "died first"],
        "steps": [
            "Extract one comparable value for each candidate before comparing.",
            "Return the requested candidate or yes/no, not the compared date/value unless asked.",
        ],
        "verifier": ["Both candidates have extracted values before the comparison answer."],
        "bad_patterns": ["Returning a date when the question asks which entity."],
        "source": "seeded",
        "score": 1.0,
    },
    {
        "ts": 0.0,
        "id": "2wiki_demonym_country",
        "task": "2wiki",
        "title": "Country vs nationality wording",
        "description": "Follow the question wording for country/nationality answers.",
        "tags": ["2wiki", "country", "nationality", "demonym"],
        "triggers": ["country", "nationality", "national", "citizen", "demonym"],
        "steps": [
            "Copy the country/nationality wording from the supporting context instead of converting between demonym and country name.",
            "If context says a person is German/Dutch/French, do not rewrite it to Germany/Netherlands/France unless that exact country span is the answer evidence.",
        ],
        "verifier": ["The final answer wording is directly supported by the relevant context span."],
        "bad_patterns": ["Changing German to Germany or Dutch to Netherlands when the context answer uses the demonym."],
        "source": "seeded",
        "score": 1.0,
    },
)
_2WIKI_DEFAULT_SKILL_IDS: tuple[str, ...] = ()
_2WIKI_SEED_SKILL_IDS = {str(skill["id"]) for skill in _2WIKI_SEED_SKILLS}
_2WIKI_GRANULARITY_SKILL_IDS = {"2wiki_birthplace_granularity", "2wiki_demonym_country"}
_2WIKI_POLICY_CARDS: tuple[dict, ...] = (
    {
        "id": "2wiki_policy_context_first",
        "title": "Context-first evidence",
        "tags": ["2wiki", "context", "bridge", "search"],
        "triggers": ["context", "provided context", "two-hop"],
        "rules": [
            "Resolve the two-hop chain from Provided context first.",
            "Use search only to fill a missing fact; never let current/external snippets replace a clear context entity or relation.",
            "Never submit unknown/not-specified; if the requested attribute is absent, do one focused lookup for the fixed entity and attribute.",
        ],
    },
    {
        "id": "2wiki_policy_keep_hop_anchor",
        "title": "Keep hop entity fixed",
        "tags": ["2wiki", "bridge", "director", "performer", "film", "song"],
        "triggers": ["director", "performer", "film", "song", "author", "composer", "presenter"],
        "rules": [
            "After identifying the intermediate entity, keep that exact entity fixed.",
            "Any search query must include the intermediate entity plus the requested attribute.",
        ],
    },
    {
        "id": "2wiki_policy_work_at_context_role",
        "title": "Work-at means context-supported role",
        "tags": ["2wiki", "work", "publication", "organization", "magazine", "newspaper"],
        "triggers": ["work at", "worked at", "works at", "wrote for", "editor of", "publication", "magazine", "newspaper"],
        "rules": [
            "For works/worked/wrote/edited questions, extract the employer, publication, or organization from context career sentences.",
            "Do not answer with a current web job unless context lacks the role and the question explicitly asks current work.",
        ],
    },
    {
        "id": "2wiki_policy_historical_granularity",
        "title": "Historical country/nationality wording",
        "tags": ["2wiki", "country", "nationality", "historical", "kingdom", "empire"],
        "triggers": ["country", "nationality", "kingdom", "empire", "from"],
        "rules": [
            "For historical figures, preserve context wording such as empire, kingdom, polity, or demonym.",
            "Do not modernize Byzantine/Ostrogothic Kingdom/German-style spans unless the evidence uses the modern country name.",
        ],
    },
    {
        "id": "2wiki_policy_family_chain",
        "title": "Family-chain verification",
        "tags": ["2wiki", "family", "father", "mother", "husband", "wife", "grandfather", "grandmother"],
        "triggers": ["father", "mother", "husband", "wife", "grandfather", "grandmother", "paternal", "maternal", "in-law"],
        "rules": [
            "Name hop 1 before answering hop 2.",
            "Do not swap spouse, parent, child, or grandparent roles.",
        ],
    },
    {
        "id": "2wiki_policy_comparison_yesno",
        "title": "Comparison and yes/no",
        "tags": ["2wiki", "comparison", "yesno"],
        "triggers": ["same", "different", "older", "younger", "earlier", "first", "both", "are both", "were both"],
        "rules": [
            "Extract one comparable value for each candidate before comparing.",
            "For same-country/same-nationality questions, answer only yes or no.",
        ],
    },
)


_LESSON_BM25_K1 = 1.5
_LESSON_BM25_B = 0.75
_LESSON_FEATURE_BOOST = 3.0
_LESSON_RECENCY_BOOST = 0.05
_LESSON_EMBEDDING_FEATURE_BOOST = 0.08
_LESSON_EMBEDDING_RECENCY_BOOST = 0.005
_LESSON_EMBEDDING_DIMS = 128
_LESSON_EMBEDDING_MAX_FEATURES = 8192


def _token_list(s: object) -> list[str]:
    return [t.lower() for t in _TOK.findall(str(s or "")) if len(t) > 1]


def _tokens(s: str) -> set[str]:
    return set(_token_list(s))


def _clean_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _NOOP_VALUES else text


def _clean_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif value:
        items = [value]
    else:
        items = []
    cleaned = []
    for item in items:
        text = _clean_value(item)
        if text:
            cleaned.append(text)
    return cleaned


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
        "paternal", "maternal", "in-law", "spouse", "son", "daughter",
    }
    if tokens & family_terms:
        features.add("family")
        features.update(tokens & family_terms)
    domain_terms = {
        "director", "performer", "film", "song", "award", "school", "work", "birth",
        "death", "cause", "composer", "presenter", "publication", "magazine",
        "newspaper", "organization",
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


def _skill_text(skill: dict) -> str:
    return " ".join(
        str(part)
        for part in (
            skill.get("id"),
            skill.get("title"),
            skill.get("description"),
            " ".join(_clean_list(skill.get("tags"))),
            " ".join(_clean_list(skill.get("triggers"))),
            " ".join(_clean_list(skill.get("steps"))),
            " ".join(_clean_list(skill.get("verifier"))),
            " ".join(_clean_list(skill.get("bad_patterns"))),
        )
        if part
    )


def _lesson_text(lesson: dict) -> str:
    question = _original_question(_clean_value(lesson.get("question")))
    failure_mode = _clean_value(lesson.get("failure_mode"))
    root_cause = _clean_value(lesson.get("root_cause"))
    corrective_strategy = _clean_value(lesson.get("corrective_strategy"))
    reusable_lesson = _clean_value(lesson.get("reusable_lesson"))
    tags = " ".join(_clean_list(lesson.get("tags")))
    weighted_parts = (
        [question] * 4
        + [reusable_lesson] * 4
        + [corrective_strategy] * 2
        + [failure_mode] * 2
        + [root_cause, tags]
    )
    return " ".join(part for part in weighted_parts if part)


def _phrase_hits(question: str, triggers: object) -> int:
    q = f" {question.lower()} "
    hits = 0
    for trigger in _clean_list(triggers):
        trig = trigger.lower().strip()
        if not trig:
            continue
        if " " in trig or "-" in trig:
            hits += int(trig in q)
        else:
            hits += int(re.search(rf"\b{re.escape(trig)}\b", q) is not None)
    return hits


def _render_2wiki_policy_cards(question: str, max_cards: int = 5) -> str:
    q = _original_question(question)
    qfeatures = _question_features(question)
    always_ids = {"2wiki_policy_context_first", "2wiki_policy_keep_hop_anchor"}
    selected = [card for card in _2WIKI_POLICY_CARDS if card["id"] in always_ids]
    selected_ids = {card["id"] for card in selected}
    scored: list[tuple[int, dict]] = []
    for card in _2WIKI_POLICY_CARDS:
        if card["id"] in selected_ids:
            continue
        tags = set(card.get("tags") or [])
        score = 8 * _phrase_hits(q, card.get("triggers")) + 5 * len((qfeatures - {"2wiki"}) & (tags - {"2wiki"}))
        if score > 0:
            scored.append((score, card))
    scored.sort(key=lambda item: (-item[0], str(item[1]["id"])))
    for _, card in scored:
        selected.append(card)
        selected_ids.add(card["id"])
        if len(selected) >= max_cards:
            break
    lines = []
    for card in selected:
        rules = " ".join(str(rule).strip() for rule in card.get("rules", []) if str(rule).strip())
        if rules:
            lines.append(f"- [{card['id']}] {card['title']}: {rules}")
    if not lines:
        return ""
    return "2Wiki typed policies (highest priority; apply when relevant):\n" + "\n".join(lines)


def _has_bad_skill_phrase(*values: object) -> bool:
    text = "\n".join(str(value or "") for value in values).lower()
    return any(phrase in text for phrase in _BAD_SKILL_PHRASES)


def _stable_skill_id(task: str, target: str, trigger: str, step: str) -> str:
    digest = hashlib.sha1(f"{task}\n{target}\n{trigger}\n{step}".encode("utf-8")).hexdigest()[:12]
    return f"{task}_reflection_{digest}"


def _enabled_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


class MemoryStore:
    def __init__(self, root: str | os.PathLike = "logs/memory", read_only: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.root / "episodes.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"
        self.skills_path = self.root / "skills.jsonl"
        self.read_only = read_only
        self._lock = RLock()
        self._lesson_bm25_key: tuple[object, ...] | None = None
        self._lesson_bm25_index: dict[str, object] | None = None
        self._lesson_embedding_key: tuple[object, ...] | None = None
        self._lesson_embedding_index: dict[str, object] | None = None

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
            self._lesson_bm25_key = None
            self._lesson_bm25_index = None
            self._lesson_embedding_key = None
            self._lesson_embedding_index = None

    def add_skill(self, skill: Skill) -> bool:
        if self.read_only:
            return False
        with self._lock:
            existing_ids = {str(item.get("id")) for item in self.all_skills()}
            if skill.id in existing_ids or skill.id in _2WIKI_SEED_SKILL_IDS:
                return False
            with self.skills_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(skill), ensure_ascii=False) + "\n")
            return True

    def add_reflection_skill(self, task: str | None, question: str, reflection: dict | None) -> Skill | None:
        if task != "2wiki" or not isinstance(reflection, dict):
            return None
        update = reflection.get("skill_update")
        if not isinstance(update, dict):
            return None
        target = _clean_value(update.get("target_skill_id"))
        if target not in _2WIKI_SEED_SKILL_IDS:
            return None
        trigger = _clean_value(update.get("trigger"))
        step = _clean_value(update.get("step"))
        verifier = _clean_value(update.get("verifier"))
        bad_pattern = _clean_value(update.get("bad_pattern"))
        if not step or len(step) < 20:
            return None
        verbs = ("identify", "verify", "extract", "search", "answer", "compare", "use", "return", "check", "resolve", "keep", "query")
        if not any(verb in step.lower() for verb in verbs):
            return None
        if _has_bad_skill_phrase(trigger, step, verifier, bad_pattern):
            return None
        failure_mode = _clean_value(reflection.get("failure_mode")) or "reflection_update"
        root_cause = _clean_value(reflection.get("root_cause"))
        features = sorted(_question_features(question) - {"2wiki"})
        skill = Skill(
            ts=self.now(),
            id=_stable_skill_id("2wiki", target, trigger, step),
            task="2wiki",
            title=f"Reflection update for {target}",
            description=(root_cause[:240] if root_cause else f"Update derived from {failure_mode}."),
            tags=["2wiki", target, failure_mode, *features[:4]],
            triggers=[trigger] if trigger else [],
            steps=[step],
            verifier=[verifier] if verifier else [],
            bad_patterns=[bad_pattern] if bad_pattern else [],
            source="reflection",
            score=0.5,
        )
        return skill if self.add_skill(skill) else None

    # ---------------- readers ----------------
    def _read(self, path: Path) -> list[dict]:
        with self._lock:
            if not path.exists():
                return []
            return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def all_lessons(self) -> list[dict]:
        return self._read(self.lessons_path)

    def all_skills(self) -> list[dict]:
        return self._read(self.skills_path)

    def retrieve_skills(self, question: str, k: int = 2, task: str | None = None) -> list[dict]:
        if task != "2wiki":
            return []
        if not _enabled_env("SII_2WIKI_ENABLE_SKILLS"):
            return []
        enable_granularity_skills = _enabled_env("SII_2WIKI_ENABLE_GRANULARITY_SKILLS")
        q = _original_question(question)
        qtok = _tokens(q)
        qfeatures = _question_features(question)
        seed_by_id = {str(skill["id"]): skill for skill in _2WIKI_SEED_SKILLS}
        selected: list[dict] = [seed_by_id[skill_id] for skill_id in _2WIKI_DEFAULT_SKILL_IDS if skill_id in seed_by_id]
        selected_ids = {str(skill.get("id")) for skill in selected}

        scored: list[tuple[float, float, dict]] = []
        candidates = [skill for skill in _2WIKI_SEED_SKILLS if str(skill.get("id")) not in selected_ids]
        candidates.extend(skill for skill in self.all_skills() if skill.get("task") == task)
        for skill in candidates:
            sid = str(skill.get("id"))
            if sid in _2WIKI_GRANULARITY_SKILL_IDS and not enable_granularity_skills:
                continue
            tags = set(_clean_list(skill.get("tags")))
            tag_overlap = len((qfeatures - {"2wiki"}) & (tags - {"2wiki"}))
            phrase_overlap = _phrase_hits(q, skill.get("triggers"))
            token_overlap = len(qtok & _tokens(_skill_text(skill)))
            is_seed = str(skill.get("source")) == "seeded"
            if sid == "2wiki_birthplace_granularity" and phrase_overlap == 0:
                continue
            if is_seed and sid not in _2WIKI_DEFAULT_SKILL_IDS and phrase_overlap == 0 and tag_overlap == 0:
                continue
            if is_seed and phrase_overlap == 0 and tag_overlap == 0 and token_overlap == 0:
                continue
            if not is_seed and phrase_overlap == 0 and tag_overlap < 2:
                continue
            score = 10 * phrase_overlap + 6 * tag_overlap + min(token_overlap, 4)
            if is_seed:
                score += 4
            if score <= 0:
                continue
            scored.append((score, _recency(skill.get("ts")), skill))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        for _, _, skill in scored:
            sid = str(skill.get("id"))
            if sid in selected_ids:
                continue
            selected.append(skill)
            selected_ids.add(sid)
            if len(selected) >= len(_2WIKI_DEFAULT_SKILL_IDS) + max(0, k):
                break
        return selected

    def _lesson_file_key(self) -> tuple[int, int]:
        try:
            stat = self.lessons_path.stat()
        except FileNotFoundError:
            return (0, 0)
        return (stat.st_mtime_ns, stat.st_size)

    def _lesson_candidates(self, task: str | None) -> list[dict]:
        if task == "2wiki" and not _enabled_env("SII_2WIKI_ENABLE_LESSONS"):
            return []
        candidates: list[dict] = []
        if (
            task == "2wiki"
            and not _enabled_env("SII_2WIKI_ENABLE_SKILLS")
            and _enabled_env("SII_2WIKI_ENABLE_LESSONS")
        ):
            candidates.extend(dict(lesson) for lesson in _2WIKI_SEED_LESSONS)
        for lesson in self.all_lessons():
            failure_mode = _clean_value(lesson.get("failure_mode")).lower()
            if task == "2wiki" and failure_mode in {"", "none", "n/a", "na", "supported_answer", "correct", "already_correct"}:
                continue
            candidates.append(lesson)
        return candidates

    def _lesson_bm25(self, task: str | None) -> dict[str, object]:
        key = (
            task or "",
            _enabled_env("SII_2WIKI_ENABLE_LESSONS"),
            _enabled_env("SII_2WIKI_ENABLE_SKILLS"),
            self._lesson_file_key(),
        )
        with self._lock:
            if self._lesson_bm25_key == key and self._lesson_bm25_index is not None:
                return self._lesson_bm25_index

            docs = []
            df: Counter[str] = Counter()
            for lesson in self._lesson_candidates(task):
                tokens = _token_list(_lesson_text(lesson))
                counts = Counter(tokens)
                length = sum(counts.values())
                if length == 0:
                    continue
                df.update(counts.keys())
                tags = " ".join(_clean_list(lesson.get("tags")))
                docs.append(
                    {
                        "lesson": lesson,
                        "counts": counts,
                        "length": length,
                        "features": _question_features(
                            f"{lesson.get('question', '')} {lesson.get('reusable_lesson', '')} {tags}"
                        ),
                        "recency": _recency(lesson.get("ts")),
                    }
                )

            n_docs = len(docs)
            avgdl = (sum(int(doc["length"]) for doc in docs) / n_docs) if n_docs else 0.0
            idf = {
                term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
                for term, freq in df.items()
            }
            self._lesson_bm25_key = key
            self._lesson_bm25_index = {
                "docs": docs,
                "avgdl": avgdl,
                "idf": idf,
            }
            return self._lesson_bm25_index

    def _lesson_embedding(self, task: str | None) -> dict[str, object] | None:
        key = (
            task or "",
            _enabled_env("SII_2WIKI_ENABLE_LESSONS"),
            _enabled_env("SII_2WIKI_ENABLE_SKILLS"),
            self._lesson_file_key(),
            os.getenv("SII_MEMORY_EMBEDDING_DIMS", str(_LESSON_EMBEDDING_DIMS)),
            os.getenv("SII_MEMORY_EMBEDDING_MAX_FEATURES", str(_LESSON_EMBEDDING_MAX_FEATURES)),
        )
        with self._lock:
            if self._lesson_embedding_key == key:
                return self._lesson_embedding_index

            lessons = self._lesson_candidates(task)
            texts = [_lesson_text(lesson) for lesson in lessons]
            docs = [
                {
                    "lesson": lesson,
                    "features": _question_features(
                        f"{lesson.get('question', '')} {lesson.get('reusable_lesson', '')} "
                        f"{' '.join(_clean_list(lesson.get('tags')))}"
                    ),
                    "recency": _recency(lesson.get("ts")),
                }
                for lesson, text in zip(lessons, texts)
                if text.strip()
            ]
            texts = [text for text in texts if text.strip()]
            if not docs:
                self._lesson_embedding_key = key
                self._lesson_embedding_index = None
                return None

            try:
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
            except Exception:
                self._lesson_embedding_key = key
                self._lesson_embedding_index = None
                return None

            try:
                max_features = max(128, int(os.getenv("SII_MEMORY_EMBEDDING_MAX_FEATURES", str(_LESSON_EMBEDDING_MAX_FEATURES))))
            except ValueError:
                max_features = _LESSON_EMBEDDING_MAX_FEATURES
            try:
                requested_dims = max(2, int(os.getenv("SII_MEMORY_EMBEDDING_DIMS", str(_LESSON_EMBEDDING_DIMS))))
            except ValueError:
                requested_dims = _LESSON_EMBEDDING_DIMS

            try:
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b\w\w+\b",
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    max_features=max_features,
                )
                tfidf = vectorizer.fit_transform(texts)
                if tfidf.shape[0] == 0 or tfidf.shape[1] == 0:
                    raise ValueError("empty lesson embedding vocabulary")
                n_components = min(requested_dims, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
                if n_components >= 2:
                    svd = TruncatedSVD(n_components=n_components, random_state=0)
                    doc_vectors = normalize(svd.fit_transform(tfidf), norm="l2")
                    backend = "lsa_svd"
                else:
                    svd = None
                    doc_vectors = normalize(tfidf, norm="l2")
                    backend = "tfidf_cosine"
            except Exception:
                self._lesson_embedding_key = key
                self._lesson_embedding_index = None
                return None

            self._lesson_embedding_key = key
            self._lesson_embedding_index = {
                "backend": backend,
                "docs": docs,
                "vectorizer": vectorizer,
                "svd": svd,
                "doc_vectors": doc_vectors,
            }
            return self._lesson_embedding_index

    @staticmethod
    def _bm25_score(
        query_counts: Counter[str],
        doc_counts: Counter[str],
        doc_len: int,
        avgdl: float,
        idf: dict[str, float],
    ) -> float:
        if not query_counts or doc_len <= 0 or avgdl <= 0:
            return 0.0
        score = 0.0
        norm = _LESSON_BM25_K1 * (1.0 - _LESSON_BM25_B + _LESSON_BM25_B * doc_len / avgdl)
        for term, qtf in query_counts.items():
            tf = doc_counts.get(term, 0)
            if tf <= 0:
                continue
            score += idf.get(term, 0.0) * ((tf * (_LESSON_BM25_K1 + 1.0)) / (tf + norm)) * min(qtf, 3)
        return score

    def _retrieve_lessons_embedding(self, question: str, k: int, task: str | None) -> list[dict]:
        index = self._lesson_embedding(task)
        if not index:
            return []
        query = _original_question(question) or question
        qfeatures = _question_features(question)
        try:
            from sklearn.preprocessing import normalize
        except Exception:
            return []
        vectorizer = index.get("vectorizer")
        doc_vectors = index.get("doc_vectors")
        docs = index.get("docs")
        if vectorizer is None or doc_vectors is None or not isinstance(docs, list):
            return []
        try:
            q = vectorizer.transform([query])
            svd = index.get("svd")
            if svd is not None:
                q_vec = normalize(svd.transform(q), norm="l2")
                sims = doc_vectors @ q_vec[0]
            else:
                q_vec = normalize(q, norm="l2")
                sims = (q_vec @ doc_vectors.T).toarray()[0]
        except Exception:
            return []

        scored: list[tuple[float, float, dict]] = []
        query_features = qfeatures - {"2wiki"}
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue
            lesson = doc.get("lesson")
            if not isinstance(lesson, dict):
                continue
            features = doc.get("features")
            feature_overlap = len(query_features & (features if isinstance(features, set) else set()))
            recency = float(doc.get("recency") or 0.0)
            score = float(sims[i]) + _LESSON_EMBEDDING_FEATURE_BOOST * feature_overlap + _LESSON_EMBEDDING_RECENCY_BOOST * recency
            if score > 0:
                scored.append((score, recency, lesson))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [lesson for _, _, lesson in scored[:k]]

    def _retrieve_lessons_bm25(self, question: str, k: int, task: str | None) -> list[dict]:
        query = _original_question(question) or question
        query_counts = Counter(_token_list(query))
        qfeatures = _question_features(question)
        index = self._lesson_bm25(task)
        docs = index.get("docs", [])
        avgdl = float(index.get("avgdl") or 0.0)
        idf = index.get("idf", {})
        if not isinstance(docs, list) or not isinstance(idf, dict):
            return []

        scored: list[tuple[float, float, dict]] = []
        query_features = qfeatures - {"2wiki"}
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            lesson = doc.get("lesson")
            counts = doc.get("counts")
            if not isinstance(lesson, dict) or not isinstance(counts, Counter):
                continue
            bm25 = self._bm25_score(query_counts, counts, int(doc.get("length") or 0), avgdl, idf)
            features = doc.get("features")
            feature_overlap = len(query_features & (features if isinstance(features, set) else set()))
            recency = float(doc.get("recency") or 0.0)
            score = bm25 + _LESSON_FEATURE_BOOST * feature_overlap + _LESSON_RECENCY_BOOST * recency
            if score > 0:
                scored.append((score, recency, lesson))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [lesson for _, _, lesson in scored[:k]]

    def retrieve_lessons(self, question: str, k: int = 3, task: str | None = None) -> list[dict]:
        mode = os.getenv("SII_MEMORY_RETRIEVAL", "embedding").strip().lower()
        if mode in {"bm25", "keyword", "lexical"}:
            return self._retrieve_lessons_bm25(question, k, task)
        lessons = self._retrieve_lessons_embedding(question, k, task)
        if lessons:
            return lessons
        return self._retrieve_lessons_bm25(question, k, task)

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
        include_skills: bool = True,
        skill_k: int = 1,
    ) -> str:
        skills = self.retrieve_skills(question, k=skill_k, task=task) if include_skills else []
        lesson_k = min(k, 1) if task == "2wiki" and _enabled_env("SII_2WIKI_ENABLE_SKILLS") else k
        lessons = self.retrieve_lessons(question, k=lesson_k, task=task)
        successes = self.retrieve_successes(question, k=max(1, min(2, k))) if include_successes else []
        if not skills and not lessons and not successes:
            if task != "2wiki":
                return ""
        sections: list[str] = []
        if task == "2wiki" and _enabled_env("SII_2WIKI_ENABLE_TYPED_POLICIES"):
            policy_block = _render_2wiki_policy_cards(question)
            if policy_block:
                sections.append(policy_block)
        if skills:
            bullets = []
            for skill in skills:
                steps = _clean_list(skill.get("steps"))[:2]
                if not steps:
                    continue
                verifier = _clean_list(skill.get("verifier"))[:1]
                line = f"- [{skill.get('id')}] {skill.get('title')}: " + " ".join(steps)
                if verifier:
                    line += f" Check: {verifier[0]}"
                bullets.append(line)
            if bullets:
                sections.append(
                    "2Wiki retrieved skills (apply only when relevant; provided context and answer-format rules still win):\n"
                    + "\n".join(bullets)
                )
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
