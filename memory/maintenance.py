"""Conservative full-scan maintenance for file-backed agent memory."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

_NOOP_VALUES = {
    "",
    "none",
    "n/a",
    "na",
    "null",
    "no strategy needed",
    "none needed",
    "already correct",
    "answer is correct",
    "no corrective action",
    "no issue",
}
_BAD_MEMORY_PHRASES = (
    "do not answer",
    "cannot answer",
    "withhold",
    "withheld",
    "refuse",
    "privacy",
    "private",
    "sensitive",
    "state uncertainty",
    "acknowledge uncertainty",
)
_CONTRADICTION_PAIRS = (
    ("always search", "never search"),
    ("always search", "do not search"),
    ("ignore provided context", "use provided context"),
    ("ignore context", "context first"),
    ("answer with full sentence", "return only the final answer span"),
    ("return a full sentence", "return only the final answer span"),
)
_GENERALIZE_REPLACEMENTS = {
    "2WikiMultihopQA": "factual QA",
    "2wiki": "factual QA",
    "gold standard": "requested answer format",
    "gold answer": "requested answer",
    "gold label": "requested answer format",
    "gold token": "requested answer token",
    "gold string": "requested answer string",
    "gold": "requested target",
    "benchmark": "task",
    "dataset": "task set",
}
_BAD_TAGS = {"2wiki", "benchmark", "dataset", "gold"}
_UPDATE_ALLOWED_FIELDS = {
    "lessons": {
        "failure_mode",
        "root_cause",
        "corrective_strategy",
        "reusable_lesson",
        "outcome",
        "score",
        "tags",
    },
    "skills": {
        "task",
        "title",
        "description",
        "tags",
        "triggers",
        "steps",
        "verifier",
        "bad_patterns",
        "source",
        "score",
    },
    "episodes": set(),
}


def _path(root: Path, kind: str) -> Path:
    return root / f"{kind}.jsonl"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(item, dict):
            rows.append(item)
        else:
            invalid += 1
    return rows, invalid


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def _flatten(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(val)}" for key, val in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    return str(value or "")


def _normalize(text: object) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", _flatten(text).lower())
    return re.sub(r"\s+", " ", value).strip()


def _clean_value(value: object) -> str:
    text = " ".join(str(value or "").split())
    return "" if text.lower() in _NOOP_VALUES else text


def _generalize_text(value: object) -> str:
    text = _clean_value(value)
    for old, new in _GENERALIZE_REPLACEMENTS.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\((?:e\.g\.|for example)[^)]*\)", "", text, flags=re.IGNORECASE)
    return text.strip()


def _generalize_list(value: object, limit: int = 16) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _generalize_text(item)
        if text and text.lower() not in _BAD_TAGS:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _generalize_question(value: object) -> str:
    text = str(value or "")
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return "Question pattern: " + _generalize_text(match.group(1))[:500]
    text = _generalize_text(text)
    return "Question pattern: " + text[:500] if text else ""


def _record_id(kind: str, item: dict[str, Any]) -> str:
    explicit = str(item.get("id") or item.get("memory_id") or "").strip()
    if explicit:
        return explicit
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return f"{kind.rstrip('s')}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _action(kind: str, action: str, item: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "action": action,
        "id": _record_id(kind, item),
        "reason": reason,
    }
    payload.update(extra)
    return payload


def _has_bad_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BAD_MEMORY_PHRASES)


def _has_obvious_contradiction(text: str) -> str | None:
    lowered = text.lower()
    for left, right in _CONTRADICTION_PAIRS:
        if left in lowered and right in lowered:
            return f"contains both '{left}' and '{right}'"
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM maintenance response must be a JSON object")
    return parsed


def _lesson_key(item: dict[str, Any]) -> str:
    return "|".join(
        _normalize(item.get(key))
        for key in ("failure_mode", "reusable_lesson", "corrective_strategy")
    )


def _skill_key(item: dict[str, Any]) -> str:
    return "|".join(
        _normalize(item.get(key))
        for key in ("title", "description", "triggers", "steps", "verifier")
    )


def _episode_key(item: dict[str, Any]) -> str:
    return "|".join(
        _normalize(item.get(key))
        for key in ("question", "answer", "correct")
    )


def _question_key(item: dict[str, Any]) -> str:
    return _normalize(item.get("question"))


def _rank(item: dict[str, Any]) -> tuple[float, float]:
    try:
        score = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        ts = float(item.get("updated_ts") or item.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    return score, ts


def _dedupe(
    kind: str,
    rows: list[dict[str, Any]],
    key_fn,
    actions: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = key_fn(row)
        if not key:
            continue
        groups.setdefault(key, []).append(row)
    remove_ids: set[int] = set()
    for group in groups.values():
        if len(group) <= 1:
            continue
        keeper = max(group, key=_rank)
        for row in group:
            if row is keeper:
                continue
            remove_ids.add(id(row))
            actions.append(_action(kind, "delete", row, reason, kept_id=_record_id(kind, keeper)))
    return [row for row in rows if id(row) not in remove_ids]


def _clean_lessons(rows: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in rows:
        original = dict(item)
        item.setdefault("id", _record_id("lessons", item))
        item.setdefault("outcome", "failure")
        item["question"] = _generalize_question(item.get("question"))
        for key in ("failure_mode", "root_cause", "corrective_strategy", "reusable_lesson", "outcome"):
            if key in item:
                item[key] = _generalize_text(item.get(key))
        if isinstance(item.get("tags"), list):
            item["tags"] = _generalize_list(item.get("tags"))
        lesson = _clean_value(item.get("reusable_lesson"))
        strategy = _clean_value(item.get("corrective_strategy"))
        combined = _flatten(item)
        contradiction = _has_obvious_contradiction(combined)
        if not lesson and not strategy:
            actions.append(_action("lessons", "delete", item, "empty_lesson_and_strategy"))
            continue
        if _has_bad_phrase(combined):
            actions.append(_action("lessons", "delete", item, "harmful_or_refusal_memory"))
            continue
        if contradiction:
            actions.append(_action("lessons", "delete", item, f"obvious_contradiction: {contradiction}"))
            continue
        if item != original:
            actions.append(_action("lessons", "update", item, "backfill_id_or_defaults"))
        kept.append(item)
    return _dedupe("lessons", kept, _lesson_key, actions, "duplicate_lesson")


def _clean_skills(rows: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in rows:
        original = dict(item)
        item.setdefault("id", _record_id("skills", item))
        item.setdefault("task", "general")
        item["task"] = _generalize_text(item.get("task")) or "general"
        if item["task"].lower() in _BAD_TAGS:
            item["task"] = "general"
        for key in ("title", "description", "source"):
            if key in item:
                item[key] = _generalize_text(item.get(key))
        for key in ("tags", "triggers", "steps", "verifier", "bad_patterns"):
            if isinstance(item.get(key), list):
                item[key] = _generalize_list(item.get(key))
        combined = _flatten(item)
        steps = _clean_value(_flatten(item.get("steps")))
        description = _clean_value(item.get("description"))
        contradiction = _has_obvious_contradiction(combined)
        if not steps and not description:
            actions.append(_action("skills", "delete", item, "empty_skill"))
            continue
        if _has_bad_phrase(combined):
            actions.append(_action("skills", "delete", item, "harmful_or_refusal_skill"))
            continue
        if contradiction:
            actions.append(_action("skills", "delete", item, f"obvious_contradiction: {contradiction}"))
            continue
        if item != original:
            actions.append(_action("skills", "update", item, "backfill_id_or_defaults"))
        kept.append(item)
    return _dedupe("skills", kept, _skill_key, actions, "duplicate_skill")


def _clean_episodes(rows: list[dict[str, Any]], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in rows:
        original = dict(item)
        item.setdefault("id", _record_id("episodes", item))
        question = _clean_value(item.get("question"))
        answer = _clean_value(item.get("answer"))
        if not question:
            actions.append(_action("episodes", "delete", item, "empty_question"))
            continue
        if not answer and item.get("stop_reason") == "final":
            actions.append(_action("episodes", "delete", item, "final_episode_without_answer"))
            continue
        if item != original:
            actions.append(_action("episodes", "update", item, "backfill_id"))
        kept.append(item)

    correct_by_question = {
        _question_key(item)
        for item in kept
        if _question_key(item) and item.get("correct") is True
    }
    filtered: list[dict[str, Any]] = []
    for item in kept:
        if item.get("correct") is False and _question_key(item) in correct_by_question:
            actions.append(_action("episodes", "delete", item, "incorrect_duplicate_for_question_with_correct_episode"))
            continue
        filtered.append(item)
    return _dedupe("episodes", filtered, _episode_key, actions, "duplicate_episode")


def _truncate(text: object, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    clipped = value[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return (clipped or value[:limit]).rstrip() + "..."


def _compact_record(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    base = {"kind": kind, "id": _record_id(kind, item)}
    if kind == "lessons":
        base.update(
            {
                "question": _truncate(item.get("question"), 260),
                "failure_mode": _truncate(item.get("failure_mode"), 120),
                "root_cause": _truncate(item.get("root_cause"), 260),
                "corrective_strategy": _truncate(item.get("corrective_strategy"), 320),
                "reusable_lesson": _truncate(item.get("reusable_lesson"), 320),
                "outcome": item.get("outcome"),
                "score": item.get("score"),
            }
        )
    elif kind == "skills":
        base.update(
            {
                "task": item.get("task"),
                "title": _truncate(item.get("title"), 160),
                "description": _truncate(item.get("description"), 260),
                "tags": item.get("tags"),
                "triggers": item.get("triggers"),
                "steps": item.get("steps"),
                "verifier": item.get("verifier"),
                "bad_patterns": item.get("bad_patterns"),
                "score": item.get("score"),
            }
        )
    else:
        base.update(
            {
                "question": _truncate(item.get("question"), 360),
                "answer": _truncate(item.get("answer"), 180),
                "correct": item.get("correct"),
                "steps": item.get("steps"),
                "tool_calls": item.get("tool_calls"),
                "tool_call_counts": item.get("tool_call_counts"),
                "stop_reason": item.get("stop_reason"),
            }
        )
    return base


def _llm_review_prompt(kind: str, records: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are the memory curator for a general ReAct agent that learns across tasks. "
        "Review concrete long-term memory records and decide which records should be kept, deleted, or rewritten. "
        "Be conservative: keep useful or merely imperfect memories. Delete only clearly harmful, empty, duplicate, "
        "self-contradictory, or wrong records. For episodes, do not rewrite answers/correctness; only delete episodes "
        "that are clearly useless duplicates, empty, failed records superseded by a correct same-question episode in this batch, "
        "or obviously malformed. For lessons/skills, update concise reusable strategy wording when it is clearly salvageable. "
        "For skills, rewrite generic titles into 3-8 word action phrases that summarize the actual procedure. "
        "Return JSON only."
    )
    user = {
        "kind": kind,
        "instructions": {
            "allowed_actions": ["keep", "delete", "update"],
            "schema": {
                "actions": [
                    {
                        "id": "record id",
                        "action": "delete or update; omit keep actions unless explaining a group decision",
                        "reason": "short concrete reason",
                        "patch": "object, only for update; do not include id",
                    }
                ]
            },
            "delete_when": [
                "empty or no reusable content",
                "refusal/privacy/withhold/uncertainty policy that hurts task performance",
                "internally contradictory instructions",
                "near duplicate where another record is clearly better",
                "episode is incorrect/no-answer and a correct same-question episode appears in this batch",
            ],
            "update_when": [
                "lesson/skill has a useful idea but wording is vague, contradictory, or too verbose",
                "skill title is generic, dataset-specific, or fails to summarize the concrete procedure",
                "patch only fields that need correction",
            ],
            "generalization_rules": [
                "do not mention gold, benchmark names, dataset names, or exact examples",
                "rewrite task-specific wording as general requested-answer/output-format guidance",
                "tags must be general reasoning/tool/task-pattern tags",
            ],
            "skill_title_rules": [
                "3-8 words",
                "summarize the actual skill content, not its origin",
                "prefer action phrases such as 'Verify relation before answering'",
                "never use 'Reflection update', 'Reflection skill', dataset names, or broad placeholders",
            ],
        },
        "records": records,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _review_batch_with_llm(kind: str, rows: list[dict[str, Any]], batch_index: int) -> dict[str, Any]:
    from agent.llm import chat

    records = [_compact_record(kind, row) for row in rows]
    response = chat(
        _llm_review_prompt(kind, records),
        tools=None,
        temperature=0,
        max_tokens=4096,
        timeout=300,
    )
    content = response.choices[0].message.content or ""
    parsed = _parse_json_object(content)
    actions = parsed.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    return {
        "kind": kind,
        "batch_index": batch_index,
        "records": len(rows),
        "raw_actions": actions,
        "raw_response": content[:4000],
    }


def _apply_llm_reviews(
    kind: str,
    rows: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {_record_id(kind, row): row for row in rows}
    delete_ids: set[str] = set()
    for review in reviews:
        for decision in review.get("raw_actions", []):
            if not isinstance(decision, dict):
                continue
            item_id = str(decision.get("id") or "").strip()
            action = str(decision.get("action") or "").strip().lower()
            reason = _truncate(decision.get("reason"), 240)
            if item_id not in by_id or action in {"", "keep"}:
                continue
            if action == "delete":
                delete_ids.add(item_id)
                actions.append(
                    _action(kind, "delete", by_id[item_id], f"llm_review: {reason or 'delete'}")
                )
                continue
            if action == "update" and kind in {"lessons", "skills"}:
                patch = decision.get("patch")
                if not isinstance(patch, dict):
                    continue
                allowed = _UPDATE_ALLOWED_FIELDS[kind]
                clean_patch = {
                    key: value
                    for key, value in patch.items()
                    if key in allowed and key not in {"id", "memory_id"}
                }
                if not clean_patch:
                    continue
                row = by_id[item_id]
                row.update(clean_patch)
                row["id"] = item_id
                row["updated_ts"] = time.time()
                actions.append(
                    _action(kind, "update", row, f"llm_review: {reason or 'rewrite'}", patch=clean_patch)
                )
    return [row for row in rows if _record_id(kind, row) not in delete_ids]


def _review_rows_with_llm(
    kind: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {"kind": kind, "records": 0, "batches": 0, "errors": []}
    ordered = list(rows)
    if kind == "episodes":
        ordered.sort(key=lambda row: (_question_key(row), str(row.get("correct")), _record_id(kind, row)))
    else:
        ordered.sort(key=lambda row: _record_id(kind, row))

    reviews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    size = max(1, batch_size)
    for batch_index, start in enumerate(range(0, len(ordered), size), 1):
        batch = ordered[start:start + size]
        try:
            reviews.append(_review_batch_with_llm(kind, batch, batch_index))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "kind": kind,
                    "batch_index": batch_index,
                    "records": len(batch),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    reviewed = _apply_llm_reviews(kind, ordered, reviews, actions)
    return reviewed, {
        "kind": kind,
        "records": len(rows),
        "batches": len(reviews),
        "errors": errors,
        "raw_action_count": sum(len(review.get("raw_actions", [])) for review in reviews),
    }


def _select_overall_records(cleaned_by_kind: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for kind, limit in (("lessons", 70), ("skills", 70), ("episodes", 30)):
        rows = list(cleaned_by_kind.get(kind) or [])
        if kind == "episodes":
            rows.sort(key=lambda row: (_rank(row)[1], row.get("correct") is True), reverse=True)
        else:
            rows.sort(key=_rank, reverse=True)
        selected[kind] = [_compact_record(kind, row) for row in rows[:limit]]
    return selected


def _fallback_overall_guidance(
    cleaned_by_kind: dict[str, list[dict[str, Any]]],
    max_chars: int,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    counts = {kind: len(rows) for kind, rows in cleaned_by_kind.items()}
    categories = [
        {
            "pattern": "Search-heavy factual questions",
            "guidance": "Use focused queries, read at least one strong source, and cross-check names, dates, relations, and units before finalizing.",
        },
        {
            "pattern": "Multi-entity or relation questions",
            "guidance": "Identify the requested target first, trace each relation direction explicitly, and avoid answering an intermediate entity.",
        },
        {
            "pattern": "Memory use",
            "guidance": "Treat memory as process guidance only; verify the current answer from current context/tools and inspect full memory before update/delete.",
        },
    ]
    overall = (
        "Use memory as procedural guidance, not evidence. For search tasks, form focused queries, compare independent sources, "
        "and verify entity identity plus answer granularity before final_answer. For relation tasks, track direction and target. "
        "When memory conflicts with current evidence, trust current evidence and update/delete the bad memory only after inspecting its id."
    )
    return {
        "ts": time.time(),
        "source": "fallback",
        "source_counts": counts,
        "overall": _truncate(overall, max_chars),
        "categories": categories,
        "avoid": [
            "Do not copy a remembered answer without current evidence.",
            "Do not keep stale, duplicate, or refusal-shaped memory.",
        ],
        "error": error,
    }


def _sanitize_overall_text(text: object) -> str:
    value = " ".join(str(text or "").split())
    value = re.sub(r"\s*\((?:e\.g\.|for example)[^)]*\)", "", value, flags=re.IGNORECASE)
    replacements = {
        "gold standard": "requested answer format",
        "gold answer": "requested answer",
        "gold granularity": "requested granularity",
        "gold format": "requested format",
        "gold": "requested target",
        "benchmark": "task",
        "dataset": "task set",
    }
    for old, new in replacements.items():
        value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
    return value


def _overall_prompt(
    cleaned_by_kind: dict[str, list[dict[str, Any]]],
    trigger: dict[str, Any],
    max_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You write compact overall guidance for a general ReAct agent from its curated long-term memory. "
        "The guidance will be inserted into future system prompts, so it must be short, actionable, and benchmark-agnostic. "
        "Do not mention gold answers, gold standards, benchmark names, dataset names, validation/test splits, exact examples, record ids, or one-off facts. "
        "The output must not contain the word 'gold' and must not contain e.g./for example parentheticals. "
        "If records mention provided context or answer formats, generalize that into current-evidence and output-discipline guidance. "
        "Group guidance by reusable problem pattern, e.g. search-heavy tasks, relation/entity tasks, comparison tasks, output discipline, memory hygiene. "
        "Emphasize verification, cross-checking, answer-target granularity, and when to update/delete bad memory. Return JSON only."
    )
    user = {
        "trigger": trigger,
        "max_overall_chars": max_chars,
        "output_schema": {
            "overall": f"single prompt-ready paragraph, <= {max_chars} chars",
            "categories": [{"pattern": "problem type", "guidance": "one concrete strategy"}],
            "avoid": ["pitfalls to avoid"],
        },
        "style_rules": [
            "general, not benchmark-specific",
            "never say gold answer/gold standard",
            "do not use examples or entity names",
            "never copy exact entity names from records",
            "write complete sentences that remain useful across tasks",
        ],
        "memory_records": _select_overall_records(cleaned_by_kind),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _generate_overall_guidance(
    root: Path,
    cleaned_by_kind: dict[str, list[dict[str, Any]]],
    trigger: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    if os.getenv("SII_MEMORY_OVERALL_LLM", "1").strip().lower() in {"0", "false", "no"}:
        return _fallback_overall_guidance(cleaned_by_kind, max_chars)
    try:
        from agent.llm import chat

        response = chat(
            _overall_prompt(cleaned_by_kind, trigger, max_chars),
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1800,
            timeout=float(os.getenv("SII_MEMORY_OVERALL_TIMEOUT", "180")),
        )
        parsed = _parse_json_object(response.choices[0].message.content or "")
        if not parsed:
            raise ValueError("overall guidance response was not a JSON object")
        categories: list[dict[str, str]] = []
        for item in parsed.get("categories") or []:
            if not isinstance(item, dict):
                continue
            pattern = _truncate(_sanitize_overall_text(item.get("pattern")), 80)
            guidance = _truncate(_sanitize_overall_text(item.get("guidance")), 260)
            if pattern and guidance:
                categories.append({"pattern": pattern, "guidance": guidance})
            if len(categories) >= 8:
                break
        overall = _truncate(_sanitize_overall_text(parsed.get("overall")), max_chars)
        if not overall:
            raise ValueError("overall guidance missing overall text")
        avoid: list[str] = []
        for item in parsed.get("avoid") or []:
            text = _truncate(_sanitize_overall_text(item), 180)
            if text:
                avoid.append(text)
            if len(avoid) >= 6:
                break
        return {
            "ts": time.time(),
            "source": "llm",
            "source_counts": {kind: len(rows) for kind, rows in cleaned_by_kind.items()},
            "overall": overall,
            "categories": categories,
            "avoid": avoid,
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {_truncate(exc, 240)}"
        if _require_llm_overall():
            existing = _existing_overall(root, error)
            if existing is not None:
                return existing
            return {
                "ts": time.time(),
                "source": "llm_error",
                "source_counts": {kind: len(rows) for kind, rows in cleaned_by_kind.items()},
                "overall": "",
                "categories": [],
                "avoid": [],
                "error": error,
            }
        return _fallback_overall_guidance(
            cleaned_by_kind,
            max_chars,
            error=error,
        )


def _require_llm_overall() -> bool:
    return os.getenv("SII_MEMORY_OVERALL_REQUIRE_LLM", "1").strip().lower() not in {"0", "false", "no"}


def _existing_overall(root: Path, error: str) -> dict[str, Any] | None:
    path = root / "overall.json"
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed = dict(parsed)
    parsed["refresh_error"] = error
    parsed["refresh_ts"] = time.time()
    return parsed


def _write_overall(root: Path, overall: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    if overall.get("source") == "llm_error":
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / "overall_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(overall, ensure_ascii=False) + "\n")


def maintain_memory(
    root: str | Path,
    *,
    report_path: str | Path | None = None,
    trigger: dict[str, Any] | None = None,
    dry_run: bool = False,
    llm_review: bool = False,
    llm_batch_size: int = 30,
    overall_max_chars: int = 1600,
) -> dict[str, Any]:
    """Full-scan memory maintenance with conservative rules and optional LLM review."""
    root = Path(root)
    started = time.time()
    actions: list[dict[str, Any]] = []
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    invalid_lines: dict[str, int] = {}
    llm_reports: list[dict[str, Any]] = []
    overall_report: dict[str, Any] | None = None
    cleaned_by_kind: dict[str, list[dict[str, Any]]] = {}
    cleaners = {
        "lessons": _clean_lessons,
        "skills": _clean_skills,
        "episodes": _clean_episodes,
    }

    for kind, cleaner in cleaners.items():
        path = _path(root, kind)
        rows, invalid = _read_jsonl(path)
        before[kind] = len(rows)
        invalid_lines[kind] = invalid
        if invalid:
            actions.append({"kind": kind, "action": "delete", "id": None, "reason": f"invalid_json_lines:{invalid}"})
        cleaned = cleaner(rows, actions)
        cleaned_by_kind[kind] = cleaned

    if llm_review:
        for kind in ("lessons", "skills", "episodes"):
            reviewed, llm_report = _review_rows_with_llm(
                kind,
                cleaned_by_kind[kind],
                batch_size=llm_batch_size,
                actions=actions,
            )
            cleaned_by_kind[kind] = reviewed
            llm_reports.append(llm_report)

    for kind, cleaned in cleaned_by_kind.items():
        after[kind] = len(cleaned)
        if not dry_run:
            _write_jsonl(_path(root, kind), cleaned)

    overall_limit = max(400, min(2400, int(overall_max_chars or 1600)))
    overall_report = _generate_overall_guidance(root, cleaned_by_kind, trigger or {}, overall_limit)
    _write_overall(root, overall_report, dry_run)

    report = {
        "ts": started,
        "root": str(root),
        "dry_run": dry_run,
        "llm_review": llm_review,
        "llm_batch_size": llm_batch_size if llm_review else None,
        "trigger": trigger or {},
        "before": before,
        "after": after,
        "invalid_lines": invalid_lines,
        "llm_reports": llm_reports,
        "overall": overall_report,
        "actions_count": len(actions),
        "actions": actions[:200],
        "truncated_actions": max(0, len(actions) - 200),
        "elapsed": time.time() - started,
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report
