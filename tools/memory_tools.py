"""Agent-facing memory retrieval tools."""
from __future__ import annotations

import json
import os
import time
import hashlib
import re
from pathlib import Path
from typing import Any

from memory.store import MemoryStore, _2WIKI_SEED_LESSONS, _2WIKI_SEED_SKILLS

from .registry import register

_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_MAX_QUERIES = 6
_KIND_LABELS = {
    "lessons": "lesson",
    "episodes": "episode",
    "skills": "skill",
}


def _memory_root() -> Path:
    return Path(os.getenv("SII_AGENT_MEMORY_ROOT", os.getenv("MEMORY_ROOT", "logs/memory")))


def _runtime_mode() -> str:
    mode = os.getenv("SII_AGENT_RUNTIME_MODE", "train").strip().lower()
    return mode if mode in {"train", "test"} else "train"


def _memory_read_only() -> bool:
    return _runtime_mode() == "test"


def _write_blocked(tool_name: str) -> str | None:
    if not _memory_read_only():
        return None
    return json.dumps(
        {
            "error": f"{tool_name} is disabled in test runtime mode; memory is read-only.",
            "runtime_mode": _runtime_mode(),
            "read_only": True,
        },
        ensure_ascii=False,
        indent=2,
    )


def _kind_path(kind: str, *, create: bool = False) -> Path:
    normalized = str(kind or "").strip().lower()
    aliases = {
        "lesson": "lessons",
        "lessons": "lessons",
        "episode": "episodes",
        "episodes": "episodes",
        "skill": "skills",
        "skills": "skills",
    }
    if normalized not in aliases:
        raise ValueError("kind must be one of: lessons, episodes, skills")
    root = _memory_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root / f"{aliases[normalized]}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def _record_id(kind: str, item: dict[str, Any]) -> str:
    explicit = str(item.get("id") or item.get("memory_id") or "").strip()
    if explicit:
        return explicit
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return f"{kind.rstrip('s')}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _with_id(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    updated = dict(item)
    updated.setdefault("id", _record_id(kind, updated))
    return updated


def _clamp_int(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _short(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit]


def _tokens(text: object) -> set[str]:
    return {tok.lower() for tok in _TOK.findall(str(text or "")) if len(tok) > 1}


def _flatten_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(val)}" for key, val in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(val) for val in value)
    return str(value or "")


def _original_question(text: str) -> str:
    value = str(text or "").strip()
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*?)(?:\n\s*\n|$)", value, flags=re.IGNORECASE | re.DOTALL)
    if match:
        value = match.group(1).strip()
    value = re.split(r"\n\s*Provided context\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return " ".join(value.split())


def _normalize_queries(query: str, queries: list[str] | None) -> list[str]:
    candidates = [str(item or "").strip() for item in (queries or [])]
    if query:
        candidates.insert(0, _original_question(query))
    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        clean = " ".join(str(item or "").split())[:240]
        key = clean.lower()
        if not clean or key in seen:
            continue
        normalized.append(clean)
        seen.add(key)
        if len(normalized) >= _MAX_QUERIES:
            break
    return normalized


def _field_texts(kind: str, item: dict[str, Any]) -> list[tuple[str, int, str]]:
    if kind == "lessons":
        return [
            ("reusable_lesson", 8, _flatten_text(item.get("reusable_lesson"))),
            ("corrective_strategy", 7, _flatten_text(item.get("corrective_strategy"))),
            ("failure_mode", 6, _flatten_text(item.get("failure_mode"))),
            ("root_cause", 5, _flatten_text(item.get("root_cause"))),
            ("tags", 5, _flatten_text(item.get("tags"))),
            ("question", 2, _original_question(_flatten_text(item.get("question")))),
        ]
    if kind == "episodes":
        return [
            ("answer", 6, _flatten_text(item.get("answer"))),
            ("correct", 4, _flatten_text(item.get("correct"))),
            ("stop_reason", 4, _flatten_text(item.get("stop_reason"))),
            ("tool_call_counts", 3, _flatten_text(item.get("tool_call_counts"))),
            ("question", 2, _original_question(_flatten_text(item.get("question")))),
        ]
    return [
        ("id", 7, _flatten_text(item.get("id"))),
        ("title", 7, _flatten_text(item.get("title"))),
        ("triggers", 8, _flatten_text(item.get("triggers"))),
        ("tags", 7, _flatten_text(item.get("tags"))),
        ("steps", 7, _flatten_text(item.get("steps"))),
        ("verifier", 5, _flatten_text(item.get("verifier"))),
        ("description", 4, _flatten_text(item.get("description"))),
        ("bad_patterns", 4, _flatten_text(item.get("bad_patterns"))),
    ]


def _score_record(kind: str, item: dict[str, Any], queries: list[str]) -> tuple[float, list[str], list[str], list[str]]:
    fields = _field_texts(kind, item)
    matched_queries: list[str] = []
    matched_terms: set[str] = set()
    matched_fields: set[str] = set()
    score = 0.0
    for query in queries:
        q_lower = query.lower()
        q_tokens = _tokens(query)
        query_score = 0.0
        for field, weight, text in fields:
            if not text:
                continue
            text_lower = text.lower()
            field_tokens = _tokens(text)
            if len(q_lower) >= 3 and q_lower in text_lower:
                query_score += 12 * weight
                matched_fields.add(field)
            overlap = q_tokens & field_tokens
            if overlap:
                query_score += weight * len(overlap)
                matched_terms.update(overlap)
                matched_fields.add(field)
                if len(q_tokens) > 1 and overlap == q_tokens:
                    query_score += 2 * weight
        if query_score > 0:
            matched_queries.append(query)
            score += query_score
    return score, matched_queries, sorted(matched_terms), sorted(matched_fields)


def _seed_records(kind: str) -> list[tuple[int | None, dict[str, Any]]]:
    if kind == "lessons":
        return [(None, dict(item)) for item in _2WIKI_SEED_LESSONS]
    if kind == "skills":
        return [(None, dict(item)) for item in _2WIKI_SEED_SKILLS]
    return []


def _scan_kind(kind: str, queries: list[str], include_seed: bool) -> tuple[list[dict[str, Any]], int]:
    rows = [(index, item) for index, item in enumerate(_read_jsonl(_kind_path(kind)))]
    scanned = len(rows)
    candidates: list[tuple[str, int | None, dict[str, Any]]] = [("memory", index, item) for index, item in rows]
    if include_seed:
        candidates.extend(("seed", index, item) for index, item in _seed_records(kind))
    results: list[dict[str, Any]] = []
    for source, index, item in candidates:
        score, matched_queries, matched_terms, matched_fields = _score_record(kind, item, queries)
        if score <= 0:
            continue
        if kind == "episodes" and item.get("correct") is True:
            score += 4
        if source == "seed":
            score += 2
        results.append(
            {
                "type": _KIND_LABELS[kind],
                "kind": kind,
                "source": source,
                "source_file": "seed" if source == "seed" else str(_kind_path(kind).name),
                "index": index,
                "id": _record_id(kind, item),
                "score": round(score, 3),
                "matched_queries": matched_queries,
                "matched_terms": matched_terms[:16],
                "matched_fields": matched_fields,
                "item": item,
            }
        )
    results.sort(key=lambda row: (-float(row["score"]), str(row["source"]), str(row["id"])))
    return results, scanned


def _format_memory_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    item = result["item"]
    base = {
        "type": result["type"],
        "id": result["id"],
        "source": result["source"],
        "source_file": result["source_file"],
        "index": result["index"],
        "score": result["score"],
        "matched_queries": result["matched_queries"],
        "matched_terms": result["matched_terms"],
        "matched_fields": result["matched_fields"],
    }
    if result["kind"] == "lessons":
        base["record"] = {
            "failure_mode": _short(item.get("failure_mode"), 160),
            "root_cause": _short(item.get("root_cause"), max_chars),
            "corrective_strategy": _short(item.get("corrective_strategy"), max_chars),
            "reusable_lesson": _short(item.get("reusable_lesson"), max_chars),
            "source_question": _short(_original_question(str(item.get("question", ""))), max_chars),
        }
    elif result["kind"] == "episodes":
        base["record"] = {
            "correct": item.get("correct"),
            "answer": _short(item.get("answer"), 240),
            "steps": item.get("steps"),
            "tool_calls": item.get("tool_calls"),
            "tool_call_counts": item.get("tool_call_counts"),
            "stop_reason": item.get("stop_reason"),
            "source_question": _short(_original_question(str(item.get("question", ""))), max_chars),
        }
    else:
        base["record"] = {
            "title": _short(item.get("title"), 180),
            "description": _short(item.get("description"), max_chars),
            "tags": item.get("tags"),
            "triggers": [_short(step, 120) for step in (item.get("triggers") or [])[:8]],
            "steps": [_short(step, max_chars) for step in (item.get("steps") or [])[:4]],
            "verifier": [_short(step, max_chars) for step in (item.get("verifier") or [])[:3]],
            "bad_patterns": [_short(step, max_chars) for step in (item.get("bad_patterns") or [])[:3]],
        }
    return base


def _compact_memory_ref(result: dict[str, Any]) -> dict[str, Any]:
    item = result["item"]
    base = {
        "kind": result["kind"],
        "type": result["type"],
        "id": result["id"],
        "source": result["source"],
        "score": result["score"],
        "matched_fields": result["matched_fields"][:8],
        "matched_terms": result["matched_terms"][:12],
    }
    if result["kind"] == "lessons":
        base["summary"] = {
            "failure_mode": _short(item.get("failure_mode"), 100),
            "strategy": _short(item.get("corrective_strategy"), 260),
            "lesson": _short(item.get("reusable_lesson"), 260),
        }
    elif result["kind"] == "skills":
        base["summary"] = {
            "title": _short(item.get("title"), 100),
            "steps": [_short(step, 220) for step in (item.get("steps") or [])[:2]],
            "verifier": [_short(step, 180) for step in (item.get("verifier") or [])[:2]],
            "avoid": [_short(step, 160) for step in (item.get("bad_patterns") or [])[:2]],
        }
    else:
        base["summary"] = {
            "correct": item.get("correct"),
            "answer": _short(item.get("answer"), 120),
            "tool_call_counts": item.get("tool_call_counts"),
            "stop_reason": item.get("stop_reason"),
        }
    return base


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_str_list(value: object, limit: int, item_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _short(item, item_chars)
        if text:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _sanitize_guidance_text(text: object) -> str:
    value = " ".join(str(text or "").split())
    value = re.sub(r"\s*\((?:e\.g\.|for example)[^)]*\)", "", value, flags=re.IGNORECASE)
    replacements = {
        "gold standard": "requested answer format",
        "gold answer": "requested answer",
        "gold granularity": "requested granularity",
        "gold format": "requested format",
        "gold text": "requested answer text",
        "gold": "requested target",
        "benchmark": "task",
        "dataset": "task set",
    }
    for old, new in replacements.items():
        value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
    return value


def _fallback_guidance(refs: list[dict[str, Any]], max_chars: int) -> dict[str, Any]:
    apply: list[str] = []
    avoid: list[str] = []
    supporting_refs: list[dict[str, str]] = []
    for ref in refs[:6]:
        summary = ref.get("summary") or {}
        if ref.get("kind") == "lessons":
            lesson = _short(summary.get("lesson") or summary.get("strategy"), 220)
            if lesson:
                apply.append(lesson)
        elif ref.get("kind") == "skills":
            for step in summary.get("steps") or []:
                if step:
                    apply.append(_short(step, 220))
            for bad in summary.get("avoid") or []:
                if bad:
                    avoid.append(_short(bad, 180))
        supporting_refs.append({"kind": str(ref.get("kind")), "id": str(ref.get("id")), "why": "high lexical match"})
        if len(apply) >= 4:
            break
    guidance = " ".join(apply[:3]) or "No strong reusable memory matched; solve from current evidence and verify the answer target."
    return {
        "guidance": _short(guidance, max_chars),
        "apply": apply[:4],
        "avoid": avoid[:3],
        "supporting_refs": supporting_refs[:6],
        "summary_source": "fallback",
    }


def _summarize_memory_guidance(
    query: str,
    queries: list[str],
    refs: list[dict[str, Any]],
    *,
    max_chars: int,
) -> dict[str, Any]:
    if not refs:
        return {
            "guidance": "No relevant reusable memory was found; solve from current evidence and avoid inventing unsupported facts.",
            "apply": [],
            "avoid": [],
            "supporting_refs": [],
            "summary_source": "empty",
        }
    if os.getenv("SII_MEMORY_SEARCH_LLM_SUMMARY", "1").strip().lower() in {"0", "false", "no"}:
        return _fallback_guidance(refs, max_chars)
    try:
        from agent.llm import chat

        system = (
            "You compress retrieved long-term memory for a general ReAct agent. "
            "Do not dump records. Produce only the most useful, task-relevant guidance. "
            "Memory is guidance, not evidence; tell the agent what procedure/checks to apply, not what final answer to give. "
            "Do not mention gold answers, gold standards, benchmark names, dataset names, exact examples, or record dumps. "
            "If records mention gold/benchmark-specific wording, generalize it to requested-answer formatting and current-task verification. "
            "If a record looks stale, harmful, duplicate, or too specific, mention that it should be inspected with memory_get before update/delete. "
            "Return strict JSON only."
        )
        user = {
            "current_question": _short(_original_question(query), 800),
            "queries_used": queries,
            "max_guidance_chars": max_chars,
            "output_schema": {
                "guidance": f"single concise paragraph, <= {max_chars} chars",
                "apply": ["2-5 concrete instructions"],
                "avoid": ["0-4 pitfalls"],
                "supporting_refs": [{"kind": "lessons|skills|episodes", "id": "record id", "why": "short reason"}],
            },
            "style_rules": [
                "general, not benchmark-specific",
                "never say gold answer/gold standard",
                "do not use examples or exact entity names",
                "do not reveal full record content",
            ],
            "retrieved_memory_refs": refs[:10],
        }
        response = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1200,
            timeout=float(os.getenv("SII_MEMORY_SEARCH_SUMMARY_TIMEOUT", "90")),
        )
        parsed = _parse_json_object(response.choices[0].message.content or "")
        if not parsed:
            raise ValueError("memory summary response was not a JSON object")
        supporting_refs: list[dict[str, str]] = []
        known = {(str(ref.get("kind")), str(ref.get("id"))) for ref in refs}
        for item in parsed.get("supporting_refs") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            item_id = str(item.get("id") or "").strip()
            if (kind, item_id) not in known:
                continue
            supporting_refs.append({"kind": kind, "id": item_id, "why": _short(_sanitize_guidance_text(item.get("why")), 140)})
            if len(supporting_refs) >= 6:
                break
        guidance = _short(_sanitize_guidance_text(parsed.get("guidance")), max_chars)
        if not guidance:
            raise ValueError("memory summary missing guidance")
        return {
            "guidance": guidance,
            "apply": _clean_str_list([_sanitize_guidance_text(item) for item in (parsed.get("apply") or [])], 5, 220),
            "avoid": _clean_str_list([_sanitize_guidance_text(item) for item in (parsed.get("avoid") or [])], 4, 180),
            "supporting_refs": supporting_refs or _fallback_guidance(refs, max_chars)["supporting_refs"],
            "summary_source": "llm",
        }
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_guidance(refs, max_chars)
        fallback["summary_error"] = f"{type(exc).__name__}: {_short(exc, 240)}"
        return fallback


@register(
    "memory_search",
    "Search concrete records from global memory. Provide the current question plus 2-6 focused query phrases/entity names; "
    "the tool scans lessons, episodes, and skills, then returns a compressed actionable guidance summary plus record ids.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Current full question."},
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Focused query phrases generated by the agent: entities, relation words, failure modes, or task patterns.",
            },
            "k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
            "task": {"type": "string", "default": "general"},
            "include_lessons": {"type": "boolean", "default": True},
            "include_episodes": {"type": "boolean", "default": True},
            "include_skills": {"type": "boolean", "default": True},
            "include_seed": {"type": "boolean", "default": False},
            "max_chars_per_item": {"type": "integer", "default": 900, "minimum": 120, "maximum": 3000},
            "guidance_max_chars": {"type": "integer", "default": 900, "minimum": 240, "maximum": 1800},
        },
        "required": ["query"],
    },
)
def memory_search(
    query: str,
    queries: list[str] | None = None,
    k: int = 8,
    task: str = "general",
    include_lessons: bool = True,
    include_episodes: bool = True,
    include_skills: bool = True,
    include_seed: bool = False,
    max_chars_per_item: int = 900,
    guidance_max_chars: int = 900,
) -> str:
    k = _clamp_int(k, 8, 1, 20)
    max_chars_per_item = _clamp_int(max_chars_per_item, 900, 120, 3000)
    guidance_max_chars = _clamp_int(guidance_max_chars, 900, 240, 1800)
    root = _memory_root()
    normalized_queries = _normalize_queries(query, queries)
    if not normalized_queries:
        return json.dumps(
            {
                "memory_root": str(root),
                "runtime_mode": _runtime_mode(),
                "read_only": _memory_read_only(),
                "error": "memory_search requires a non-empty query or queries.",
            },
            ensure_ascii=False,
            indent=2,
        )
    kinds = []
    if include_lessons:
        kinds.append("lessons")
    if include_episodes:
        kinds.append("episodes")
    if include_skills:
        kinds.append("skills")
    scanned: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for kind in kinds:
        kind_results, count = _scan_kind(kind, normalized_queries, include_seed=bool(include_seed))
        scanned[kind] = count
        results.extend(kind_results)
    results.sort(key=lambda row: (-float(row["score"]), str(row["kind"]), str(row["id"])))
    compact_refs = [_compact_memory_ref(result) for result in results[:k]]
    guidance = _summarize_memory_guidance(query, normalized_queries, compact_refs, max_chars=guidance_max_chars)
    return json.dumps(
        {
            "memory_root": str(root),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "query": query,
            "queries_used": normalized_queries,
            "scanned_records": scanned,
            "guidance": guidance,
            "record_refs": compact_refs,
            "full_records_suppressed": True,
            "usage_note": (
                "Use the guidance as an action plan, not as evidence. record_refs are compact ids/summaries only; "
                "use memory_get for full content before risky updates/deletes. "
                "Use memory_update/delete only when the returned id clearly identifies stale or bad memory. "
                "Memory is guidance, not evidence; verify current answers from current context/tools."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_stats",
    "Return counts and paths for the configured agent memory folder.",
    {
        "type": "object",
        "properties": {},
    },
)
def memory_stats() -> str:
    root = _memory_root()
    store = MemoryStore(root=root, read_only=True)
    return json.dumps(
        {
            "memory_root": str(root),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "episodes": len(store.all_episodes()),
            "lessons": len(store.all_lessons()),
            "skills": len(store.all_skills()),
            "files": {
                "episodes": str(store.episodes_path),
                "lessons": str(store.lessons_path),
                "skills": str(store.skills_path),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_list",
    "List records from the configured memory folder with stable ids for get/update/delete.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "default": "lessons", "enum": ["lessons", "episodes", "skills"]},
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            "offset": {"type": "integer", "default": 0, "minimum": 0},
            "max_chars_per_item": {"type": "integer", "default": 500, "minimum": 120, "maximum": 2000},
        },
    },
)
def memory_list(kind: str = "lessons", limit: int = 20, offset: int = 0, max_chars_per_item: int = 500) -> str:
    limit = _clamp_int(limit, 20, 1, 100)
    offset = _clamp_int(offset, 0, 0, 1_000_000)
    max_chars_per_item = _clamp_int(max_chars_per_item, 500, 120, 2000)
    path = _kind_path(kind)
    rows = _read_jsonl(path)
    normalized_kind = path.stem
    items = []
    for index, item in enumerate(rows[offset:offset + limit], offset):
        item_with_id = _with_id(normalized_kind, item)
        preview = {key: _short(value, max_chars_per_item) for key, value in item_with_id.items() if key != "id"}
        items.append({"index": index, "id": item_with_id["id"], "item": preview})
    return json.dumps(
        {
            "memory_root": str(_memory_root()),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "kind": normalized_kind,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "items": items,
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_get",
    "Get one memory record by id from lessons, episodes, or skills.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "default": "lessons", "enum": ["lessons", "episodes", "skills"]},
            "id": {"type": "string"},
        },
        "required": ["id"],
    },
)
def memory_get(id: str, kind: str = "lessons") -> str:
    path = _kind_path(kind)
    normalized_kind = path.stem
    for index, item in enumerate(_read_jsonl(path)):
        item_with_id = _with_id(normalized_kind, item)
        if item_with_id["id"] == id:
            return json.dumps(
                {
                    "memory_root": str(_memory_root()),
                    "runtime_mode": _runtime_mode(),
                    "read_only": _memory_read_only(),
                    "kind": normalized_kind,
                    "index": index,
                    "item": item_with_id,
                },
                ensure_ascii=False,
                indent=2,
            )
    return json.dumps(
        {
            "error": f"memory id not found: {id}",
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "kind": normalized_kind,
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_create",
    "Create one training memory record. Use lessons for reusable corrections, skills for procedural know-how, "
    "and episodes only for manually important examples.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "default": "lessons", "enum": ["lessons", "episodes", "skills"]},
            "item": {
                "type": "object",
                "description": (
                    "For lessons include question, failure_mode, root_cause, corrective_strategy, reusable_lesson. "
                    "For skills include title, description, tags, triggers, steps, verifier, bad_patterns."
                ),
            },
        },
        "required": ["item"],
    },
)
def memory_create(item: dict[str, Any], kind: str = "lessons") -> str:
    if not isinstance(item, dict):
        return "ERROR: item must be an object"
    blocked = _write_blocked("memory_create")
    if blocked is not None:
        return blocked
    path = _kind_path(kind, create=True)
    normalized_kind = path.stem
    rows = _read_jsonl(path)
    record = dict(item)
    record.setdefault("ts", time.time())
    if normalized_kind == "lessons":
        record.setdefault("outcome", "manual")
        record.setdefault("failure_mode", "manual")
        record.setdefault("root_cause", "")
        record.setdefault("corrective_strategy", "")
        record.setdefault("reusable_lesson", "")
        record.setdefault("question", "")
    record = _with_id(normalized_kind, record)
    existing_ids = {_record_id(normalized_kind, row) for row in rows}
    if record["id"] in existing_ids:
        return json.dumps({"error": f"memory id already exists: {record['id']}", "id": record["id"]}, ensure_ascii=False, indent=2)
    rows.append(record)
    _write_jsonl(path, rows)
    return json.dumps(
        {
            "memory_root": str(_memory_root()),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "kind": normalized_kind,
            "created": record,
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_update",
    "Patch a stale, wrong, duplicate, or low-quality memory record by id after reading it via memory_search/list/get.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "default": "lessons", "enum": ["lessons", "episodes", "skills"]},
            "id": {"type": "string"},
            "patch": {"type": "object"},
        },
        "required": ["id", "patch"],
    },
)
def memory_update(id: str, patch: dict[str, Any], kind: str = "lessons") -> str:
    if not isinstance(patch, dict):
        return "ERROR: patch must be an object"
    blocked = _write_blocked("memory_update")
    if blocked is not None:
        return blocked
    path = _kind_path(kind, create=True)
    normalized_kind = path.stem
    rows = _read_jsonl(path)
    for index, item in enumerate(rows):
        item_with_id = _with_id(normalized_kind, item)
        if item_with_id["id"] == id:
            item_with_id.update(patch)
            item_with_id["id"] = id
            item_with_id["updated_ts"] = time.time()
            rows[index] = item_with_id
            _write_jsonl(path, rows)
            return json.dumps(
                {
                    "memory_root": str(_memory_root()),
                    "runtime_mode": _runtime_mode(),
                    "read_only": _memory_read_only(),
                    "kind": normalized_kind,
                    "updated": item_with_id,
                },
                ensure_ascii=False,
                indent=2,
            )
    return json.dumps(
        {
            "error": f"memory id not found: {id}",
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "kind": normalized_kind,
        },
        ensure_ascii=False,
        indent=2,
    )


@register(
    "memory_delete",
    "Delete a clearly harmful, stale, empty, duplicate, or misleading memory record by id after reading it.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "default": "lessons", "enum": ["lessons", "episodes", "skills"]},
            "id": {"type": "string"},
        },
        "required": ["id"],
    },
)
def memory_delete(id: str, kind: str = "lessons") -> str:
    blocked = _write_blocked("memory_delete")
    if blocked is not None:
        return blocked
    path = _kind_path(kind, create=True)
    normalized_kind = path.stem
    rows = _read_jsonl(path)
    kept = []
    deleted: dict[str, Any] | None = None
    for item in rows:
        item_with_id = _with_id(normalized_kind, item)
        if item_with_id["id"] == id and deleted is None:
            deleted = item_with_id
            continue
        kept.append(item)
    if deleted is None:
        return json.dumps(
            {
                "error": f"memory id not found: {id}",
                "runtime_mode": _runtime_mode(),
                "read_only": _memory_read_only(),
                "kind": normalized_kind,
            },
            ensure_ascii=False,
            indent=2,
        )
    _write_jsonl(path, kept)
    return json.dumps(
        {
            "memory_root": str(_memory_root()),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "kind": normalized_kind,
            "deleted": deleted,
        },
        ensure_ascii=False,
        indent=2,
    )
