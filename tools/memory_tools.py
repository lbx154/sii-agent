"""Agent-facing memory retrieval tools."""
from __future__ import annotations

import json
import os
import time
import hashlib
import re
import ast
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from memory.store import MemoryStore, _2WIKI_SEED_LESSONS, _2WIKI_SEED_SKILLS, _token_list as _memory_token_list

from .registry import register
from .runtime_context import get_tool_context

_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_MAX_QUERIES = 8
_KIND_LABELS = {
    "lessons": "lesson",
    "episodes": "episode",
    "skills": "skill",
}
_ANSWER_KEY_RE = re.compile(r"(?:^|_)(?:answer|expected|gold|reference|final)(?:_|$)", re.IGNORECASE)
_CACHE_MAX = 512
_CACHE_LOCK = Lock()
_EXPANSION_CACHE: OrderedDict[str, tuple[list[str], dict[str, Any]]] = OrderedDict()
_RERANK_CACHE: OrderedDict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = OrderedDict()
_GENERAL_MEMORY_SCOPES = {"keep_general", "general", "target_general", "global"}
_TASK_SPECIFIC_SCOPES = {"keep_task_specific", "keep_simplevqa", "task_specific"}
_GENERAL_TASK_FAMILIES = {"general", "target_general", "web_research"}
_VISUAL_TASK_FAMILIES = {"simplevqa", "visual_qa", "vqa_search", "ocr"}
_TEXT_TASK_FAMILIES = {"browsecomp_special", "multihop_text", "comparison", "relation_chain", "2wiki"}
_GENERAL_CATEGORIES = {
    "global_verification",
    "query_reformulation",
    "evidence_triage",
    "format",
    "multilingual",
    "tool_recovery",
    "generic_reasoning",
    "browser_navigation",
}
_CROSS_MODAL_GENERAL_CATEGORIES = {
    "global_verification",
    "query_reformulation",
    "evidence_triage",
    "format",
    "multilingual",
    "tool_recovery",
    "generic_reasoning",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _memory_root() -> Path:
    return Path(os.getenv("SII_AGENT_MEMORY_ROOT", os.getenv("MEMORY_ROOT", "logs/memory")))


def _runtime_mode() -> str:
    mode = os.getenv("SII_AGENT_RUNTIME_MODE", "train").strip().lower()
    return mode if mode in {"train", "test"} else "train"


def _memory_read_only() -> bool:
    return _runtime_mode() == "test"


def _include_incorrect_episodes() -> bool:
    return _env_bool("SII_MEMORY_INCLUDE_INCORRECT_EPISODES", False)


def _show_episode_answers() -> bool:
    return _env_bool("SII_MEMORY_SHOW_EPISODE_ANSWERS", not _memory_read_only())


def _safe_checklist_enabled() -> bool:
    return _env_bool("SII_MEMORY_SAFE_CHECKLIST", True)


def _memory_return_ref_limit() -> int:
    return _clamp_int(os.getenv("SII_MEMORY_MAX_RETURN_REFS", "3"), 3, 0, 6)


def _memory_guidance_char_limit() -> int:
    return _clamp_int(os.getenv("SII_MEMORY_GUIDANCE_MAX_CHARS", "480"), 480, 240, 900)


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


def _redact_memory_item(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    normalized = str(kind or "").strip().lower()
    if normalized in {"episode", "episodes"} and not _show_episode_answers():
        redacted: dict[str, Any] = {}
        for key, value in item.items():
            key_text = str(key)
            if _ANSWER_KEY_RE.search(key_text):
                continue
            if key_text in {"steps", "trajectory", "tool_calls"}:
                continue
            redacted[key_text] = value
        redacted["episode_answer_redacted"] = True
        return redacted
    return dict(item)


def _clamp_int(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _short(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit]


def _hash_text(text: object, limit: int = 4096) -> str:
    value = str(text or "")
    if len(value) > limit:
        value = f"{value[:limit]}\n...len={len(value)}"
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    with _CACHE_LOCK:
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value


def _cache_set(cache: OrderedDict[str, Any], key: str, value: Any) -> None:
    with _CACHE_LOCK:
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _tokens(text: object) -> set[str]:
    return set(_memory_token_list(text))


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


def _heuristic_decision_text(query: str, prompt_context: dict[str, Any]) -> str:
    query_question = _original_question(query)
    if query_question:
        return query_question
    return _original_question(str(prompt_context.get("prompt_text") or ""))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_visual_query(text: str, image_attached: bool) -> bool:
    return image_attached or _contains_any(
        text,
        (
            "image", "photo", "picture", "visual", "screenshot", "poster", "cover", "logo", "diagram", "map",
            "图", "图片", "图中", "照片", "画面", "截图", "海报", "封面", "地图", "标志", "车牌",
        ),
    )


def _looks_ocr_query(text: str) -> bool:
    return _contains_any(
        text,
        (
            "ocr", "text in the image", "cover text", "book cover", "printed", "inscription", "caption",
            "label", "sign", "title", "abbreviation", "acronym",
            "文字", "文本", "字样", "写着", "英文缩写", "缩写", "书名", "标题", "封面",
            "这本书", "书籍", "牌匾", "标牌", "标语", "铭文", "碑文", "题字", "印章", "字幕",
        ),
    )


def _looks_comparison_query(text: str) -> bool:
    if _contains_any(
        text,
        (
            "came out first", "died first", "born later", "born earlier", "released earlier", "released later",
            "older", "younger", "earlier", "later", "same country", "same nationality", "different country",
            "both located", "which film has", "which came first",
            "更早", "较早", "更晚", "较晚", "最早", "最晚", "相同", "不同", "同一", "两者",
            "两个", "二者", "比较", "是否都", "都位于", "哪个更", "哪一个更",
        ),
    ):
        return True
    return bool(re.search(r"\bwhich\b.*\b(or|first|earlier|later|older|younger|same|different|both|more|less)\b", text))


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


def _merge_queries(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            clean = " ".join(str(item or "").split())[:240]
            key = clean.lower()
            if not clean or key in seen:
                continue
            merged.append(clean)
            seen.add(key)
            if len(merged) >= _MAX_QUERIES:
                return merged
    return merged


def _field_texts(kind: str, item: dict[str, Any]) -> list[tuple[str, int, str]]:
    if kind == "lessons":
        return [
            ("reusable_lesson", 1, _flatten_text(item.get("reusable_lesson"))),
            ("corrective_strategy", 1, _flatten_text(item.get("corrective_strategy"))),
            ("failure_mode", 1, _flatten_text(item.get("failure_mode"))),
            ("task_family", 1, _flatten_text(item.get("task_family"))),
            ("memory_scope", 1, _flatten_text(item.get("memory_scope"))),
            ("root_cause", 1, _flatten_text(item.get("root_cause"))),
            ("tags", 1, _flatten_text(item.get("tags"))),
            ("filter_reason", 1, _flatten_text(item.get("filter_reason"))),
            ("question", 1, _original_question(_flatten_text(item.get("question")))),
        ]
    if kind == "episodes":
        fields = [
            ("correct", 4, _flatten_text(item.get("correct"))),
            ("stop_reason", 4, _flatten_text(item.get("stop_reason"))),
            ("tool_call_counts", 3, _flatten_text(item.get("tool_call_counts"))),
            ("question", 2, _original_question(_flatten_text(item.get("question")))),
        ]
        if _show_episode_answers():
            fields.insert(0, ("answer", 6, _flatten_text(item.get("answer"))))
        return fields
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
                phrase_overlap = {token for token in overlap if "_" in token}
                query_score += weight * len(phrase_overlap)
                matched_terms.update(overlap)
                matched_fields.add(field)
                if len(q_tokens) > 1 and overlap == q_tokens:
                    query_score += 2 * weight
        if query_score > 0:
            matched_queries.append(query)
            score += query_score
    return score, matched_queries, sorted(matched_terms), sorted(matched_fields)


def _infer_memory_metadata(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    text = _flatten_text(item).lower()
    explicit_task_family = str(item.get("task_family") or "").strip().lower()
    memory_scope = str(item.get("memory_scope") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    tags = [str(tag).strip().lower() for tag in (item.get("tags") or []) if str(tag).strip()]
    task_family = explicit_task_family or "general"
    if explicit_task_family:
        task_family = explicit_task_family
    elif "browsecomp" in text or "benchmark_answered" in text or "benchmark-csv" in text:
        task_family = "browsecomp_special"
    elif "simplevqa" in text:
        task_family = "simplevqa"
    elif "2wikimultihopqa" in text or "provided context" in text or "2wiki" in text:
        task_family = "2wiki"
    elif "film" in text or "director" in text:
        task_family = "film_director"
    explicit_modality = str(item.get("modality") or "").strip().lower()
    modality = explicit_modality if explicit_modality in {"image", "text"} else (
        "image"
        if task_family in {"visual_qa", "vqa_search", "ocr"}
        or any(term in text for term in ("image", "photo", "visual", "screenshot", "picture", "ocr"))
        else "text"
    )
    confidence = 0.5
    if kind == "episodes":
        if item.get("correct") is True:
            confidence = 0.8
        elif item.get("correct") is False:
            confidence = 0.1
    elif kind in {"lessons", "skills"}:
        try:
            confidence = max(0.1, min(1.0, float(item.get("score"))))
        except (TypeError, ValueError):
            confidence = 0.6
    return {
        "task_family": task_family,
        "memory_scope": memory_scope,
        "category": category,
        "tags": tags[:12],
        "modality": modality,
        "confidence": round(confidence, 3),
        "has_explicit_metadata": any(
            key in item for key in ("task_family", "memory_scope", "category", "modality", "source_run", "confidence")
        ),
        "source_run": str(item.get("source_run") or item.get("run") or item.get("run_root") or ""),
    }


def _current_memory_profile(query: str, prompt_context: dict[str, Any]) -> dict[str, Any]:
    prompt_text = str(prompt_context.get("prompt_text") or "")
    decision_text = _heuristic_decision_text(query, prompt_context)
    decision_lower = decision_text.lower()
    full_lower = f"{query}\n{prompt_text}".lower()
    image_attached = bool(prompt_context.get("image_attached"))
    visual = _looks_visual_query(decision_lower, image_attached=image_attached) or _contains_any(
        full_lower,
        ("image path:", "image local path", "image source:", "image file/source"),
    )
    ocr = _looks_ocr_query(decision_lower)
    comparison = _looks_comparison_query(decision_lower)
    task = str(prompt_context.get("task") or "").strip().lower()
    if "simplevqa" in task:
        task_family = "simplevqa"
    elif "provided context" in full_lower or "2wiki" in task or "2wikimultihopqa" in full_lower:
        task_family = "2wiki"
    elif task in {"benchmark_csv", "special"}:
        task_family = "visual_qa" if visual else "browsecomp_special"
    elif visual:
        task_family = "ocr" if ocr else "visual_qa"
    elif comparison:
        task_family = "comparison"
    else:
        task_family = "web_research"
    return {
        "task": task,
        "task_family": task_family,
        "modality": "image" if visual else "text",
        "visual": visual,
        "ocr": ocr,
        "comparison": comparison,
    }


def _is_general_memory(result: dict[str, Any]) -> bool:
    metadata = result.get("metadata") or {}
    scope = str(metadata.get("memory_scope") or "").lower()
    task_family = str(metadata.get("task_family") or "").lower()
    category = str(metadata.get("category") or "").lower()
    tags = {str(tag).lower() for tag in (metadata.get("tags") or [])}
    if scope in _TASK_SPECIFIC_SCOPES:
        return False
    if scope in _GENERAL_MEMORY_SCOPES:
        return True
    if task_family in {"general", "target_general"}:
        return True
    return bool(category and category in _GENERAL_CATEGORIES) or bool(tags & _GENERAL_CATEGORIES)


def _general_memory_transfers(result: dict[str, Any], profile: dict[str, Any]) -> bool:
    metadata = result.get("metadata") or {}
    modality = str(metadata.get("modality") or "").lower()
    category = str(metadata.get("category") or "").lower()
    tags = {str(tag).lower() for tag in (metadata.get("tags") or [])}
    cross_modal_policy = bool(category and category in _CROSS_MODAL_GENERAL_CATEGORIES) or bool(
        tags & _CROSS_MODAL_GENERAL_CATEGORIES
    )
    task_family = str(metadata.get("task_family") or "").lower()
    if task_family in _VISUAL_TASK_FAMILIES and profile["modality"] != "image" and not cross_modal_policy:
        return False
    if task_family in _TEXT_TASK_FAMILIES and profile["modality"] != "text" and not cross_modal_policy:
        return False
    if not modality or modality == profile["modality"]:
        return True
    return cross_modal_policy


def _is_strong_specific_memory(result: dict[str, Any], profile: dict[str, Any]) -> bool:
    metadata = result.get("metadata") or {}
    task_family = str(metadata.get("task_family") or "").lower()
    modality = str(metadata.get("modality") or "").lower()
    score = float(result.get("score") or 0.0)
    matched_terms = list(result.get("matched_terms") or [])
    if modality and modality != profile["modality"]:
        return False
    if profile["modality"] == "image":
        if task_family not in _VISUAL_TASK_FAMILIES:
            return False
        return score >= 10.0 or len(matched_terms) >= 4
    if profile["task_family"] == "browsecomp_special":
        return task_family == "browsecomp_special" and (score >= 10.0 or len(matched_terms) >= 4)
    if profile["task_family"] == "2wiki":
        return task_family in {"2wiki", "multihop_text", "comparison", "relation_chain"} and (
            score >= 10.0 or len(matched_terms) >= 4
        )
    if profile["task_family"] in {"comparison", "web_research"}:
        return task_family in {"web_research", "multihop_text", "comparison", "relation_chain"} and (
            score >= 12.0 or len(matched_terms) >= 5
        )
    return task_family in _TEXT_TASK_FAMILIES and task_family == profile["task_family"]


def _layer_memory_results(
    results: list[dict[str, Any]],
    prompt_context: dict[str, Any],
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = _current_memory_profile(query, prompt_context)
    general: list[dict[str, Any]] = []
    specific: list[dict[str, Any]] = []
    rejected = 0
    for result in results:
        row = dict(result)
        if _is_general_memory(row) and _general_memory_transfers(row, profile):
            row["memory_layer"] = "general_policy"
            row["applicability_reason"] = "cross-domain abstract policy"
            general.append(row)
        elif _is_strong_specific_memory(row, profile):
            row["memory_layer"] = "task_specific"
            row["applicability_reason"] = "strong task/modality match"
            specific.append(row)
        else:
            rejected += 1
    max_refs = _memory_return_ref_limit()
    max_general = min(max_refs, _clamp_int(os.getenv("SII_MEMORY_MAX_GENERAL_REFS", "2"), 2, 0, max_refs))
    max_specific = max(0, max_refs - max_general)
    selected = general[:max_general] + specific[:max_specific]
    if len(selected) < max_refs:
        selected.extend(specific[max_specific : max_specific + (max_refs - len(selected))])
    return selected[:max_refs], {
        "enabled": True,
        "profile": profile,
        "input_candidates": len(results),
        "kept": len(selected[:max_refs]),
        "kept_general_policy": min(len(general), max_general),
        "kept_task_specific": max(0, min(len(specific), max_refs - min(len(general), max_general))),
        "rejected_as_not_applicable": rejected,
        "policy": (
            "general cross-domain memories are used only as abstract checklist policies; "
            "task-specific memories require a strong task/modality match"
        ),
    }


def _soft_metadata_adjustment(
    result: dict[str, Any],
    prompt_context: dict[str, Any],
    matched_terms: list[str],
) -> float:
    metadata = result.get("metadata") or {}
    prompt_text = str(prompt_context.get("prompt_text") or "").lower()
    image_attached = bool(prompt_context.get("image_attached"))
    task = str(prompt_context.get("task") or "").lower()
    score = 0.0
    if image_attached and metadata.get("modality") == "image":
        score += 4.0
    if image_attached and metadata.get("modality") == "text":
        score -= 1.0
    task_family = str(metadata.get("task_family") or "")
    if task in {"benchmark_csv", "special"}:
        if task_family in {"browsecomp_special", "simplevqa"}:
            score += 4.0
        elif task_family == "2wiki" and len(matched_terms) < 4:
            score -= 4.0
    if "provided context" not in prompt_text and task_family == "2wiki" and len(matched_terms) < 3:
        score -= 2.0
    try:
        confidence = float(metadata.get("confidence") or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    score += (confidence - 0.5) * 2.0
    return score


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
        if kind == "episodes" and item.get("correct") is False and not _include_incorrect_episodes():
            continue
        score, matched_queries, matched_terms, matched_fields = _score_record(kind, item, queries)
        if score <= 0:
            continue
        if kind == "episodes" and item.get("correct") is True:
            score += 4
        if source == "seed":
            score += 2
        redacted_item = _redact_memory_item(kind, item)
        results.append(
            {
                "type": _KIND_LABELS[kind],
                "kind": kind,
                "source": source,
                "source_file": "seed" if source == "seed" else str(_kind_path(kind).name),
                "index": index,
                "id": _record_id(kind, item),
                "score": round(score, 3),
                "lexical_score": round(score, 3),
                "metadata": _infer_memory_metadata(kind, item),
                "matched_queries": matched_queries,
                "matched_terms": matched_terms[:16],
                "matched_fields": matched_fields,
                "item": redacted_item,
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
        "lexical_score": result.get("lexical_score", result["score"]),
        "metadata": result.get("metadata", {}),
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
    metadata = result.get("metadata", {}) or {}
    compact_metadata = {
        key: metadata.get(key)
        for key in ("task_family", "memory_scope", "category", "modality", "confidence")
        if metadata.get(key) not in (None, "", [])
    }
    base = {
        "kind": result["kind"],
        "type": result["type"],
        "id": result["id"],
        "source": result["source"],
        "score": result["score"],
        "lexical_score": result.get("lexical_score", result["score"]),
        "metadata": compact_metadata,
        "memory_layer": result.get("memory_layer"),
        "applicability_reason": result.get("applicability_reason"),
        "matched_fields": result["matched_fields"][:4],
        "matched_terms": result["matched_terms"][:6],
    }
    if result["kind"] == "lessons":
        base["summary"] = {
            "failure_mode": _short(item.get("failure_mode"), 100),
            "strategy": _short(item.get("corrective_strategy"), 160),
            "lesson": _short(item.get("reusable_lesson"), 160),
        }
    elif result["kind"] == "skills":
        base["summary"] = {
            "title": _short(item.get("title"), 100),
            "steps": [_short(step, 220) for step in (item.get("steps") or [])[:2]],
            "verifier": [_short(step, 180) for step in (item.get("verifier") or [])[:2]],
            "avoid": [_short(step, 160) for step in (item.get("bad_patterns") or [])[:2]],
        }
    else:
        summary = {
            "correct": item.get("correct"),
            "tool_call_counts": item.get("tool_call_counts"),
            "stop_reason": item.get("stop_reason"),
        }
        if item.get("correct") is True and _show_episode_answers() and item.get("answer") is not None:
            summary["answer"] = _short(item.get("answer"), 120)
        base["summary"] = summary
    return base


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            try:
                parsed_literal = ast.literal_eval(raw)
                return parsed_literal if isinstance(parsed_literal, dict) else None
            except Exception:
                pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(raw[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw = raw[start:index + 1]
                break
    else:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def _response_text(response: Any) -> str:
    try:
        msg = response.choices[0].message
    except Exception:  # noqa: BLE001
        return ""
    content = str(getattr(msg, "content", "") or "")
    if content.strip():
        return content
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None and isinstance(msg, dict):
        reasoning = msg.get("reasoning_content")
    return str(reasoning or "")


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


def _current_prompt_context() -> dict[str, Any]:
    ctx = get_tool_context()
    content = ctx.get("user_content")
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []
    image_sources: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url") or "")
                    if url:
                        image_parts.append({"type": "image_url", "image_url": image_url})
                        image_sources.append("data_url" if url.startswith("data:image/") else url)
    elif content is not None:
        text_parts.append(str(content))
    question = str(ctx.get("question") or "")
    prompt_text = "\n\n".join(part for part in text_parts if part).strip() or question
    if not image_sources:
        for match in re.finditer(r"Image (?:local path/source|file/source|source):\s*(.+)", prompt_text, flags=re.IGNORECASE):
            image_sources.append(match.group(1).strip())
    return {
        "task": str(ctx.get("task") or ""),
        "prompt_text": _short(prompt_text, 2400),
        "image_attached": bool(image_parts),
        "image_sources": image_sources[:3],
        "image_parts": image_parts[:1],
    }


def _summarize_current_trace(max_chars: int = 5000) -> str:
    ctx = get_tool_context()
    trajectory = ctx.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return ""
    latest_evidence_summary = next(
        (
            event for event in reversed(trajectory)
            if isinstance(event, dict)
            and event.get("role") == "system"
            and event.get("name") == "evidence_summary"
        ),
        None,
    )
    latest_context_compact = next(
        (
            event for event in reversed(trajectory)
            if isinstance(event, dict)
            and event.get("role") == "system"
            and event.get("name") == "context_compact"
        ),
        None,
    )
    events = list(trajectory[-20:])
    if latest_evidence_summary is not None and all(event is not latest_evidence_summary for event in events):
        events.insert(0, latest_evidence_summary)
    if latest_context_compact is not None and all(event is not latest_context_compact for event in events):
        events.insert(0, latest_context_compact)
    parts: list[str] = []
    for i, event in enumerate(events, 1):
        if not isinstance(event, dict):
            continue
        role = str(event.get("role") or "")
        if role == "assistant":
            calls = []
            for tc in event.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                calls.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            content = _short(event.get("content"), 500)
            parts.append(f"#{i} assistant calls: {'; '.join(calls) or '(no tool)'} content: {content}")
        elif role == "tool":
            name = str(event.get("name") or "?")
            args = _short(event.get("args"), 280)
            content = _short(event.get("content"), 900)
            parts.append(f"#{i} tool[{name}] args={args} result={content}")
        elif role == "system":
            name = str(event.get("name") or "system")
            content_limit = 2200 if name == "context_compact" else 1800 if name == "evidence_summary" else 500
            content = _short(event.get("content"), content_limit)
            parts.append(f"#{i} system[{name}] {content}")
    return _short("\n".join(parts), max_chars)


def _trace_signature() -> tuple[Any, ...]:
    ctx = get_tool_context()
    trajectory = ctx.get("trajectory")
    if not isinstance(trajectory, list):
        return (0,)
    tool_events = [
        event for event in trajectory
        if isinstance(event, dict) and event.get("role") in {"tool", "system"}
    ][-5:]
    return (
        len(trajectory),
        tuple(
            (
                str(event.get("name") or event.get("role") or ""),
                _hash_text(event.get("content"), 1200),
            )
            for event in tool_events
        ),
    )


def _context_cache_key(query: str, queries: list[str]) -> str:
    prompt_context = _current_prompt_context()
    image_key = "|".join(prompt_context.get("image_sources") or [])
    if prompt_context.get("image_attached") and not image_key:
        image_key = "attached"
    payload = {
        "memory_root": str(_memory_root()),
        "task": prompt_context.get("task") or "",
        "query": _original_question(query),
        "prompt": _hash_text(prompt_context.get("prompt_text"), 8000),
        "image": _hash_text(image_key),
        "queries": queries,
        "trace": _trace_signature(),
        "show_episode_answers": _show_episode_answers(),
    }
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), 16000)


def _memory_query_expansion_enabled() -> bool:
    return _env_bool("SII_MEMORY_SEARCH_EXPAND_QUERIES", True)


def _heuristic_memory_queries(query: str, prompt_context: dict[str, Any], max_queries: int = _MAX_QUERIES) -> list[str]:
    prompt_text = str(prompt_context.get("prompt_text") or "")
    decision_text = _heuristic_decision_text(query, prompt_context)
    decision_lower = decision_text.lower()
    full_lower = f"{query}\n{prompt_text}".lower()
    queries: list[str] = []
    image_attached = bool(prompt_context.get("image_attached"))
    visual = _looks_visual_query(decision_lower, image_attached=image_attached) or _contains_any(
        full_lower,
        ("image path:", "image local path", "image source:", "image file/source"),
    )
    ocr = _looks_ocr_query(decision_lower)
    comparison = _looks_comparison_query(decision_lower)
    if visual:
        if ocr:
            queries.extend(
                [
                    "ocr text extraction exact text book cover visual evidence",
                    "image to text printed title cover text",
                ]
            )
        elif comparison:
            queries.extend(
                [
                    "visual comparison compare candidates distinguishing features",
                    "comparison answer format visual evidence",
                ]
            )
        else:
            queries.extend(
                [
                    "visual qa image search entity identification",
                    "visual evidence verification reverse image search",
                ]
            )
    if ocr:
        queries.append("ocr text extraction image answer format")
    if comparison:
        queries.append("comparison compare both candidates answer format")
    if any(term in decision_lower for term in ("father", "mother", "wife", "husband", "spouse", "child", "grandfather", "grandmother", "in-law")):
        queries.append("multi-hop relation chain verify each hop")
    if "provided context" in full_lower or "2wiki" in full_lower:
        queries.append("multi-hop text qa context evidence verification")
    if any(term in decision_lower for term in ("award", "publisher", "author", "designer", "director", "species", "chemical", "formula")):
        queries.append("exact entity name verification search strategy")
    queries.append("concise final answer requested format")
    return _merge_queries(queries)[:max_queries]


def _expand_memory_queries(query: str, queries: list[str], max_queries: int = _MAX_QUERIES) -> tuple[list[str], dict[str, Any]]:
    if not _memory_query_expansion_enabled():
        return queries, {"enabled": False}
    cache_key = _context_cache_key(query, queries)
    cached = _cache_get(_EXPANSION_CACHE, cache_key)
    if cached is not None:
        cached_queries, cached_meta = cached
        meta = dict(cached_meta)
        meta["cache"] = "hit"
        return list(cached_queries), meta
    prompt_context = _current_prompt_context()
    if not prompt_context["prompt_text"] and not query:
        return queries, {"enabled": True, "expanded": False, "reason": "empty_prompt"}
    try:
        from agent.llm import chat

        system = (
            "You generate retrieval queries for an agent memory store. "
            "The memory contains reusable lessons, skills, and prior episodes. "
            "Given the current prompt, optional image, and initial query phrases, produce short query phrases that help find transferable procedures. "
            "Think by analogy: include task pattern, answer type, visual clue type, relation chain, search strategy, and verification failure modes. "
            "Do NOT solve the current question, guess the answer, copy benchmark wording, or include final-answer candidates. "
            "Return strict JSON only: {\"queries\":[\"...\"]}."
        )
        payload = {
            "current_prompt": prompt_context["prompt_text"],
            "current_trace_so_far": _summarize_current_trace(4000),
            "image_context": {
                "attached_to_this_message": prompt_context["image_attached"],
                "sources": prompt_context["image_sources"],
            },
            "initial_queries": queries,
            "rules": [
                "2-8 short phrases, each <= 12 words",
                "prefer reusable procedures over entity-only phrases",
                "include visual/image-specific retrieval phrases when the image matters",
                "do not include candidate answers",
            ],
        }
        user_text = json.dumps(payload, ensure_ascii=False)
        user_content: str | list[dict[str, Any]]
        if prompt_context["image_parts"]:
            user_content = [{"type": "text", "text": user_text}, *prompt_context["image_parts"]]
        else:
            user_content = user_text
        response = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=600,
            timeout=float(os.getenv("SII_MEMORY_SEARCH_EXPAND_TIMEOUT", "45")),
        )
        parsed = _parse_json_object(_response_text(response))
        expanded_raw = parsed.get("queries") if isinstance(parsed, dict) else []
        expanded = [
            _short(item, 160)
            for item in (expanded_raw or [])
            if isinstance(item, str) and item.strip()
        ]
        merged = _merge_queries(queries, expanded[: max(0, max_queries - len(queries))])
        meta = {
            "enabled": True,
            "expanded": bool(expanded),
            "added": [item for item in merged if item not in queries],
            "image_attached": prompt_context["image_attached"],
            "cache": "miss",
        }
        _cache_set(_EXPANSION_CACHE, cache_key, (merged, meta))
        return merged, meta
    except Exception as exc:  # noqa: BLE001
        meta = {
            "enabled": True,
            "expanded": False,
            "error": f"{type(exc).__name__}: {_short(exc, 240)}",
            "cache": "miss",
        }
        _cache_set(_EXPANSION_CACHE, cache_key, (queries, meta))
        return queries, meta


def _memory_rerank_enabled() -> bool:
    return _env_bool("SII_MEMORY_SEARCH_RERANK", True)


def _apply_metadata_soft_rerank(results: list[dict[str, Any]], prompt_context: dict[str, Any]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for result in results:
        row = dict(result)
        metadata_delta = _soft_metadata_adjustment(row, prompt_context, list(row.get("matched_terms") or []))
        row["metadata_score_delta"] = round(metadata_delta, 3)
        row["score"] = round(float(row.get("score") or 0.0) + metadata_delta, 3)
        adjusted.append(row)
    adjusted.sort(key=lambda row: (-float(row["score"]), str(row["kind"]), str(row["id"])))
    return adjusted


def _rerank_memory_results(
    query: str,
    queries: list[str],
    results: list[dict[str, Any]],
    *,
    k: int,
    auto_prefetch: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not results:
        return results, {"enabled": False, "reason": "no_candidates"}
    if not _memory_rerank_enabled():
        return results, {"enabled": False}
    if auto_prefetch and not _env_bool("SII_MEMORY_RERANK_AUTO_PREFETCH", False):
        return results, {"enabled": True, "skipped": True, "reason": "auto_prefetch"}
    if len(results) >= 5:
        top = float(results[0].get("score") or 0)
        fifth = max(float(results[min(4, len(results) - 1)].get("score") or 0), 0.001)
        if top > 2.0 * fifth:
            return results, {"enabled": True, "skipped": True, "reason": "large_score_gap", "top": top, "fifth": fifth}
    candidate_limit = _clamp_int(os.getenv("SII_MEMORY_RERANK_CANDIDATES", "20"), 20, 5, 30)
    candidates = results[:candidate_limit]
    cache_key = _context_cache_key(query, queries) + "|" + _hash_text("|".join(str(item.get("id")) for item in candidates))
    cached = _cache_get(_RERANK_CACHE, cache_key)
    if cached is not None:
        reranked, meta = cached
        updated_meta = dict(meta)
        updated_meta["cache"] = "hit"
        return list(reranked), updated_meta
    try:
        from agent.llm import chat

        prompt_context = _current_prompt_context()
        refs = [_compact_memory_ref(result) for result in candidates]
        payload = {
            "current_prompt": prompt_context["prompt_text"],
            "current_trace_so_far": _summarize_current_trace(5000),
            "image_context": {
                "attached_to_this_message": prompt_context["image_attached"],
                "sources": prompt_context["image_sources"],
            },
            "queries_used": queries,
            "candidate_refs": refs,
            "instructions": [
                "Rank only candidates whose reusable method fits the current prompt/image/trace.",
                "Do not rank by old answer text; episode answers are intentionally redacted.",
                "Return IDs only, best first. Omit unrelated candidates.",
            ],
        }
        user_text = json.dumps(payload, ensure_ascii=False)
        user_content: str | list[dict[str, Any]]
        if prompt_context["image_parts"]:
            user_content = [{"type": "text", "text": user_text}, *prompt_context["image_parts"]]
        else:
            user_content = user_text
        response = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a relevance reranker for an agent memory store. "
                        "Select reusable lessons/skills/episodes that transfer to the current task. "
                        "Return strict JSON: {\"ranked_ids\":[\"...\"]}."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=300,
            timeout=float(os.getenv("SII_MEMORY_RERANK_TIMEOUT", "5")),
        )
        parsed = _parse_json_object(response.choices[0].message.content or "")
        ranked_ids = [str(item) for item in (parsed or {}).get("ranked_ids", []) if str(item).strip()]
        by_id = {str(result.get("id")): result for result in candidates}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item_id in ranked_ids:
            result = by_id.get(item_id)
            if result is None or item_id in seen:
                continue
            row = dict(result)
            row["rerank_selected"] = True
            selected.append(row)
            seen.add(item_id)
            if len(selected) >= k:
                break
        for result in results:
            item_id = str(result.get("id"))
            if item_id in seen:
                continue
            selected.append(result)
            if len(selected) >= len(results):
                break
        meta = {
            "enabled": True,
            "skipped": False,
            "candidate_count": len(candidates),
            "ranked_ids": ranked_ids[:k],
            "cache": "miss",
        }
        _cache_set(_RERANK_CACHE, cache_key, (selected, meta))
        return selected, meta
    except Exception as exc:  # noqa: BLE001
        meta = {
            "enabled": True,
            "skipped": True,
            "reason": "error",
            "error": f"{type(exc).__name__}: {_short(exc, 240)}",
            "cache": "miss",
        }
        _cache_set(_RERANK_CACHE, cache_key, (results, meta))
        return results, meta


def _memory_ref_text(ref: dict[str, Any]) -> str:
    summary = ref.get("summary") or {}
    metadata = ref.get("metadata") or {}
    return " ".join(
        str(part or "")
        for part in (
            summary.get("failure_mode"),
            summary.get("strategy"),
            summary.get("lesson"),
            summary.get("title"),
            " ".join(str(step) for step in (summary.get("steps") or [])),
            " ".join(str(step) for step in (summary.get("verifier") or [])),
            " ".join(str(step) for step in (summary.get("avoid") or [])),
            metadata.get("task_family"),
            metadata.get("memory_scope"),
            metadata.get("category"),
            " ".join(str(tag) for tag in (metadata.get("tags") or [])),
        )
    ).lower()


def _safe_checklist_items(refs: list[dict[str, Any]], profile: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    apply: list[str] = []
    avoid: list[str] = []
    current_modality = str((profile or {}).get("modality") or "")

    def add_apply(item: str) -> None:
        if item and item not in apply and len(apply) < 3:
            apply.append(item)

    def add_avoid(item: str) -> None:
        if item and item not in avoid and len(avoid) < 2:
            avoid.append(item)

    for ref in refs[:3]:
        text = _memory_ref_text(ref)
        if any(term in text for term in ("format", "granularity", "unit", "date", "answer span", "ticker", "abbreviation")):
            add_apply("Before final_answer, copy the exact requested span; preserve units, full names, date format, and abbreviations/tickers.")
        if current_modality == "image" and any(term in text for term in ("visual", "image", "ocr", "logo", "screenshot", "cover", "photo")):
            add_apply("For image questions, first extract visible text/visual identity, then verify the answer with current search evidence.")
        if any(term in text for term in ("query", "search", "evidence", "triage", "verification", "hallucinated", "wrong_tool")):
            add_apply("Use memory only as a checklist: verify the final candidate against current tool evidence before answering.")
            add_avoid("Do not let memory choose the answer or replace current evidence.")
        if any(term in text for term in ("constraint", "multi-hop", "relation", "comparison", "candidate", "self_contradictory")):
            add_apply("Check each required constraint separately and reject candidates that satisfy only part of the question.")
            add_avoid("Do not follow a candidate after a missing or contradicted constraint.")
        if any(term in text for term in ("browser", "page", "open", "tool_recovery")):
            add_apply("Open or inspect only promising sources; if a source does not support the target field, switch query instead of looping.")
    if not apply:
        add_apply("Use memory only as a generic verification checklist; solve from current evidence and keep the final answer concise.")
        add_avoid("Do not treat retrieved memory as evidence for the current answer.")
    return apply[:3], avoid[:2]


def _safe_checklist_guidance(
    refs: list[dict[str, Any]],
    max_chars: int,
    *,
    source: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    apply, avoid = _safe_checklist_items(refs, profile)
    guidance = "Memory checklist only, not evidence: " + " | ".join(apply)
    supporting_refs = [
        {
            "kind": str(ref.get("kind")),
            "id": str(ref.get("id")),
            "why": _short(str(ref.get("applicability_reason") or ref.get("memory_layer") or "applicable checklist"), 120),
        }
        for ref in refs[:3]
    ]
    return {
        "guidance": _short(guidance, max_chars),
        "apply": [_short(item, 180) for item in apply[:3]],
        "avoid": [_short(item, 160) for item in avoid[:2]],
        "supporting_refs": supporting_refs,
        "summary_source": source,
    }


def _compact_query_expansion_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("enabled", "expanded", "cache", "error", "image_attached"):
        if key in meta:
            compact[key] = meta[key]
    added = meta.get("added")
    if isinstance(added, list):
        compact["added_count"] = len(added)
        compact["added_preview"] = [_short(item, 120) for item in added[:3]]
    return compact


def _fallback_guidance(refs: list[dict[str, Any]], max_chars: int, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    if _safe_checklist_enabled():
        return _safe_checklist_guidance(refs, max_chars, source="safe_checklist_fallback", profile=profile)
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
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not refs:
        return {
            "guidance": "No applicable reusable memory was found; solve from current evidence and avoid inventing unsupported facts.",
            "apply": [],
            "avoid": [],
            "supporting_refs": [],
            "summary_source": "empty",
        }
    if _safe_checklist_enabled():
        return _safe_checklist_guidance(refs, max_chars, source="safe_checklist", profile=profile)
    if os.getenv("SII_MEMORY_SEARCH_LLM_SUMMARY", "1").strip().lower() in {"0", "false", "no"}:
        return _fallback_guidance(refs, max_chars, profile=profile)
    try:
        from agent.llm import chat

        system = (
            "You compress retrieved long-term memory for a general ReAct agent. "
            "You must transfer reusable methods from memory to the current problem, not copy old answers. "
            "Consider the full current prompt and any attached image together with the memory refs when deciding what transfers. "
            "Do not dump records. Produce only the most useful, task-relevant checklist. "
            "Memory is guidance, not evidence; tell the agent what generic checks to apply, not what final answer to give. "
            "Never infer current candidates, entities, source names, dates, teams, directions, or answers from the current prompt/image. "
            "For web-search tasks, convert memory into generic search discipline: verify each constraint, quote exact phrases, reject partial matches, and preserve requested answer format. "
            "If an attached image is present, give only generic visual/OCR verification advice; do not identify the image. "
            "Do not mention gold answers, gold standards, benchmark names, dataset names, exact examples, or record dumps. "
            "If records mention gold/benchmark-specific wording, generalize it to requested-answer formatting and current-task verification. "
            "If a record looks stale, harmful, duplicate, or too specific, mention that it should be inspected with memory_get before update/delete. "
            "Return strict JSON only."
        )
        prompt_context = _current_prompt_context()
        user = {
            "current_question": _short(_original_question(query), 800),
            "current_prompt": prompt_context["prompt_text"],
            "current_trace_so_far": _summarize_current_trace(6000),
            "image_context": {
                "attached_to_this_message": prompt_context["image_attached"],
                "sources": prompt_context["image_sources"],
            },
            "queries_used": queries,
            "max_guidance_chars": max_chars,
            "output_schema": {
                "guidance": f"single concise paragraph, <= {max_chars} chars",
                "apply": ["1-3 generic checklist items"],
                "avoid": ["0-2 pitfalls, especially treating memory as evidence"],
                "supporting_refs": [{"kind": "lessons|skills|episodes", "id": "record id", "why": "short reason"}],
            },
            "style_rules": [
                "general, not benchmark-specific",
                "never say gold answer/gold standard",
                "do not use examples or exact entity names",
                "do not name or imply current answer candidates",
                "do not reveal full record content",
                "transfer only reusable search/verification procedures that fit the current prompt and image",
                "if no reusable memory is specific enough, say to solve from current evidence and do not invent unsupported answers",
            ],
            "retrieved_memory_refs": refs[:10],
        }
        user_text = json.dumps(user, ensure_ascii=False)
        user_content: str | list[dict[str, Any]]
        if prompt_context["image_parts"]:
            user_content = [{"type": "text", "text": user_text}, *prompt_context["image_parts"]]
        else:
            user_content = user_text
        response = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
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
            "apply": _clean_str_list([_sanitize_guidance_text(item) for item in (parsed.get("apply") or [])], 3, 180),
            "avoid": _clean_str_list([_sanitize_guidance_text(item) for item in (parsed.get("avoid") or [])], 2, 160),
            "supporting_refs": supporting_refs or _fallback_guidance(refs, max_chars, profile=profile)["supporting_refs"],
            "summary_source": "llm",
        }
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_guidance(refs, max_chars, profile=profile)
        fallback["summary_error"] = f"{type(exc).__name__}: {_short(exc, 240)}"
        return fallback


@register(
    "memory_search",
    "Search concrete records from global memory. Provide the current question plus optional focused phrases; "
    "the tool expands the input into task-pattern queries and matches multiple lesson keys "
    "(lesson, strategy, failure mode, tags, task family, source question) before returning actionable guidance plus record ids.",
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
            "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20},
            "task": {"type": "string", "default": "general"},
            "include_lessons": {"type": "boolean", "default": True},
            "include_episodes": {"type": "boolean", "default": True},
            "include_skills": {"type": "boolean", "default": True},
            "include_seed": {"type": "boolean", "default": False},
            "auto_prefetch": {"type": "boolean", "default": False},
            "max_chars_per_item": {"type": "integer", "default": 900, "minimum": 120, "maximum": 3000},
            "guidance_max_chars": {"type": "integer", "default": 480, "minimum": 240, "maximum": 900},
        },
        "required": [],
    },
)
def memory_search(
    query: str = "",
    queries: list[str] | None = None,
    k: int = 3,
    task: str = "general",
    include_lessons: bool = True,
    include_episodes: bool = True,
    include_skills: bool = True,
    include_seed: bool = False,
    auto_prefetch: bool = False,
    max_chars_per_item: int = 900,
    guidance_max_chars: int = 480,
) -> str:
    k = _clamp_int(k, 3, 1, 20)
    max_chars_per_item = _clamp_int(max_chars_per_item, 900, 120, 3000)
    guidance_max_chars = min(_clamp_int(guidance_max_chars, 480, 240, 900), _memory_guidance_char_limit())
    root = _memory_root()
    if not str(query or "").strip() and queries:
        query = str(queries[0])
    initial_queries = _normalize_queries(query, queries)
    prompt_context = _current_prompt_context()
    heuristic_queries = _heuristic_memory_queries(query, prompt_context)
    initial_queries = _merge_queries(initial_queries, heuristic_queries)
    if not initial_queries:
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
    normalized_queries, query_expansion = _expand_memory_queries(query, initial_queries)
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
    results = _apply_metadata_soft_rerank(results, prompt_context)
    results, layering = _layer_memory_results(results, prompt_context, query)
    results, rerank = _rerank_memory_results(query, normalized_queries, results, k=max(_memory_return_ref_limit(), 1), auto_prefetch=bool(auto_prefetch))
    return_limit = min(k, _memory_return_ref_limit())
    compact_refs = [_compact_memory_ref(result) for result in results[:return_limit]]
    guidance = _summarize_memory_guidance(
        query,
        normalized_queries,
        compact_refs,
        max_chars=guidance_max_chars,
        profile=layering.get("profile") if isinstance(layering, dict) else None,
    )
    trace_summary = _summarize_current_trace()
    return json.dumps(
        {
            "memory_root": str(root),
            "runtime_mode": _runtime_mode(),
            "read_only": _memory_read_only(),
            "query": query,
            "initial_queries": initial_queries[:5],
            "heuristic_queries": heuristic_queries[:5],
            "queries_used": normalized_queries[:5],
            "queries_used_count": len(normalized_queries),
            "query_expansion": _compact_query_expansion_meta(query_expansion),
            "rerank": rerank,
            "layering": layering,
            "scanned_records": scanned,
            "guidance_context": {
                "used_current_prompt": bool(prompt_context["prompt_text"]),
                "used_current_trace": bool(trace_summary),
                "current_trace_chars": len(trace_summary),
                "image_attached": prompt_context["image_attached"],
                "image_sources": prompt_context["image_sources"],
            },
            "guidance": guidance,
            "record_refs": compact_refs,
            "full_records_suppressed": True,
            "suppressed_result_count": max(0, len(results) - len(compact_refs)),
            "usage_note": (
                "Use the guidance only as a short checklist, not as evidence or a source of answer candidates. "
                "record_refs are compact ids/summaries only; "
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
        original_with_id = _with_id(normalized_kind, item)
        item_with_id = _with_id(normalized_kind, _redact_memory_item(normalized_kind, original_with_id))
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
            redacted_item = _with_id(normalized_kind, _redact_memory_item(normalized_kind, item_with_id))
            return json.dumps(
                {
                    "memory_root": str(_memory_root()),
                    "runtime_mode": _runtime_mode(),
                    "read_only": _memory_read_only(),
                    "kind": normalized_kind,
                    "index": index,
                    "item": redacted_item,
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
