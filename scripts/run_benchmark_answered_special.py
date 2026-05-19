"""Run the answered browser benchmark with local scoring and submission traces.

The benchmark protocol here is intentionally test-only:
- no reflection/retry runner
- no train-time gold verification
- no memory writes
- baseline has no memory tools
- memory mode can query read-only memory without overall prompt injection
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.multimodal import image_url_from_source, multimodal_user_content  # noqa: E402
from agent.react import run_react  # noqa: E402
from agent.scoring import score_answer  # noqa: E402
from harness.controller import HarnessConfig  # noqa: E402
from tools.search import web_search  # noqa: E402


csv.field_size_limit(sys.maxsize)

BASELINE_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browsecomp_search",
    "browsecomp_open",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "final_answer",
)

MEMORY_QUERY_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browsecomp_search",
    "browsecomp_open",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "memory_search",
    "memory_stats",
    "memory_list",
    "memory_get",
    "final_answer",
)

VISUAL_TOOLS = {
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
}
BROWSER_TOOLS = {
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
}
BROWSECOMP_TOOLS = {
    "browsecomp_search",
    "browsecomp_open",
}
STOPWORDS = {
    "about",
    "a",
    "an",
    "according",
    "after",
    "also",
    "and",
    "another",
    "are",
    "as",
    "available",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "company",
    "could",
    "during",
    "each",
    "from",
    "for",
    "had",
    "has",
    "have",
    "identify",
    "into",
    "in",
    "its",
    "known",
    "many",
    "more",
    "most",
    "name",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "over",
    "person",
    "please",
    "question",
    "requested",
    "same",
    "show",
    "some",
    "specific",
    "that",
    "the",
    "their",
    "there",
    "this",
    "through",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "whose",
    "with",
    "would",
    "years",
    "fictional",
    "character",
    "occasionally",
    "audience",
    "involving",
}

BASELINE_EXTRA = (
    "Evaluation mode. No memory tools are available in this run. Use live search and available page tools as needed. "
    "Cross-check current evidence before final_answer. No reflection retry is available; solve carefully within "
    "this single ReAct attempt. Only use visual/image tools when the prompt provides an actual image file path "
    "or direct image URL; never pass an ordinary web page or search-result URL to image tools. "
    "For text-only BrowseComp-style rows, use browsecomp_search/browsecomp_open early because they query the local fixed corpus. "
    "Timed benchmark strategy: avoid broad search loops. After a few evidence calls, identify the strongest "
    "candidate, use at most one targeted verification if essential, then answer concisely."
)

MEMORY_EXTRA = (
    "Evaluation/test mode with read-only memory. You can and should call memory_search early when it may help: "
    "query the full question plus 2-6 focused phrases/entities/patterns. Treat memory_search guidance only as "
    "procedural advice, not evidence. Do not create/update/delete memory. Verify the final "
    "answer using current browsecomp_search/web_search or page evidence. For text-only BrowseComp-style rows, use browsecomp_search/browsecomp_open early because they query the local fixed corpus. "
    "No reflection retry is available; solve carefully within this single ReAct attempt. "
    "Only use visual/image tools when the prompt provides an actual image file path or direct image URL; never pass "
    "an ordinary web page or search-result URL to image tools. Timed benchmark strategy: avoid broad search loops. "
    "After memory plus a few evidence calls, identify the strongest candidate, use at most one targeted verification "
    "if essential, then answer concisely. For person names, preserve the full source-backed name exactly; do not drop "
    "the first given name or alter particles/casing."
)


def _set_default_env() -> None:
    os.environ["LLM_BACKEND"] = os.getenv("LLM_BACKEND") or "vllm"
    os.environ["VLLM_BASE_URL"] = os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8004/v1"
    if os.environ["VLLM_BASE_URL"].rstrip("/") == "http://127.0.0.1:8000/v1":
        os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8004/v1"
    os.environ["VLLM_MODEL"] = os.getenv("VLLM_MODEL") or "Qwen3.5-9B"
    if os.environ["VLLM_MODEL"] == "Qwen/Qwen3.5-9B":
        os.environ["VLLM_MODEL"] = "Qwen3.5-9B"
    os.environ["VLLM_API_KEY"] = os.getenv("VLLM_API_KEY") or "EMPTY"
    os.environ["VLLM_ENABLE_THINKING"] = os.getenv("VLLM_ENABLE_THINKING") or "0"
    os.environ["SII_AGENT_RUNTIME_MODE"] = os.getenv("SII_AGENT_RUNTIME_MODE") or "test"
    os.environ.setdefault("SII_AGENT_MEMORY_ROOT", "logs/memory")
    os.environ["SII_MEMORY_OVERALL_IN_PROMPT"] = "0"
    os.environ.setdefault("SII_MEMORY_OVERALL_PROMPT_MAX_CHARS", "2200")
    os.environ.setdefault("SII_MEMORY_SEARCH_LLM_SUMMARY", "1")
    os.environ.setdefault("SII_MEMORY_SEARCH_EXPAND_QUERIES", "1")
    os.environ.setdefault("SII_MEMORY_AUTO_PREFETCH", "0")
    os.environ.setdefault("SII_MEMORY_SEARCH_RERANK", "1")
    os.environ.setdefault("SII_MEMORY_RERANK_AUTO_PREFETCH", "0")
    os.environ.setdefault("SII_MEMORY_SHOW_EPISODE_ANSWERS", "0")
    os.environ.setdefault("SII_AGENT_EVIDENCE_SUMMARY", "1")
    os.environ.setdefault("SII_EVIDENCE_SUMMARY_WEB_EVERY", "3")
    os.environ.setdefault("SII_EVIDENCE_SUMMARY_TIMEOUT", "30")
    os.environ.setdefault("SII_EVIDENCE_SUMMARY_MIN_TIME_LEFT", "70")
    os.environ.setdefault("SII_EVIDENCE_STATE_MAX_CHARS", "2200")
    os.environ.setdefault("SII_AGENT_CONTEXT_COMPACT", "1")
    os.environ.setdefault("SII_CONTEXT_COMPACT_EVERY", "6")
    os.environ.setdefault("SII_CONTEXT_COMPACT_TIMEOUT", "30")
    os.environ.setdefault("SII_CONTEXT_COMPACT_MIN_TIME_LEFT", "80")
    os.environ.setdefault("SII_CONTEXT_COMPACT_MAX_CHARS", "3500")
    os.environ.setdefault("SII_CONTEXT_COMPACT_KEEP_RECENT_MESSAGES", "12")
    os.environ.setdefault("SEARCH_PROXY_MIN_K", "5")
    os.environ.setdefault("SEARCH_PROXY_FILTER_GARBAGE", "1")
    os.environ.setdefault("SEARCH_PROXY_FILTER_BENCHMARK_LEAKS", "0")
    os.environ.setdefault("SEARCH_PROXY_FILTER_EXTRA_K", "5")
    if "BROWSECOMP_INDEX_PATH" not in os.environ:
        bm25_index = ROOT / "indexes/bm25"
        sqlite_index = ROOT / "data/browsecomp-plus/browsecomp_fts.sqlite"
        if bm25_index.exists():
            os.environ["BROWSECOMP_INDEX_PATH"] = str(bm25_index)
        elif sqlite_index.exists():
            os.environ["BROWSECOMP_INDEX_PATH"] = str(sqlite_index)


def _cap_timeout_env(name: str, max_seconds: float) -> None:
    try:
        cap = max(1.0, float(max_seconds))
    except (TypeError, ValueError):
        return
    raw = os.getenv(name)
    if raw is None:
        os.environ[name] = str(int(cap) if cap.is_integer() else cap)
        return
    try:
        current = float(raw)
    except ValueError:
        os.environ[name] = str(int(cap) if cap.is_integer() else cap)
        return
    if current > cap:
        os.environ[name] = str(int(cap) if cap.is_integer() else cap)


def _load_rows(csv_path: Path, n: int, offset: int) -> tuple[list[str], list[dict[str, str]], list[tuple[int, dict[str, str]]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        all_rows = [dict(row) for row in reader]
    selected: list[tuple[int, dict[str, str]]] = []
    limit = len(all_rows) if n <= 0 else n
    for idx, row in enumerate(all_rows):
        if idx < offset:
            continue
        if len(selected) >= limit:
            break
        selected.append((idx, row))
    return fieldnames, all_rows, selected


def _build_question(row: dict[str, str], *, include_prefetch: bool = True) -> str:
    problem = " ".join(str(row.get("problem") or "").split())
    image = str(row.get("_image_ref") or "").strip()
    image_attached = str(row.get("_prompt_image_attached") or "").strip().lower() in {"1", "true", "yes"}
    image_attach_error = str(row.get("_prompt_image_error") or "").strip()
    parts = [problem]
    prefetch_context = str(row.get("_prefetch_context") or "").strip() if include_prefetch else ""
    if prefetch_context:
        parts.append(
            "Unverified automatic search snippets. They may contain useful answer candidates or noise. "
            "Use them as weak hints; verify when possible, and preserve exact wording, units, and date format "
            "when a snippet directly answers the question.\n\n"
            f"{prefetch_context}"
        )
    if image:
        if image_attached:
            parts.append(
                "The image itself is attached to this message for direct visual inspection.\n"
                f"Image local path/source for tools: {image}"
            )
        else:
            parts.append(f"Image file/source: {image}")
            if image_attach_error:
                parts.append(f"Direct model image attachment failed: {image_attach_error}")
        parts.append(
            "If the image is needed, call image_to_text, image_to_search_queries, visual_web_search, "
            "or reverse_image_search with this source path/URL. Do not copy raw image data into tool arguments."
        )
    parts.append(
        "Return only the concise answer span requested by the problem. For counts, return the number only; "
        "for names, return the full name only; do not include explanatory words or units unless they are part "
        "of the requested value."
    )
    return "\n\n".join(parts)


def _build_user_prompt(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    include_prefetch: bool = True,
) -> tuple[str | list[dict[str, Any]], dict[str, Any]]:
    image_ref = str(row.get("_image_ref") or "").strip()
    prompt_images = bool(getattr(args, "prompt_images", True))
    prompt_meta: dict[str, Any] = {"enabled": prompt_images, "attached": False}
    prompt_image, prompt_error = (None, None)
    image_meta = row.get("_image_meta")
    if image_ref and prompt_images:
        prompt_image, prompt_error = image_url_from_source(
            image_ref,
            image_meta if isinstance(image_meta, dict) else None,
        )
    row_for_question = dict(row)
    if prompt_image:
        row_for_question["_prompt_image_attached"] = "1"
        prompt_meta.update(
            {
                "attached": True,
                "source": "url" if prompt_image.startswith(("http://", "https://")) else "data_url",
            }
        )
    elif prompt_error:
        row_for_question["_prompt_image_error"] = prompt_error
        prompt_meta["error"] = prompt_error
    question = _build_question(row_for_question, include_prefetch=include_prefetch)
    if not prompt_image:
        return question, prompt_meta
    return multimodal_user_content(question, prompt_image), prompt_meta


def _words(text: str) -> list[str]:
    return [word.strip("'’.-") for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text) if word.strip("'’.-")]


def _informative_words(text: str) -> list[str]:
    items = []
    for word in _words(text):
        key = word.strip("'’.-").lower()
        if len(key) < 3 or key in STOPWORDS:
            continue
        items.append(word.strip("'’.-"))
    return items


def _keyword_query(problem: str, limit: int = 18) -> str:
    words = _informative_words(problem)
    scored = []
    for i, word in enumerate(words):
        key = word.lower()
        score = int(len(key) >= 7) + int(any(ch.isdigit() for ch in key)) + int(any(ch.isupper() for ch in word))
        scored.append((score, i, word))
    keep = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    keep_ordered = [word for _, _, word in sorted(keep, key=lambda item: item[1])]
    return " ".join(keep_ordered)[:260]


def _phrase_query(problem: str) -> str:
    words = _words(problem)
    candidates: list[tuple[int, int, str]] = []
    for n in range(2, 5):
        for i in range(0, max(0, len(words) - n + 1)):
            chunk = words[i : i + n]
            keys = [word.strip("'’.-").lower() for word in chunk]
            if any(key in STOPWORDS for key in keys[:1] + keys[-1:]):
                continue
            if not any(len(key) >= 7 or any(ch.isdigit() for ch in key) for key in keys):
                continue
            score = sum((len(key) if len(key) >= 7 else 0) + (4 if any(ch.isdigit() for ch in key) else 0) for key in keys)
            candidates.append((score, i, " ".join(chunk)))
    phrases = []
    seen = set()
    for _, _, phrase in sorted(candidates, key=lambda item: (-item[0], item[1])):
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= 2:
            break
    if not phrases:
        return _keyword_query(problem, limit=10)
    target = "answer"
    lower = problem.lower()
    if "character" in lower:
        target = "character"
    elif "full name" in lower or "who" in lower:
        target = "full name"
    elif "how many" in lower or "number" in lower:
        target = "number"
    return " ".join(f'"{phrase}"' for phrase in phrases) + f" {target}"


def _prefetch_search_context(problem: str, args: argparse.Namespace) -> tuple[str, list[dict[str, str]]]:
    if args.prefetch_searches <= 0:
        return "", []
    queries = []
    for query in (_phrase_query(problem), _keyword_query(problem)):
        query = " ".join(query.split())
        if query and query.lower() not in {item.lower() for item in queries}:
            queries.append(query)
        if len(queries) >= args.prefetch_searches:
            break
    records = []
    blocks = []
    for query in queries:
        old_fetch = os.environ.get("SEARCH_PROXY_FETCH")
        old_max_chars = os.environ.get("SEARCH_PROXY_MAX_CHARS")
        if args.prefetch_fetch_max_chars > 0:
            os.environ["SEARCH_PROXY_FETCH"] = "1"
            os.environ["SEARCH_PROXY_MAX_CHARS"] = str(args.prefetch_fetch_max_chars)
        try:
            result = web_search(query, k=args.prefetch_k)
        finally:
            if old_fetch is None:
                os.environ.pop("SEARCH_PROXY_FETCH", None)
            else:
                os.environ["SEARCH_PROXY_FETCH"] = old_fetch
            if old_max_chars is None:
                os.environ.pop("SEARCH_PROXY_MAX_CHARS", None)
            else:
                os.environ["SEARCH_PROXY_MAX_CHARS"] = old_max_chars
        records.append({"query": query, "content": result})
        if _is_usable_prefetch_result(result):
            blocks.append(f"[prefetch query: {query}]\n{result[: args.prefetch_max_chars]}")
    return "\n\n".join(blocks), records


def _is_usable_prefetch_result(result: str) -> bool:
    text = str(result or "").strip()
    if not text:
        return False
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("## "):
        text = "\n".join(lines[1:]).strip()
    lowered = text.lower()
    return not (
        lowered.startswith("error:")
        or lowered.startswith("(no results")
        or lowered.startswith("(no usable results")
    )


def _should_prefetch_search(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.prefetch_searches <= 0:
        return False
    return bool(row.get("_image_ref")) or bool(getattr(args, "prefetch_text", False))


def _postprocess_answer(question: str, answer: str | None) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"^(?:the answer is|answer:)\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(" .;:")
    question_l = str(question or "").lower()
    if re.search(r"\b(how many|number of|count of)\b", question_l):
        match = re.match(r"^([+-]?\d[\d,]*(?:\.\d+)?%?)\s+[A-Za-z][A-Za-z -]*(?:\s+available|\s+remaining)?$", text)
        if match:
            text = match.group(1)
        if re.fullmatch(r"\d{5,}", text):
            return f"{int(text):,}"
    return text


def _guess_image_ext(mime: str | None, data: bytes) -> str:
    if mime:
        guessed = mimetypes.guess_extension(mime.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _decode_image_payload(image: str) -> tuple[bytes, str | None, str]:
    value = "".join(image.strip().split())
    if value.startswith("data:image/"):
        header, sep, encoded = value.partition(",")
        if not sep or not encoded:
            raise ValueError("data image URL is missing a base64 payload")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        return base64.b64decode(encoded, validate=True), mime, "data_url"
    return base64.b64decode(value, validate=True), None, "raw_base64"


def _prepare_image(row: dict[str, str], idx: int, images_dir: Path) -> dict[str, Any]:
    image = str(row.get("image") or "").strip()
    if not image:
        row["_image_ref"] = ""
        row["_image_meta"] = {"kind": "none", "original_chars": 0}
        return row["_image_meta"]

    if image.startswith(("http://", "https://")):
        row["_image_ref"] = image
        row["_image_meta"] = {"kind": "url", "original_chars": len(image), "source": image}
        return row["_image_meta"]

    if len(image) < 4096:
        try:
            image_path = Path(image).expanduser()
            if image_path.exists() and image_path.is_file():
                row["_image_ref"] = str(image_path)
                row["_image_meta"] = {"kind": "path", "original_chars": len(image), "source": str(image_path)}
                return row["_image_meta"]
        except OSError:
            pass

    try:
        data, mime, kind = _decode_image_payload(image)
    except (binascii.Error, ValueError) as exc:
        row["_image_ref"] = ""
        row["_image_meta"] = {
            "kind": "invalid_image_payload",
            "original_chars": len(image),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return row["_image_meta"]

    digest = hashlib.sha1(data).hexdigest()[:12]
    ext = _guess_image_ext(mime, data)
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"benchmark-csv-{idx}-{digest}{ext}"
    if not path.exists():
        path.write_bytes(data)
    row["_image_ref"] = str(path)
    row["_image_meta"] = {
        "kind": kind,
        "original_chars": len(image),
        "bytes": len(data),
        "mime": mime,
        "path": str(path),
    }
    return row["_image_meta"]


def _run_one(
    mode: str,
    tools: tuple[str, ...],
    extra_system: str,
    idx: int,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_tools = tools if row.get("_image_ref") else tuple(name for name in tools if name not in VISUAL_TOOLS)
    if row.get("_image_ref"):
        row_tools = tuple(name for name in row_tools if name not in BROWSECOMP_TOOLS)
    if not args.enable_browser_tools:
        row_tools = tuple(name for name in row_tools if name not in BROWSER_TOOLS)
    row_extra_system = extra_system
    if not row.get("_image_ref"):
        row_extra_system += (
            " This row has no image input; use text/web tools for web pages, not visual/image tools. "
            "Start with browsecomp_search for distinctive phrases/constraints, then use browsecomp_open on promising docids before falling back to live web_search."
        )
    if not args.enable_browser_tools:
        row_extra_system += " Browser tools are disabled for this timed run; rely on web_search/wiki/visual snippets and answer from the best-supported evidence."
    cfg = HarnessConfig(
        max_steps=args.max_steps,
        max_wall_seconds=args.max_wall_seconds,
        max_llm_tokens=args.max_llm_tokens,
        max_llm_call_seconds=args.max_llm_call_seconds,
        min_llm_call_seconds=args.min_llm_call_seconds,
        allowed_tools=row_tools,
        max_parallel_tool_calls=args.max_parallel_tool_calls,
        max_web_search_calls=args.max_web_search_calls,
        max_research_tool_calls=args.max_research_tool_calls,
        synthesize_after_tool_calls=args.synthesize_after_tool_calls,
    )
    expected = str(row.get("answer") or "").strip()
    started = time.time()
    try:
        if _should_prefetch_search(row, args):
            prefetch_context, prefetch_records = _prefetch_search_context(str(row.get("problem") or ""), args)
        else:
            prefetch_context, prefetch_records = "", []
        row_for_prompt = dict(row)
        row_for_prompt["_prefetch_context"] = prefetch_context
        user_content, prompt_image_meta = _build_user_prompt(row_for_prompt, args, include_prefetch=True)
        clean_question = _build_question(row, include_prefetch=False)
        result = run_react(
            clean_question,
            cfg=cfg,
            extra_system=row_extra_system,
            expected=None,
            task="benchmark_csv",
            user_content=user_content,
        )
        final_answer = _postprocess_answer(row.get("problem", ""), result.final_answer)
        scores = score_answer(final_answer, expected)
        prefetch_trajectory = [
            {
                "role": "tool",
                "name": "prefetch_web_search",
                "args": {"query": item["query"]},
                "content": item["content"],
            }
            for item in prefetch_records
        ]
        tool_call_counts = dict(result.tool_call_counts)
        if prefetch_records:
            tool_call_counts["prefetch_web_search"] = len(prefetch_records)
        return {
            "id": f"benchmark-csv-{idx}",
            "index": idx,
            "mode": mode,
            "problem": row.get("problem", ""),
            "image": row.get("_image_ref", ""),
            "image_meta": row.get("_image_meta", {}),
            "prompt_image": prompt_image_meta,
            "answer": final_answer,
            "expected": expected,
            "correct": bool(scores.get("correct")),
            "exact": bool(scores.get("exact")),
            "f1": float(scores.get("f1") or 0.0),
            "rationale": result.rationale,
            "steps": result.steps,
            "tool_calls": result.tool_calls + len(prefetch_records),
            "tool_call_counts": tool_call_counts,
            "stop_reason": result.stop_reason,
            "finish_reasons": result.finish_reasons,
            "evidence_state": result.evidence_state,
            "compact_context": result.compact_context,
            "elapsed": time.time() - started,
            "trajectory": (prefetch_trajectory + result.trajectory) if args.save_trace else [],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        scores = score_answer(None, expected)
        return {
            "id": f"benchmark-csv-{idx}",
            "index": idx,
            "mode": mode,
            "problem": row.get("problem", ""),
            "image": row.get("_image_ref", ""),
            "image_meta": row.get("_image_meta", {}),
            "prompt_image": {
                "enabled": bool(getattr(args, "prompt_images", True)),
                "attached": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            "answer": "",
            "expected": expected,
            "correct": bool(scores.get("correct")),
            "exact": bool(scores.get("exact")),
            "f1": float(scores.get("f1") or 0.0),
            "rationale": "",
            "steps": 0,
            "tool_calls": 0,
            "tool_call_counts": {},
            "stop_reason": f"error: {type(exc).__name__}: {exc}",
            "finish_reasons": {},
            "evidence_state": {},
            "compact_context": {},
            "elapsed": time.time() - started,
            "trajectory": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _submission_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["index"],
        "problem": record["problem"],
        "image": record["image"],
        "image_meta": record.get("image_meta", {}),
        "answer": record["answer"],
        "steps": record["steps"],
        "tool_call_counts": record["tool_call_counts"],
        "stop_reason": record["stop_reason"],
        "elapsed": record["elapsed"],
        "evidence_state": record.get("evidence_state", {}),
        "compact_context": record.get("compact_context", {}),
        "trajectory": record["trajectory"],
    }


def _write_mode_outputs(
    run_root: Path,
    mode: str,
    records: list[dict[str, Any]],
    fieldnames: list[str],
    input_rows: list[dict[str, str]],
) -> dict[str, str]:
    mode_root = run_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    group_name = f"group_{mode}"
    trace_path = mode_root / f"{group_name}.json"
    answer_path = mode_root / f"{group_name}.csv"
    zip_path = mode_root / f"{group_name}.zip"

    ordered = sorted(records, key=lambda item: int(item["index"]))
    trace_payload = [_submission_trace_record(record) for record in ordered]
    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    answer_by_index = {int(record["index"]): str(record.get("answer") or "") for record in ordered}
    output_rows: list[dict[str, str]] = []
    for idx, row in enumerate(input_rows):
        output = {field: row.get(field, "") for field in fieldnames}
        output["answer"] = answer_by_index.get(idx, "")
        output_rows.append(output)
    with answer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(trace_path, arcname=trace_path.name)
        zf.write(answer_path, arcname=answer_path.name)
    return {"trace": str(trace_path), "answers_csv": str(answer_path), "zip": str(zip_path)}


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "accuracy": sum(1 for record in records if record["correct"]) / len(records),
        "exact_match": sum(1 for record in records if record["exact"]) / len(records),
        "avg_f1": sum(float(record["f1"]) for record in records) / len(records),
        "avg_steps": sum(int(record["steps"]) for record in records) / len(records),
        "avg_tool_calls": sum(int(record["tool_calls"]) for record in records) / len(records),
        "stop_reasons": dict(Counter(str(record["stop_reason"]) for record in records)),
        "tool_call_counts": dict(sum((Counter(record["tool_call_counts"]) for record in records), Counter())),
        "errors": sum(1 for record in records if record.get("error")),
    }


def _mode_specs(requested: str) -> list[tuple[str, tuple[str, ...], str]]:
    specs = [
        ("baseline_no_memory", BASELINE_TOOLS, BASELINE_EXTRA),
        ("memory_query_only", MEMORY_QUERY_TOOLS, MEMORY_EXTRA),
    ]
    if requested == "both":
        return specs
    return [spec for spec in specs if spec[0] == requested]


def _run_mode(
    run_root: Path,
    mode: str,
    tools: tuple[str, ...],
    extra_system: str,
    selected_rows: list[tuple[int, dict[str, str]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jsonl_path = run_root / f"{mode}.jsonl"
    existing_records: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("index"), int):
                existing_records.append(record)
                completed_indices.add(int(record["index"]))
    remaining_rows = [(idx, row) for idx, row in selected_rows if idx not in completed_indices]
    print(
        f"START_MODE {mode} rows={len(selected_rows)} completed={len(completed_indices)} "
        f"remaining={len(remaining_rows)} concurrency={args.concurrency}",
        flush=True,
    )
    records: list[dict[str, Any]] = list(existing_records)
    lock = threading.Lock()
    file_mode = "a" if args.resume and jsonl_path.exists() else "w"
    with jsonl_path.open(file_mode, encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_run_one, mode, tools, extra_system, idx, row, args): idx
            for idx, row in remaining_rows
        }
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            records.append(record)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "done": done,
                        "index": record["index"],
                        "predicted": record["answer"],
                        "expected": record["expected"],
                        "correct": record["correct"],
                        "f1": record["f1"],
                        "steps": record["steps"],
                        "stop_reason": record["stop_reason"],
                        "tools": record["tool_call_counts"],
                        "elapsed": record["elapsed"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return sorted(records, key=lambda item: int(item["index"]))


def _write_summary(
    run_root: Path,
    config: dict[str, Any],
    mode_records: dict[str, list[dict[str, Any]]],
    outputs: dict[str, dict[str, str]],
    selected_rows: list[tuple[int, dict[str, str]]],
) -> dict[str, Any]:
    mode_summaries = {mode: _summarize(records) for mode, records in mode_records.items()}
    baseline = {int(record["index"]): record for record in mode_records.get("baseline_no_memory", [])}
    memory = {int(record["index"]): record for record in mode_records.get("memory_query_only", [])}
    comparison = []
    for idx, row in selected_rows:
        b_record = baseline.get(idx)
        m_record = memory.get(idx)
        comparison.append(
            {
                "index": idx,
                "expected": str(row.get("answer") or ""),
                "baseline_predicted": b_record.get("answer") if b_record else None,
                "baseline_correct": b_record.get("correct") if b_record else None,
                "baseline_f1": b_record.get("f1") if b_record else None,
                "memory_predicted": m_record.get("answer") if m_record else None,
                "memory_correct": m_record.get("correct") if m_record else None,
                "memory_f1": m_record.get("f1") if m_record else None,
            }
        )
    summary = {
        "run_root": str(run_root),
        "config": config,
        "mode_summaries": mode_summaries,
        "delta_memory_minus_baseline": {
            "accuracy": mode_summaries.get("memory_query_only", {}).get("accuracy", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("accuracy", 0.0),
            "exact_match": mode_summaries.get("memory_query_only", {}).get("exact_match", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("exact_match", 0.0),
            "avg_f1": mode_summaries.get("memory_query_only", {}).get("avg_f1", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("avg_f1", 0.0),
        },
        "outputs": outputs,
        "comparison": comparison,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/root/harness-sii-browser-service/benchmark_answered.csv")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--mode", choices=["both", "baseline_no_memory", "memory_query_only"], default="both")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to run. Use 0 for all rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=26)
    parser.add_argument("--max-llm-tokens", type=int, default=120000)
    parser.add_argument("--max-wall-seconds", type=float, default=180.0)
    parser.add_argument("--max-llm-call-seconds", type=float, default=600.0)
    parser.add_argument("--min-llm-call-seconds", type=float, default=30.0)
    parser.add_argument("--max-parallel-tool-calls", type=int, default=4)
    parser.add_argument("--max-web-search-calls", type=int, default=0)
    parser.add_argument("--max-research-tool-calls", type=int, default=12)
    parser.add_argument("--synthesize-after-tool-calls", type=int, default=6)
    parser.add_argument("--enable-browser-tools", action="store_true", help="Allow browser_open/browser interaction tools.")
    parser.add_argument("--prefetch-searches", type=int, default=2)
    parser.add_argument("--prefetch-text", action="store_true", help="Also inject automatic web-search prefetch for non-image text rows.")
    parser.add_argument("--prefetch-k", type=int, default=5)
    parser.add_argument("--prefetch-max-chars", type=int, default=12000)
    parser.add_argument("--prefetch-fetch-max-chars", type=int, default=0)
    parser.add_argument(
        "--no-prompt-images",
        dest="prompt_images",
        action="store_false",
        help="Do not attach image inputs directly to the model prompt; keep only image tool references.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in mode JSONL files.")
    parser.add_argument("--no-save-trace", dest="save_trace", action="store_false")
    parser.set_defaults(save_trace=True, prompt_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_default_env()
    _cap_timeout_env("SEARCH_PROXY_TIMEOUT", args.max_wall_seconds)
    _cap_timeout_env("SEARCH_PROXY_UPLOAD_TIMEOUT", args.max_wall_seconds)
    _cap_timeout_env("SII_MEMORY_SEARCH_SUMMARY_TIMEOUT", args.max_wall_seconds)
    csv_path = Path(args.csv)
    run_name = args.run_name or f"benchmark_answered_special_26s_120k_c{args.concurrency}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.out) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames, input_rows, selected_rows = _load_rows(csv_path, args.n, args.offset)
    images_dir = run_root.resolve() / "images"
    image_metas = []
    for idx, row in selected_rows:
        image_metas.append({"index": idx, **_prepare_image(row, idx, images_dir)})
    config = {
        "run_root": str(run_root),
        "csv_path": str(csv_path),
        "n": len(selected_rows),
        "offset": args.offset,
        "concurrency": args.concurrency,
        "backend": os.getenv("LLM_BACKEND"),
        "base_url": os.getenv("VLLM_BASE_URL"),
        "model": os.getenv("VLLM_MODEL"),
        "enable_thinking": os.getenv("VLLM_ENABLE_THINKING"),
        "runtime_mode": os.getenv("SII_AGENT_RUNTIME_MODE"),
        "memory_root": os.getenv("SII_AGENT_MEMORY_ROOT"),
        "memory_read_only": True,
        "reflection_retry": False,
        "max_steps": args.max_steps,
        "max_llm_tokens": args.max_llm_tokens,
        "max_wall_seconds": args.max_wall_seconds,
        "max_llm_call_seconds": args.max_llm_call_seconds,
        "min_llm_call_seconds": args.min_llm_call_seconds,
        "max_parallel_tool_calls": args.max_parallel_tool_calls,
        "max_web_search_calls": args.max_web_search_calls,
        "max_research_tool_calls": args.max_research_tool_calls,
        "synthesize_after_tool_calls": args.synthesize_after_tool_calls,
        "browser_tools_enabled": args.enable_browser_tools,
        "prompt_images": args.prompt_images,
        "prefetch_searches": args.prefetch_searches,
        "prefetch_text": args.prefetch_text,
        "prefetch_k": args.prefetch_k,
        "prefetch_max_chars": args.prefetch_max_chars,
        "prefetch_fetch_max_chars": args.prefetch_fetch_max_chars,
        "search_proxy_timeout": os.getenv("SEARCH_PROXY_TIMEOUT"),
        "search_proxy_min_k": os.getenv("SEARCH_PROXY_MIN_K"),
        "search_proxy_filter_garbage": os.getenv("SEARCH_PROXY_FILTER_GARBAGE"),
        "search_proxy_filter_benchmark_leaks": os.getenv("SEARCH_PROXY_FILTER_BENCHMARK_LEAKS"),
        "search_proxy_filter_extra_k": os.getenv("SEARCH_PROXY_FILTER_EXTRA_K"),
        "browsecomp_index_path": os.getenv("BROWSECOMP_INDEX_PATH"),
        "memory_search_summary_timeout": os.getenv("SII_MEMORY_SEARCH_SUMMARY_TIMEOUT"),
        "memory_search_expand_queries": os.getenv("SII_MEMORY_SEARCH_EXPAND_QUERIES"),
        "memory_search_expand_timeout": os.getenv("SII_MEMORY_SEARCH_EXPAND_TIMEOUT"),
        "memory_auto_prefetch": os.getenv("SII_MEMORY_AUTO_PREFETCH"),
        "memory_search_rerank": os.getenv("SII_MEMORY_SEARCH_RERANK"),
        "memory_rerank_auto_prefetch": os.getenv("SII_MEMORY_RERANK_AUTO_PREFETCH"),
        "memory_show_episode_answers": os.getenv("SII_MEMORY_SHOW_EPISODE_ANSWERS"),
        "evidence_summary": os.getenv("SII_AGENT_EVIDENCE_SUMMARY"),
        "evidence_summary_web_every": os.getenv("SII_EVIDENCE_SUMMARY_WEB_EVERY"),
        "evidence_summary_timeout": os.getenv("SII_EVIDENCE_SUMMARY_TIMEOUT"),
        "evidence_summary_min_time_left": os.getenv("SII_EVIDENCE_SUMMARY_MIN_TIME_LEFT"),
        "context_compact": os.getenv("SII_AGENT_CONTEXT_COMPACT"),
        "context_compact_every": os.getenv("SII_CONTEXT_COMPACT_EVERY"),
        "context_compact_timeout": os.getenv("SII_CONTEXT_COMPACT_TIMEOUT"),
        "context_compact_min_time_left": os.getenv("SII_CONTEXT_COMPACT_MIN_TIME_LEFT"),
        "context_compact_max_chars": os.getenv("SII_CONTEXT_COMPACT_MAX_CHARS"),
        "output_format_reference": str(ROOT / "benchmarkreadme.md"),
        "submission_format": "group_{mode}.json trace, group_{mode}.csv answers, group_{mode}.zip bundle",
        "overall_in_prompt": False,
        "overall_preview": "",
        "save_trace": args.save_trace,
        "image_handling": {
            "images_dir": str(images_dir),
            "nonempty": sum(1 for meta in image_metas if meta.get("kind") != "none"),
            "decoded": sum(1 for meta in image_metas if meta.get("kind") in {"raw_base64", "data_url"}),
            "invalid": sum(1 for meta in image_metas if meta.get("kind") == "invalid_image_payload"),
            "max_original_chars": max((int(meta.get("original_chars") or 0) for meta in image_metas), default=0),
        },
    }
    (run_root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    mode_records: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, dict[str, str]] = {}
    for mode, tools, extra_system in _mode_specs(args.mode):
        records = _run_mode(run_root, mode, tools, extra_system, selected_rows, args)
        mode_records[mode] = records
        outputs[mode] = _write_mode_outputs(run_root, mode, records, fieldnames, input_rows)
        print(json.dumps({"mode_summary": mode, **_summarize(records), "outputs": outputs[mode]}, ensure_ascii=False), flush=True)

    summary = _write_summary(run_root, config, mode_records, outputs, selected_rows)
    print("FINAL_SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
