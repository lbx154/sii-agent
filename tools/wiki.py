"""Offline Wikipedia search over a local wiki25 JSONL shard/index."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .registry import register

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _default_index_path() -> Path:
    configured = os.getenv("WIKI25_INDEX_PATH")
    if configured:
        return Path(configured)
    sqlite_path = Path("data/wiki25/wiki25_fts.sqlite")
    if sqlite_path.exists():
        return sqlite_path
    return Path("data/wiki25/wiki25_sample.jsonl")


def _is_sqlite_index(index_path: str) -> bool:
    return Path(index_path).suffix in {".db", ".sqlite", ".sqlite3"}


def _sqlite_timeout() -> float:
    try:
        return float(os.getenv("WIKI_SQLITE_TIMEOUT", os.getenv("SQLITE_TOOL_TIMEOUT", "30")))
    except ValueError:
        return 30.0


@lru_cache(maxsize=1)
def _load_index(index_path: str) -> tuple[object, list[dict]]:
    from rank_bm25 import BM25Okapi

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python -m scripts.build_wiki_index"
        )

    docs: list[dict] = []
    tokenized: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            contents = str(item.get("contents", ""))
            title, text = _split_title(contents)
            doc = {"id": item.get("id"), "title": title, "text": text}
            docs.append(doc)
            tokenized.append(_tokenize(f"{title} {text}"))

    if not docs:
        raise ValueError(f"{path} contains no documents")
    return BM25Okapi(tokenized), docs


def _fts_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokenize(query):
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    if not terms:
        terms = _tokenize(query)
    return terms[:10]


def _quoted(term: str) -> str:
    return f'"{term}"'


def _fts_queries(query: str) -> list[str]:
    terms = _fts_terms(query)
    if not terms:
        return ['""']

    queries: list[str] = []
    max_and_terms = min(6, len(terms))
    for n in range(max_and_terms, 1, -1):
        queries.append(" AND ".join(_quoted(term) for term in terms[:n]))
    queries.append(" OR ".join(_quoted(term) for term in terms))
    return queries


def _search_sqlite(index_path: str, query: str, k: int) -> str:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python -m scripts.build_wiki_fts"
        )

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_sqlite_timeout())
    try:
        rows = []
        for match_query in _fts_queries(query):
            rows = conn.execute(
                """
                SELECT id, title, text, bm25(wiki_fts, 2.0, 1.0) AS score
                FROM wiki_fts
                WHERE wiki_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_query, k),
            ).fetchall()
            if rows:
                break
    finally:
        conn.close()

    out: list[str] = []
    for rank, (doc_id, title, text, score) in enumerate(rows, 1):
        snippet = str(text).replace("\n", " ")
        out.append(
            f"[{rank}] {title} (wiki25 id={doc_id}, score={-score:.2f})\n"
            f"    {snippet}"
        )
    return "\n".join(out) if out else "(no results)"


def _get_page_sqlite(index_path: str, title_or_id: str, max_chars: int) -> str:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python -m scripts.build_wiki_fts"
        )
    needle = str(title_or_id).strip()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=_sqlite_timeout())
    try:
        rows = conn.execute(
            """
            SELECT id, title, text
            FROM wiki_fts
            WHERE id = ? OR lower(title) = lower(?)
            LIMIT 3
            """,
            (needle, needle),
        ).fetchall()
        if not rows:
            candidates = conn.execute(
                """
                SELECT id, title, text, bm25(wiki_fts, 2.0, 1.0) AS score
                FROM wiki_fts
                WHERE wiki_fts MATCH ?
                ORDER BY score
                LIMIT 3
                """,
                (_quoted(needle),),
            ).fetchall()
            if candidates:
                candidate_text = "\n".join(
                    f"- {title} (wiki25 id={doc_id}, score={-_score:.2f})"
                    for doc_id, title, _text, _score in candidates
                )
                return (
                    f"(no exact page found for {needle!r}; candidate titles only. "
                    "Use wiki_search or call wiki_page with the exact title/id.)\n"
                    f"{candidate_text}"
                )
        if not rows:
            return "(no page found)"
        out = []
        for doc_id, title, text in rows:
            content = str(text).strip()
            out.append(f"# {title} (wiki25 id={doc_id})\n{content}")
        return "\n\n".join(out)
    finally:
        conn.close()


def _split_title(contents: str) -> tuple[str, str]:
    if "\n" not in contents:
        return "", contents
    title, text = contents.split("\n", 1)
    return title.strip().strip('"'), text.strip()


@register(
    "wiki_search",
    "Search a local offline Wikipedia/wiki25 BM25 index. Use when web search is slow, unavailable, or the question is encyclopedic.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Wikipedia search query"},
            "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
)
def wiki_search(query: str, k: int = 5) -> str:
    index_path = str(_default_index_path())
    if _is_sqlite_index(index_path):
        try:
            return _search_sqlite(index_path, query, k)
        except Exception as e:  # noqa: BLE001
            return f"OFFLINE_WIKI_UNAVAILABLE: {e}"

    try:
        bm25, docs = _load_index(index_path)
    except Exception as e:  # noqa: BLE001
        return f"OFFLINE_WIKI_UNAVAILABLE: {e}"

    scores = bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return "(no results)"
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    out: list[str] = []
    for rank, idx in enumerate(top, 1):
        if scores[idx] <= 0:
            continue
        doc = docs[idx]
        snippet = doc["text"].replace("\n", " ")
        out.append(
            f"[{rank}] {doc['title']} (wiki25 id={doc['id']}, score={scores[idx]:.2f})\n"
            f"    {snippet}"
        )
    return "\n".join(out) if out else "(no results)"


@register(
    "wiki_page",
    "Read a local offline Wikipedia/wiki25 page by exact title or wiki25 id. Use after wiki_search finds a likely page and you need full page details.",
    {
        "type": "object",
        "properties": {
            "title_or_id": {"type": "string", "description": "Exact Wikipedia title or wiki25 id from wiki_search"},
            "max_chars": {
                "type": "integer",
                "default": 4000,
                "minimum": 500,
                "maximum": 12000,
                "description": "Deprecated/ignored; wiki_page now returns full available page text.",
            },
        },
        "required": ["title_or_id"],
    },
)
def wiki_page(title_or_id: str, max_chars: int = 4000) -> str:
    index_path = str(_default_index_path())
    if _is_sqlite_index(index_path):
        try:
            return _get_page_sqlite(index_path, title_or_id, max_chars)
        except Exception as e:  # noqa: BLE001
            return f"OFFLINE_WIKI_UNAVAILABLE: {e}"

    try:
        _bm25, docs = _load_index(index_path)
    except Exception as e:  # noqa: BLE001
        return f"OFFLINE_WIKI_UNAVAILABLE: {e}"
    needle = str(title_or_id).strip().lower()
    matches = [
        doc for doc in docs
        if str(doc.get("id", "")).lower() == needle or str(doc.get("title", "")).lower() == needle
    ]
    if not matches:
        candidates = [doc for doc in docs if needle in str(doc.get("title", "")).lower()][:3]
        if candidates:
            candidate_text = "\n".join(
                f"- {doc.get('title', '')} (wiki25 id={doc.get('id', '')})"
                for doc in candidates
            )
            return (
                f"(no exact page found for {title_or_id!r}; candidate titles only. "
                "Use wiki_search or call wiki_page with the exact title/id.)\n"
                f"{candidate_text}"
            )
        return "(no page found)"
    out = []
    for doc in matches[:3]:
        content = str(doc.get("text", "")).strip()
        out.append(f"# {doc.get('title', '')} (wiki25 id={doc.get('id', '')})\n{content}")
    return "\n\n".join(out)
