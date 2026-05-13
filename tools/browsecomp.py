"""BrowseComp-Plus fixed-corpus retrieval tools backed by SQLite FTS5."""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from .registry import register
from .retrieval_log import record_retrieved_docids

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
    "please", "tell", "name", "fits", "following", "criteria", "criterion",
    "particular", "certain", "people", "article", "website", "academic",
    "institution", "learning", "establishment",
}


def _default_index_path() -> Path:
    return Path(os.getenv("BROWSECOMP_INDEX_PATH", "data/browsecomp-plus/browsecomp_fts.sqlite"))


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _tokenize(query):
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return (terms or _tokenize(query))[:14]


def _quoted(term: str) -> str:
    return f'"{term}"'


def _match_queries(query: str) -> list[str]:
    terms = _terms(query)
    if not terms:
        return ['""']
    queries: list[str] = []
    for n in range(min(5, len(terms)), 1, -1):
        queries.append(" AND ".join(_quoted(term) for term in terms[:n]))
    queries.append(" OR ".join(_quoted(term) for term in terms))
    return queries


def _window_queries(query: str) -> list[str]:
    tokens = _tokenize(query)
    queries: list[str] = []
    for i, token in enumerate(tokens):
        if not re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        window = [
            t for t in tokens[max(0, i - 8): i + 10]
            if t not in STOPWORDS
        ]
        if len(window) >= 3:
            queries.extend(_match_queries(" ".join(window)))
    return queries


def _connect(index_path: Path) -> sqlite3.Connection:
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Build it with: python -m scripts.build_browsecomp_fts"
        )
    return sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=30)


@register(
    "browsecomp_search",
    "Search the fixed BrowseComp-Plus corpus. Returns top documents with docid, score, URL, and text snippet. Cite docids as [docid].",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
)
def browsecomp_search(query: str, k: int = 5) -> str:
    index_path = _default_index_path()
    try:
        conn = _connect(index_path)
        try:
            scored_rows: dict[str, tuple[float, str, str]] = {}
            match_queries = _match_queries(query) + _window_queries(query)
            for match_query in dict.fromkeys(match_queries):
                for docid, url, text, score in conn.execute(
                    """
                    SELECT docid, url, text, bm25(docs_fts, 1.0, 0.2) AS score
                    FROM docs_fts
                    WHERE docs_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match_query, max(k, 10)),
                ).fetchall():
                    current = scored_rows.get(str(docid))
                    if current is None or score < current[0]:
                        scored_rows[str(docid)] = (float(score), str(url), str(text))
            rows = [
                (docid, url, text, score)
                for docid, (score, url, text) in sorted(
                    scored_rows.items(),
                    key=lambda item: item[1][0],
                )[:k]
            ]
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        return f"BROWSECOMP_UNAVAILABLE: {type(e).__name__}: {e}"

    record_retrieved_docids([row[0] for row in rows])
    out: list[str] = []
    for docid, url, text, score in rows:
        snippet = " ".join(str(text).split())[:1200]
        out.append(f"[{docid}] score={-score:.2f} url={url}\n{snippet}")
    return "\n\n".join(out) if out else "(no results)"


@register(
    "browsecomp_get_document",
    "Retrieve a full BrowseComp-Plus corpus document by docid.",
    {
        "type": "object",
        "properties": {
            "docid": {"type": "string", "description": "BrowseComp-Plus document id"},
            "max_chars": {"type": "integer", "default": 4000, "minimum": 500, "maximum": 20000},
        },
        "required": ["docid"],
    },
)
def browsecomp_get_document(docid: str, max_chars: int = 4000) -> str:
    index_path = _default_index_path()
    try:
        conn = _connect(index_path)
        try:
            row = conn.execute(
                "SELECT docid, url, text FROM docs WHERE docid = ?",
                (str(docid),),
            ).fetchone()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        return f"BROWSECOMP_UNAVAILABLE: {type(e).__name__}: {e}"

    if row is None:
        return "(document not found)"
    record_retrieved_docids([row[0]])
    text = " ".join(str(row[2]).split())[:max_chars]
    return f"[{row[0]}] url={row[1]}\n{text}"
