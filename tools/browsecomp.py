"""BrowseComp-Plus fixed-corpus retrieval tools.

Default backend matches the official BrowseComp-Plus BM25 setup: a Pyserini
Lucene index downloaded from Tevatron/browsecomp-plus-indexes. A SQLite FTS5
path is still accepted for legacy experiments.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from .registry import register
from .retrieval_log import record_retrieved_docids

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_K = 5
SNIPPET_MAX_TOKENS = 512
DEFAULT_TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER_NAME = os.getenv("BROWSECOMP_SNIPPET_TOKENIZER", DEFAULT_TOKENIZER_NAME)
BM25_K1 = float(os.getenv("BROWSECOMP_BM25_K1", "10"))
BM25_B = float(os.getenv("BROWSECOMP_BM25_B", "1"))
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
    "please", "tell", "name", "fits", "following", "criteria", "criterion",
    "particular", "certain", "people", "article", "website", "academic",
    "institution", "learning", "establishment",
    "looking", "look", "find", "identify", "answer", "question", "query",
    "clue", "clues", "following", "based", "known", "called", "named",
    "which", "that", "also", "first", "last", "one", "two", "three",
}


def _default_index_path() -> Path:
    return Path(os.getenv("BROWSECOMP_INDEX_PATH", "indexes/bm25"))


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
    return (terms or _tokenize(query))[:24]


def _quoted(term: str) -> str:
    return f'"{term}"'


def _is_sqlite_index(index_path: Path) -> bool:
    return index_path.suffix in {".db", ".sqlite", ".sqlite3"}


def _connect_sqlite(index_path: Path) -> sqlite3.Connection:
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Build the legacy SQLite index with: "
            "python -m scripts.build_browsecomp_fts"
        )
    return sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, timeout=30)


def _fts_query(query: str) -> str:
    terms = _terms(query)
    if not terms:
        return '""'
    return " OR ".join(_quoted(term) for term in terms)


@lru_cache(maxsize=1)
def _snippet_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for official BrowseComp-Plus 512-token snippets. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if TOKENIZER_NAME == DEFAULT_TOKENIZER_NAME:
        repo_root = Path(__file__).resolve().parents[1]
        for local_name in ("Qwen3-32B", "Qwen3.5-9B"):
            for local_path in (Path(local_name), repo_root / local_name):
                if (local_path / "tokenizer_config.json").exists():
                    return AutoTokenizer.from_pretrained(str(local_path))

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load snippet tokenizer '{TOKENIZER_NAME}'. "
            "Set BROWSECOMP_SNIPPET_TOKENIZER to a local Qwen tokenizer path."
        ) from exc


def _truncate_snippet(text: str, max_tokens: int = SNIPPET_MAX_TOKENS) -> str:
    normalized = " ".join(str(text).split())
    if max_tokens <= 0:
        return normalized
    tokenizer = _snippet_tokenizer()
    tokens = tokenizer.encode(normalized, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return normalized
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


@lru_cache(maxsize=2)
def _lucene_searcher(index_path: str) -> Any:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the official BM25 index with: "
            "python -m scripts.download_browsecomp_index"
        )
    try:
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        raise RuntimeError(
            "pyserini is required for the official BrowseComp-Plus BM25 index. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    try:
        searcher = LuceneSearcher(str(path))
        searcher.set_bm25(k1=BM25_K1, b=BM25_B)
        return searcher
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{path} is not a valid Pyserini/Lucene index. "
            "Download the official index with: python -m scripts.download_browsecomp_index"
        ) from exc


def _contents_from_raw(raw: object) -> str:
    if raw is None:
        return ""
    raw_text = str(raw)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    if isinstance(parsed, dict):
        return str(parsed.get("contents") or parsed.get("text") or raw_text)
    return raw_text


def _lucene_document_text(searcher: Any, docid: str) -> str | None:
    doc = searcher.doc(str(docid))
    if doc is None:
        return None
    return _contents_from_raw(doc.raw())


def _search_lucene(index_path: Path, query: str, k: int = DEFAULT_K) -> list[tuple[str, float, str]]:
    searcher = _lucene_searcher(str(index_path))
    rows: list[tuple[str, float, str]] = []
    for hit in searcher.search(query, k):
        raw = None
        lucene_document = getattr(hit, "lucene_document", None)
        if lucene_document is not None:
            raw = lucene_document.get("raw")
        text = _contents_from_raw(raw)
        if not text:
            text = _lucene_document_text(searcher, str(hit.docid)) or ""
        rows.append((str(hit.docid), float(hit.score), text))
    return rows


def _search_sqlite(index_path: Path, query: str, k: int = DEFAULT_K) -> list[tuple[str, float, str]]:
    conn = _connect_sqlite(index_path)
    try:
        rows = conn.execute(
            """
            SELECT docid, bm25(docs_fts, 1.0, 0.2) AS score, text
            FROM docs_fts
            WHERE docs_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (_fts_query(query), k),
        ).fetchall()
    finally:
        conn.close()
    return [(str(docid), float(score), str(text)) for docid, score, text in rows]


def _search_rows(query: str, k: int = DEFAULT_K) -> list[tuple[str, float, str]]:
    index_path = _default_index_path()
    if _is_sqlite_index(index_path):
        rows = _search_sqlite(index_path, query, k)
        return [(docid, -score, text) for docid, score, text in rows]
    return _search_lucene(index_path, query, k)


def _official_search_payload(query: str, k: int = DEFAULT_K) -> str:
    try:
        k = max(1, min(int(k), 20))
        rows = _search_rows(query, k)
        record_retrieved_docids([docid for docid, _, _ in rows])
        results = [
            {
                "docid": docid,
                "score": score,
                "snippet": _truncate_snippet(text),
            }
            for docid, score, text in rows
        ]
    except Exception as e:  # noqa: BLE001
        return f"BROWSECOMP_UNAVAILABLE: {type(e).__name__}: {e}"
    return json.dumps(results, ensure_ascii=False, indent=2)


@register(
    "search",
    "Search the BrowseComp-Plus fixed corpus only. Use for BrowseComp tasks, not general web/SimpleVQA questions. Returns top-5 hits with docid, score, and snippet.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
def search(query: str) -> str:
    return _official_search_payload(query)


@register(
    "browsecomp_search",
    "Search the local BrowseComp fixed corpus/index. Use this first for text-only BrowseComp-style questions. Returns docid, score, and a query-focused snippet from the local corpus; this is not live web search.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Focused search query string, preferably with distinctive quoted phrases or candidate names."},
            "k": {"type": "integer", "default": DEFAULT_K, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
def browsecomp_search(query: str, k: int = DEFAULT_K) -> str:
    return _official_search_payload(query, k=k)


@register(
    "get_document",
    "Retrieve a full BrowseComp-Plus corpus document by its docid.",
    {
        "type": "object",
        "properties": {
            "docid": {"type": "string", "description": "Document ID to retrieve"},
        },
        "required": ["docid"],
        "additionalProperties": False,
    },
)
def get_document(docid: str) -> str:
    return _get_document_payload(docid)


@register(
    "browsecomp_open",
    "Open/retrieve a full document from the local BrowseComp fixed corpus by docid returned from browsecomp_search.",
    {
        "type": "object",
        "properties": {
            "docid": {"type": "string", "description": "Document ID returned by browsecomp_search."},
        },
        "required": ["docid"],
        "additionalProperties": False,
    },
)
def browsecomp_open(docid: str) -> str:
    return _get_document_payload(docid)


def _get_document_payload(docid: str) -> str:
    index_path = _default_index_path()
    try:
        if _is_sqlite_index(index_path):
            conn = _connect_sqlite(index_path)
            try:
                row = conn.execute(
                    "SELECT docid, text FROM docs WHERE docid = ?",
                    (str(docid),),
                ).fetchone()
            finally:
                conn.close()
            text = None if row is None else str(row[1])
        else:
            text = _lucene_document_text(_lucene_searcher(str(index_path)), str(docid))
    except Exception as e:  # noqa: BLE001
        return f"BROWSECOMP_UNAVAILABLE: {type(e).__name__}: {e}"

    if text is None:
        return json.dumps({"error": f"Document with docid '{docid}' not found"}, ensure_ascii=False)
    return json.dumps({"docid": str(docid), "text": text}, ensure_ascii=False, indent=2)
