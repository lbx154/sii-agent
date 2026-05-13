"""Offline Wikipedia search over a local wiki25 JSONL shard/index."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from .registry import register

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _default_index_path() -> Path:
    return Path(os.getenv("WIKI25_INDEX_PATH", "data/wiki25/wiki25_sample.jsonl"))


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
        snippet = doc["text"].replace("\n", " ")[:500]
        out.append(
            f"[{rank}] {doc['title']} (wiki25 id={doc['id']}, score={scores[idx]:.2f})\n"
            f"    {snippet}"
        )
    return "\n".join(out) if out else "(no results)"
