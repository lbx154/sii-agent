"""Build a legacy SQLite FTS5 index for BrowseComp-Plus.

The default BrowseComp path uses the official Pyserini/Lucene BM25 index from
Tevatron/browsecomp-plus-indexes. Use scripts.download_browsecomp_index for
official-compatible runs; this script is kept for lightweight local fallback.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path


def _iter_source(source: Path | None, limit: int | None = None) -> Iterable[dict]:
    count = 0
    if source is not None:
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return
        return

    from datasets import load_dataset

    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    for item in ds:
        yield item
        count += 1
        if limit is not None and count >= limit:
            return


def build(out: Path, source: Path | None = None, batch_size: int = 5000, limit: int | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    conn = sqlite3.connect(out)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE docs(docid TEXT PRIMARY KEY, url TEXT, text TEXT)")
        conn.execute(
            """
            CREATE VIRTUAL TABLE docs_fts USING fts5(
                docid UNINDEXED,
                url UNINDEXED,
                text,
                tokenize='porter unicode61'
            )
            """
        )

        rows: list[tuple[str, str, str]] = []
        count = 0
        for item in _iter_source(source, limit=limit):
            docid = str(item.get("docid", ""))
            if not docid:
                continue
            rows.append((docid, str(item.get("url", "")), str(item.get("text", ""))))
            if len(rows) >= batch_size:
                _insert_rows(conn, rows)
                count += len(rows)
                rows.clear()

        if rows:
            _insert_rows(conn, rows)
            count += len(rows)
        conn.commit()
        return count
    finally:
        conn.close()


def _insert_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    conn.executemany("INSERT OR REPLACE INTO docs(docid, url, text) VALUES (?, ?, ?)", rows)
    conn.executemany("INSERT INTO docs_fts(docid, url, text) VALUES (?, ?, ?)", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None, help="Optional local corpus JSONL")
    parser.add_argument("--out", type=Path, default=Path("data/browsecomp-plus/browsecomp_fts.sqlite"))
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="0 means all documents")
    args = parser.parse_args()
    count = build(
        args.out,
        source=args.source,
        batch_size=args.batch_size,
        limit=args.limit or None,
    )
    print(f"Wrote {count} docs to {args.out}")


if __name__ == "__main__":
    main()
