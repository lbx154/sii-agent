"""Build a persistent SQLite FTS5 index from the local wiki25 JSONL file."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _split_title(contents: str) -> tuple[str, str]:
    if "\n" not in contents:
        return "", contents
    title, text = contents.split("\n", 1)
    return title.strip().strip('"'), text.strip()


def build(source: Path, out: Path, batch_size: int = 5000) -> int:
    if not source.exists():
        raise FileNotFoundError(source)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    conn = sqlite3.connect(out)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            """
            CREATE VIRTUAL TABLE wiki_fts USING fts5(
                id UNINDEXED,
                title,
                text,
                tokenize='unicode61'
            )
            """
        )

        rows: list[tuple[str, str, str]] = []
        count = 0
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                title, text = _split_title(str(item.get("contents", "")))
                rows.append((str(item.get("id", "")), title, text))
                if len(rows) >= batch_size:
                    conn.executemany(
                        "INSERT INTO wiki_fts(id, title, text) VALUES (?, ?, ?)",
                        rows,
                    )
                    count += len(rows)
                    rows.clear()

        if rows:
            conn.executemany(
                "INSERT INTO wiki_fts(id, title, text) VALUES (?, ?, ?)",
                rows,
            )
            count += len(rows)
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/wiki25/wiki25_sample.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/wiki25/wiki25_fts.sqlite"))
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    count = build(args.source, args.out, args.batch_size)
    print(f"Wrote {count} docs to {args.out}")


if __name__ == "__main__":
    main()
