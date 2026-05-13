"""Build a compact offline wiki25 BM25 source JSONL from HuggingFace shards.

The full wiki25 corpus is split into ~24 GB of newline-delimited JSON files.
For fast local experiments this script defaults to a compact prefix index; pass
--max-docs 0 to process all downloaded/input shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

HF_BASE = "https://huggingface.co/datasets/XLDDD/wiki25/resolve/main"
SHARDS = [f"wiki25_part_aa{chr(c)}" for c in range(ord("a"), ord("x"))]


def iter_remote_lines(shard: str):
    req = Request(f"{HF_BASE}/{shard}", headers={"User-Agent": "sii-agent/0.1"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed HTTPS URL
        for raw in resp:
            yield raw.decode("utf-8")


def iter_local_lines(path: Path):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.name.startswith("wiki25_part_"))
    else:
        files = [path]
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            yield from f


def valid_line(line: str) -> str | None:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "id" not in item or "contents" not in item:
        return None
    return json.dumps({"id": item["id"], "contents": item["contents"]}, ensure_ascii=False)


def build(out: Path, max_docs: int, source: Path | None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as dst:
        if source:
            line_iter = iter_local_lines(source)
            for line in line_iter:
                normalized = valid_line(line)
                if normalized is None:
                    continue
                dst.write(normalized + "\n")
                count += 1
                if max_docs and count >= max_docs:
                    break
        else:
            per_shard_limit = 0
            if max_docs:
                per_shard_limit = max(1, (max_docs + len(SHARDS) - 1) // len(SHARDS))
            for shard in SHARDS:
                shard_count = 0
                for line in iter_remote_lines(shard):
                    normalized = valid_line(line)
                    if normalized is None:
                        continue
                    dst.write(normalized + "\n")
                    count += 1
                    shard_count += 1
                    if max_docs and count >= max_docs:
                        return count
                    if per_shard_limit and shard_count >= per_shard_limit:
                        break
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/wiki25/wiki25_sample.jsonl")
    parser.add_argument("--max-docs", type=int, default=50_000, help="0 means no limit")
    parser.add_argument("--source", type=Path, help="Local wiki25 shard file or directory")
    args = parser.parse_args()
    count = build(Path(args.out), args.max_docs, args.source)
    print(f"Wrote {count} docs to {args.out}")


if __name__ == "__main__":
    main()
