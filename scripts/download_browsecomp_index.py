"""Download official BrowseComp-Plus prebuilt retrieval indexes."""
from __future__ import annotations

import argparse
from pathlib import Path


def download(
    out: Path,
    repo_id: str = "Tevatron/browsecomp-plus-indexes",
    include_dense: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    patterns = ["bm25/*"]
    if include_dense:
        patterns.extend([
            "qwen3-embedding-0.6b/*",
            "qwen3-embedding-4b/*",
            "qwen3-embedding-8b/*",
        ])

    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(out),
    )
    return out / "bm25"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official BrowseComp-Plus indexes.")
    parser.add_argument("--out", type=Path, default=Path("indexes"))
    parser.add_argument("--repo-id", default="Tevatron/browsecomp-plus-indexes")
    parser.add_argument(
        "--include-dense",
        action="store_true",
        help="Also download official Qwen3-Embedding FAISS index shards.",
    )
    args = parser.parse_args()
    bm25_path = download(args.out, repo_id=args.repo_id, include_dense=args.include_dense)
    print(f"Downloaded official BrowseComp-Plus BM25 index to {bm25_path}")


if __name__ == "__main__":
    main()
