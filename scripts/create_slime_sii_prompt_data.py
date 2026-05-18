"""Export held-out-safe SII prompts for slime OPD smoke/full runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="2wiki", choices=["2wiki", "simpleqa", "browsecomp-plus"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Comma-separated tool names to expose during slime rollout.",
    )
    return parser.parse_args()


def default_allowed_tools(task: str) -> list[str]:
    if task == "browsecomp-plus":
        return ["search", "final_answer"]
    return ["wiki_search", "web_search", "browser_open", "browser_open_many", "final_answer"]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed_tools = (
        [name.strip() for name in args.allowed_tools.split(",") if name.strip()]
        if args.allowed_tools
        else default_allowed_tools(args.task)
    )

    with out.open("w", encoding="utf-8") as f:
        for ex in load_examples(args.task, n=args.n, offset=args.offset, split=args.split):
            metadata = {
                "id": ex.get("id"),
                "task": ex.get("task", args.task),
                "split": ex.get("split", args.split),
                "allowed_tools": allowed_tools,
            }
            if ex.get("query_id") is not None:
                metadata["query_id"] = ex.get("query_id")
            if ex.get("evidence_docids"):
                metadata["evidence_docids"] = ex.get("evidence_docids")
            record = {
                "question": ex["question"],
                "answer": ex.get("answer"),
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} prompts to {out}")


if __name__ == "__main__":
    main()
