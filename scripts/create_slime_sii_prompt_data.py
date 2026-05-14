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
    parser.add_argument("--task", default="2wiki", choices=["2wiki", "simpleqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allowed-tools",
        default="wiki_search,web_search,final_answer",
        help="Comma-separated tool names to expose during slime rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed_tools = [name.strip() for name in args.allowed_tools.split(",") if name.strip()]

    with out.open("w", encoding="utf-8") as f:
        for ex in load_examples(args.task, n=args.n, offset=args.offset, split=args.split):
            record = {
                "question": ex["question"],
                "answer": ex.get("answer"),
                "metadata": {
                    "id": ex.get("id"),
                    "task": ex.get("task", args.task),
                    "split": ex.get("split", args.split),
                    "allowed_tools": allowed_tools,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} prompts to {out}")


if __name__ == "__main__":
    main()
