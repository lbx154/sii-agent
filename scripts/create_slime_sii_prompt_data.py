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
    parser.add_argument("--task", default="2wiki", choices=["2wiki", "simpleqa", "browsecomp-plus", "mmsearch"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=32, help="Number of prompts; 0 means all available examples.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Comma-separated tool names to expose during slime rollout.",
    )
    parser.add_argument(
        "--chat-prompt",
        action="store_true",
        help="Write question as a chat message list for slime processor-backed models.",
    )
    return parser.parse_args()


def default_allowed_tools(task: str) -> list[str]:
    if task == "browsecomp-plus":
        return ["search", "final_answer"]
    if task == "mmsearch":
        return [
            "visual_web_search",
            "image_to_text",
            "image_to_search_queries",
            "reverse_image_search",
            "web_search",
            "wiki_search",
            "wiki_page",
            "browser_open",
            "browser_open_many",
            "final_answer",
        ]
    return ["wiki_search", "web_search", "browser_open", "browser_open_many", "final_answer"]


def mmsearch_text_allowed_tools() -> list[str]:
    return ["web_search", "wiki_search", "wiki_page", "browser_open", "browser_open_many", "final_answer"]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed_tools = (
        [name.strip() for name in args.allowed_tools.split(",") if name.strip()]
        if args.allowed_tools
        else default_allowed_tools(args.task)
    )

    n = None if args.n == 0 else args.n
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in load_examples(args.task, n=n, offset=args.offset, split=args.split):
            example_allowed_tools = (
                mmsearch_text_allowed_tools()
                if args.task == "mmsearch" and not ex.get("image") and not args.allowed_tools
                else allowed_tools
            )
            metadata = {
                "id": ex.get("id"),
                "task": ex.get("task", args.task),
                "split": ex.get("split", args.split),
                "allowed_tools": example_allowed_tools,
            }
            if ex.get("query_id") is not None:
                metadata["query_id"] = ex.get("query_id")
            if ex.get("evidence_docids"):
                metadata["evidence_docids"] = ex.get("evidence_docids")
            if ex.get("image"):
                metadata["image"] = ex.get("image")
            for key in ("category", "data_source", "area", "subfield", "timestamp", "gt_requery"):
                if ex.get(key) is not None:
                    metadata[key] = ex.get(key)
            record = {
                "question": [{"role": "user", "content": ex["question"]}] if args.chat_prompt else ex["question"],
                "answer": ex.get("answer"),
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} prompts to {out}")


if __name__ == "__main__":
    main()
