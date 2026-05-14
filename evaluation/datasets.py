"""Dataset loaders → uniform list of {id, question, answer}."""
from __future__ import annotations
from typing import Iterator

DEFAULT_SPLITS = {"simpleqa": "test", "2wiki": "validation"}
AVAILABLE_SPLITS = {
    "simpleqa": ("test", "few_shot"),
    "2wiki": ("train", "validation", "test"),
}


def load_simpleqa(n: int | None = None, split: str = "test") -> Iterator[dict]:
    from datasets import load_dataset
    if split == "train":
        raise ValueError("SimpleQA has no public train split in the configured loaders; use 2Wiki train or SimpleQA few_shot/test with an explicit held-out carve-out.")
    if split not in AVAILABLE_SPLITS["simpleqa"]:
        raise ValueError(f"Unsupported SimpleQA split '{split}'. Available: {AVAILABLE_SPLITS['simpleqa']}")

    if split == "test":
        # Tiny / canonical SimpleQA (text). Fallback to OpenAI's release if available.
        try:
            ds = load_dataset("basicv8vc/SimpleQA", split=split)
            key_q, key_a = "problem", "answer"
        except Exception:
            ds = load_dataset("lighteval/SimpleQA", split=split)
            key_q, key_a = "question", "answer"
    else:
        ds = load_dataset("lighteval/SimpleQA", split=split)
        key_q = "question" if "question" in ds.column_names else "problem"
        key_a = "answer"
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        yield {"id": f"simpleqa-{split}-{i}", "task": "simpleqa", "split": split, "question": ex[key_q], "answer": ex[key_a]}


def load_2wiki(n: int | None = None, split: str = "validation") -> Iterator[dict]:
    from datasets import load_dataset
    if split not in AVAILABLE_SPLITS["2wiki"]:
        raise ValueError(f"Unsupported 2Wiki split '{split}'. Available: {AVAILABLE_SPLITS['2wiki']}")
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        ans = ex.get("answer") or (ex.get("answers", {}) or {}).get("text", [None])[0]
        yield {"id": f"2wiki-{split}-{i}", "task": "2wiki", "split": split, "question": ex["question"], "answer": ans}


LOADERS = {"simpleqa": load_simpleqa, "2wiki": load_2wiki}


def load_examples(task: str, n: int | None = None, offset: int = 0, split: str | None = None) -> list[dict]:
    split = split or DEFAULT_SPLITS[task]
    limit = n + offset if n is not None else None
    return list(LOADERS[task](limit, split=split))[offset:]
