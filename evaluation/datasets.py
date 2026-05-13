"""Dataset loaders → uniform list of {id, question, answer}."""
from __future__ import annotations
from typing import Iterator


def load_simpleqa(n: int | None = None) -> Iterator[dict]:
    from datasets import load_dataset
    # Tiny / canonical SimpleQA (text). Fallback to OpenAI's release if available.
    try:
        ds = load_dataset("basicv8vc/SimpleQA", split="test")
        key_q, key_a = "problem", "answer"
    except Exception:
        ds = load_dataset("lighteval/SimpleQA", split="test")
        key_q, key_a = "question", "answer"
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        yield {"id": f"simpleqa-{i}", "question": ex[key_q], "answer": ex[key_a]}


def load_2wiki(n: int | None = None) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    for i, ex in enumerate(ds):
        if n is not None and i >= n:
            break
        ans = ex.get("answer") or (ex.get("answers", {}) or {}).get("text", [None])[0]
        yield {"id": f"2wiki-{i}", "question": ex["question"], "answer": ans}


LOADERS = {"simpleqa": load_simpleqa, "2wiki": load_2wiki}
