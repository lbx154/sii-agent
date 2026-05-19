"""Answer normalization and lightweight exact/F1 scoring."""
from __future__ import annotations

import re
import string
import unicodedata

HONORIFICS = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir", "dame"}


def normalize_answer(text: str | None) -> str:
    """Lowercase, strip punctuation/articles, and normalize whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    tokens = text.split()
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)
    return " ".join(tokens)


def token_f1(predicted: str | None, expected: str | None) -> float:
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(expected).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            overlap += 1
            gold_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str | None, expected: str | None) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)


def score_answer(predicted: str | None, expected: str | None) -> dict[str, float | bool | None]:
    if expected is None:
        return {"exact": None, "f1": None, "correct": None}
    exact = exact_match(predicted, expected)
    f1 = token_f1(predicted, expected)
    return {"exact": exact, "f1": f1, "correct": exact or f1 >= 0.9}


def judge_answer(predicted: str | None, expected: str | None) -> bool | None:
    correct = score_answer(predicted, expected)["correct"]
    return correct if isinstance(correct, bool) else None
