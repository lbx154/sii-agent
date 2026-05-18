"""Conservatively route 2Wiki answers using baseline/evolved runs plus selector notes."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agent.scoring import score_answer


LOW_CONFIDENCE_RE = re.compile(
    r"^(?:none|unknown|not specified|not stated|not mentioned|cannot determine|"
    r"could not determine|unable to determine|unable to identify|not enough information|"
    r"insufficient information|cannot answer|unable to answer)$",
    re.I,
)
FULL_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{3,4}\b|\b[A-Za-z]+\s+\d{1,2},\s*\d{3,4}\b")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _run_path(run_dir: Path) -> Path:
    return run_dir / "runs.jsonl" if run_dir.is_dir() else run_dir


def _original_question(question: str | None) -> str:
    text = str(question or "")
    if "Question: " in text:
        return text.split("Question: ", 1)[1].split("\n", 1)[0].strip()
    if "Provided context:" in text:
        return text.split("Provided context:", 1)[0].strip()
    return text.strip()


def _answer(record: dict[str, Any]) -> str:
    return str(record.get("predicted") or "").strip()


def _low_confidence(answer: str) -> bool:
    return bool(LOW_CONFIDENCE_RE.fullmatch(answer.strip()))


def _year_only(answer: str) -> bool:
    return bool(re.fullmatch(r"\d{3,4}", answer.strip()))


def _full_date(answer: str) -> bool:
    return bool(FULL_DATE_RE.search(answer))


def _title_like(answer: str) -> bool:
    return bool(
        re.search(
            r"\b(film|award|university|college|school|academy|national|hero|navy|revenge|diary)\b",
            answer,
            re.I,
        )
    )


def _person_like(answer: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+(?:['’-][A-Z]?[a-z]+)?\b.*\b[A-Z][a-z]+", answer))


def _choose(
    baseline: dict[str, Any],
    evolved: dict[str, Any],
    selector: dict[str, Any] | None,
) -> tuple[str, str]:
    baseline_answer = _answer(baseline)
    evolved_answer = _answer(evolved)
    if baseline_answer.lower().strip() == evolved_answer.lower().strip():
        return "evolved", "same_answer"

    if _low_confidence(evolved_answer) and not _low_confidence(baseline_answer):
        return "baseline", "evolved_low_confidence"

    question = _original_question(evolved.get("question") or baseline.get("question"))
    q = question.lower()
    selector_info = selector.get("selector", {}) if selector else {}
    selector_choice = str(selector_info.get("choice") or "").lower()
    reason = str(selector_info.get("reason") or "").lower()

    if any(term in q for term in ("when", "date")) and _year_only(evolved_answer) and _full_date(baseline_answer):
        return "baseline", "baseline_full_date"

    if (
        any(term in q for term in ("which film", "what film", "which movie", "what movie"))
        and _title_like(baseline_answer)
        and not _title_like(evolved_answer)
    ):
        return "baseline", "baseline_film_answer_type"

    if (
        any(term in q for term in ("cause of death", "how did"))
        and selector_choice == "baseline"
        and 0 < len(baseline_answer.split()) <= 4
        and 0 < len(evolved_answer.split()) <= 6
    ):
        return "baseline", "selector_cause_of_death"

    if (
        any(term in q for term in ("grandmother", "grandfather", "mother", "father", "wife", "husband", "spouse"))
        and _person_like(baseline_answer)
        and _person_like(evolved_answer)
        and selector_choice == "baseline"
        and ("not mentioned" in reason or "context states" in reason or "therefore" in reason)
    ):
        return "baseline", "selector_family_chain"

    if (
        any(term in q for term in ("which", "came out first", "born first", "older", "younger"))
        and selector_choice == "baseline"
        and any(term in reason for term in ("earlier", "later", "older", "younger", "born first", "correct answer"))
    ):
        return "baseline", "selector_comparison"

    geo_terms = ("country", "nationality", "where", "place of birth", "place of death", "born", "died")
    if selector_choice == "baseline" and any(term in reason for term in ("wrong", "incorrect")) and not any(term in q for term in geo_terms):
        return "baseline", "selector_non_geo_rejection"

    return "evolved", "default_evolved"


def route(args: argparse.Namespace) -> dict[str, Any]:
    baseline_run = Path(args.baseline_run)
    evolved_run = Path(args.evolved_run)
    selector_run = Path(args.selector_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baseline_by_id = {str(record["id"]): record for record in _load_jsonl(_run_path(baseline_run))}
    evolved_by_id = {str(record["id"]): record for record in _load_jsonl(_run_path(evolved_run))}
    selector_by_id = {str(record["id"]): record for record in _load_jsonl(_run_path(selector_run))}
    ids = [str(record_id) for record_id in evolved_by_id if record_id in baseline_by_id]
    if not ids:
        raise ValueError("No overlapping ids between baseline/evolved runs.")

    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    choices: Counter[str] = Counter()
    for record_id in ids:
        choice, reason = _choose(baseline_by_id[record_id], evolved_by_id[record_id], selector_by_id.get(record_id))
        source = baseline_by_id[record_id] if choice == "baseline" else evolved_by_id[record_id]
        selected_answer = _answer(source)
        record = dict(source)
        record["predicted"] = selected_answer
        scores = score_answer(selected_answer, record.get("expected"))
        record["correct"] = scores["correct"]
        record["exact"] = scores["exact"]
        record["f1"] = scores["f1"]
        record["router"] = {
            "choice": choice,
            "reason": reason,
            "baseline_answer": _answer(baseline_by_id[record_id]),
            "evolved_answer": _answer(evolved_by_id[record_id]),
            "selector_choice": (selector_by_id.get(record_id) or {}).get("selector", {}).get("choice"),
            "selector_reason": (selector_by_id.get(record_id) or {}).get("selector", {}).get("reason"),
        }
        choices[choice] += 1
        reasons[reason] += 1
        records.append(record)

    (out / "runs.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    n_correct = sum(1 for record in records if record.get("correct") is True)
    exact_correct = sum(1 for record in records if record.get("exact") is True)
    f1_values = [float(record["f1"]) for record in records if isinstance(record.get("f1"), (int, float))]
    summary = {
        "task": "2wiki",
        "mode": "conservative_selector_router",
        "n": len(records),
        "accuracy": n_correct / len(records) if records else 0.0,
        "exact_match": exact_correct / len(records) if records else 0.0,
        "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "baseline_run": str(baseline_run),
        "evolved_run": str(evolved_run),
        "selector_run": str(selector_run),
        "choices": dict(choices),
        "reasons": dict(reasons),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--evolved-run", required=True)
    parser.add_argument("--selector-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("ROUTER_CONFIG=" + json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(json.dumps(route(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
