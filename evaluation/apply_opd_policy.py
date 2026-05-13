"""Apply a trained OPD trajectory ranker to multiple candidate run JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.scoring import score_answer
from training.opd import load_run_records, score_record, train_task_splits


def _group(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["id"])].append(record)
    return dict(grouped)


def _source_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("_source"))].append(record)
    metrics = {}
    for source, items in by_source.items():
        correct = sum(1 for item in items if item.get("correct"))
        metrics[source] = {
            "n": len(items),
            "accuracy": correct / len(items) if items else 0.0,
            "avg_tool_calls": sum(float(item.get("tool_calls") or 0) for item in items) / len(items) if items else 0.0,
        }
    return metrics


def _oracle_metrics(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    total = correct = 0
    for candidates in grouped.values():
        scored = [bool(candidate.get("correct")) for candidate in candidates]
        if not scored:
            continue
        total += 1
        correct += int(any(scored))
    return {"n": total, "accuracy": correct / total if total else 0.0}


def _assert_no_split_leakage(model: dict[str, Any], records: list[dict[str, Any]], allow_same_split: bool) -> None:
    if allow_same_split:
        return
    train_splits = set(model.get("metadata", {}).get("train_task_splits") or [])
    eval_splits = set(train_task_splits(records))
    overlap = train_splits & eval_splits
    if overlap:
        raise ValueError(
            "Refusing to evaluate OPD on the same task/split used for training: "
            f"{sorted(overlap)}. Use a held-out split or pass --allow-same-split for debugging only."
        )


def apply(args: argparse.Namespace) -> dict[str, Any]:
    model = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    records = load_run_records(args.runs)
    _assert_no_split_leakage(model, records, args.allow_same_split)
    grouped = _group(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected_records = []
    n_total = n_correct = exact_correct = n_f1 = 0
    sum_f1 = sum_tools = sum_steps = 0.0
    for example_id in sorted(grouped):
        candidates = grouped[example_id]
        if not candidates:
            continue
        selected = max(candidates, key=lambda record: score_record(record, model))
        selected = dict(selected)
        selected["opd_score"] = score_record(selected, model)
        selected["opd_candidate_count"] = len(candidates)
        selected["opd_selected_source"] = selected.get("_source")
        selected_records.append(selected)

        scores = score_answer(selected.get("predicted"), selected.get("expected"))
        n_total += 1
        if scores["correct"]:
            n_correct += 1
        if scores["exact"]:
            exact_correct += 1
        if isinstance(scores["f1"], (int, float)):
            sum_f1 += float(scores["f1"])
            n_f1 += 1
        sum_tools += float(selected.get("tool_calls") or 0)
        sum_steps += float(selected.get("steps") or 0)

    with (out / "runs.jsonl").open("w", encoding="utf-8") as f:
        for record in selected_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "policy": str(args.policy),
        "runs": [str(path) for path in args.runs],
        "n": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "exact_match": exact_correct / n_total if n_total else 0.0,
        "avg_f1": sum_f1 / n_f1 if n_f1 else 0.0,
        "avg_steps": sum_steps / n_total if n_total else 0.0,
        "avg_tool_calls": sum_tools / n_total if n_total else 0.0,
        "source_metrics": _source_metrics(records),
        "oracle_best_of_n": _oracle_metrics(grouped),
        "train_task_splits": model.get("metadata", {}).get("train_task_splits"),
        "eval_task_splits": train_task_splits(records),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a trained OPD policy to candidate run JSONL files.")
    parser.add_argument("--policy", required=True, help="Path to opd_policy.json produced by training.opd.")
    parser.add_argument("--runs", nargs="+", required=True, help="Candidate run dirs or runs.jsonl files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-same-split", action="store_true", help="Debug only: allow eval on a train task/split.")
    args = parser.parse_args()
    apply(args)


if __name__ == "__main__":
    main()

