"""Route BrowseComp-Plus queries between a primary run and a fallback run.

The intended use is adaptive evaluation: trust a cheap primary trajectory when its
answer looks reliable, and use a more expensive selector/ensemble only when the
primary trajectory shows cheap failure signals.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


REFUSAL_RE = re.compile(
    r"cannot find|could not identify|not enough information|insufficient|unable to|"
    r"i do not have|i don't have|no definitive|cannot determine|not found|"
    r"do not know|cannot verify",
    re.I,
)


def _safe_qid(query_id: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(query_id))


def _answer(run: dict) -> str:
    result = run.get("result") or []
    if not result:
        return ""
    return str(result[-1].get("output") or "").strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _should_route(primary: dict, answer_max_chars: int, search_threshold: int, route_refusals: bool) -> tuple[bool, list[str]]:
    answer = _answer(primary)
    counts = primary.get("tool_call_counts") or {}
    reasons: list[str] = []
    if route_refusals and REFUSAL_RE.search(answer):
        reasons.append("refusal")
    if answer_max_chars > 0 and len(answer) > answer_max_chars:
        reasons.append(f"answer_len>{answer_max_chars}")
    search_calls = int(counts.get("search", 0))
    if search_threshold > 0 and search_calls >= search_threshold:
        reasons.append(f"search>={search_threshold}")
    return bool(reasons), reasons


def route(args: argparse.Namespace) -> dict:
    primary = Path(args.primary_run)
    fallback = Path(args.fallback_run)
    out = Path(args.out)
    runs_out = out / "runs"
    runs_out.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.ground_truth) if args.ground_truth else primary / "browsecomp_plus_decrypted.jsonl"
    qrel_path = Path(args.qrel_evidence) if args.qrel_evidence else primary / "qrel_evidence.txt"
    shutil.copyfile(gt_path, out / "browsecomp_plus_decrypted.jsonl")
    if qrel_path.exists():
        shutil.copyfile(qrel_path, out / "qrel_evidence.txt")

    choices: Counter[str] = Counter()
    for primary_path in sorted((primary / "runs").glob("*.json"), key=lambda path: int(path.stem)):
        query_id = primary_path.stem
        primary_record = _load_json(primary_path)
        use_fallback, reasons = _should_route(
            primary_record,
            answer_max_chars=args.answer_max_chars,
            search_threshold=args.search_threshold,
            route_refusals=not args.no_route_refusals,
        )
        if use_fallback:
            fallback_path = fallback / "runs" / f"{_safe_qid(query_id)}.json"
            if not fallback_path.exists():
                raise FileNotFoundError(f"Missing fallback run for query {query_id}: {fallback_path}")
            record = _load_json(fallback_path)
            record.setdefault("metadata", {})["smart_router"] = {
                "choice": "fallback",
                "reasons": reasons,
                "primary_run": str(primary),
                "fallback_run": str(fallback),
            }
            choices["fallback"] += 1
        else:
            record = primary_record
            record.setdefault("metadata", {})["smart_router"] = {
                "choice": "primary",
                "reasons": [],
                "primary_run": str(primary),
                "fallback_run": str(fallback),
            }
            choices["primary"] += 1
        (runs_out / f"{_safe_qid(query_id)}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "task": "browsecomp-plus",
        "mode": "smart_router",
        "n": sum(choices.values()),
        "primary_run": str(primary),
        "fallback_run": str(fallback),
        "rule": {
            "route_refusals": not args.no_route_refusals,
            "answer_max_chars": args.answer_max_chars,
            "search_threshold": args.search_threshold,
        },
        "choices": dict(choices),
        "runs_dir": str(runs_out),
        "ground_truth": str(out / "browsecomp_plus_decrypted.jsonl"),
        "qrel_evidence": str(out / "qrel_evidence.txt") if qrel_path.exists() else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-run", required=True)
    parser.add_argument("--fallback-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--qrel-evidence", default=None)
    parser.add_argument("--answer-max-chars", type=int, default=300)
    parser.add_argument("--search-threshold", type=int, default=9)
    parser.add_argument("--no-route-refusals", action="store_true")
    args = parser.parse_args()
    print(json.dumps(route(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
