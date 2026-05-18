"""Select between two completed 2Wiki runs without using gold answers."""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.progress import track

from agent.scoring import score_answer


_LOCAL = threading.local()
_JSON_RE = re.compile(r"\{.*\}", re.S)
_LOW_CONFIDENCE_RE = re.compile(
    r"\b(?:none|unknown|not specified|not stated|not mentioned|cannot determine|"
    r"could not determine|unable to determine|unable to identify|not enough information|"
    r"insufficient information|cannot answer|unable to answer)\b",
    re.I,
)

SYSTEM_PROMPT = """You are a strict answer selector for 2WikiMultihopQA.

You will receive the full question, including any provided context, and two candidate final answers labeled A and B. Choose the candidate that best answers the question using only the provided question/context and the two candidates. Do not use or assume any hidden gold answer.

Selection rules:
- Prefer the candidate whose entity/relation matches the requested two-hop question.
- Prefer the candidate with the right answer type and granularity: person vs organization, country/nationality, city/state/country, award category, yes/no, date, or cause of death.
- Reject candidates that answer the wrong hop, wrong family relation, wrong comparison direction, or a role/location when an entity is requested.
- If both candidates are semantically equivalent, choose the shorter candidate.
- If both look flawed, still choose the less wrong candidate; do not invent a third answer.

Return JSON only: {"choice": "A"|"B", "selected_answer": "copy the chosen candidate answer exactly", "reason": "brief"}.
"""


def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache_key = (base_url, api_key, timeout)
    cached = getattr(_LOCAL, "client", None)
    if cached is None or getattr(_LOCAL, "client_key", None) != cache_key:
        _LOCAL.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _LOCAL.client_key = cache_key
    return _LOCAL.client


def _json_from_text(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text or "")
    payload = match.group(0) if match else text
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"selector returned non-object JSON: {text[:200]}")
    return parsed


def _load_run(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "runs.jsonl" if run_dir.is_dir() else run_dir
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _answer(record: dict[str, Any]) -> str:
    return str(record.get("predicted") or "").strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _low_confidence(answer: str) -> bool:
    if not answer:
        return True
    return bool(_LOW_CONFIDENCE_RE.search(answer))


def _cheap_choice(
    baseline_answer: str,
    evolved_answer: str,
    default_choice: str,
) -> tuple[str, str] | None:
    if _norm(baseline_answer) == _norm(evolved_answer):
        return "same", "identical answers"
    baseline_bad = _low_confidence(baseline_answer)
    evolved_bad = _low_confidence(evolved_answer)
    if baseline_bad and not evolved_bad:
        return "evolved", "baseline low-confidence/empty"
    if evolved_bad and not baseline_bad:
        return "baseline", "evolved low-confidence/empty"
    if baseline_bad and evolved_bad:
        return default_choice, "both low-confidence; used default"
    return None


def _selector_user_prompt(question: str, baseline_answer: str, evolved_answer: str) -> str:
    return json.dumps(
        {
            "question": question,
            "candidate_A": baseline_answer,
            "candidate_B": evolved_answer,
            "label_mapping": "A is candidate_baseline; B is candidate_evolved",
        },
        ensure_ascii=False,
        indent=2,
    )


def _answer_matches(value: str, candidate: str) -> bool:
    value_key = _norm(value).strip(" .;:\"'`")
    candidate_key = _norm(candidate).strip(" .;:\"'`")
    return bool(value_key) and value_key == candidate_key


def _choice_from_selected_answer(
    selected_answer: str,
    baseline_answer: str,
    evolved_answer: str,
) -> str | None:
    if _answer_matches(selected_answer, baseline_answer) and not _answer_matches(selected_answer, evolved_answer):
        return "baseline"
    if _answer_matches(selected_answer, evolved_answer) and not _answer_matches(selected_answer, baseline_answer):
        return "evolved"
    return None


def _choose_one(
    baseline: dict[str, Any],
    evolved: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_answer = _answer(baseline)
    evolved_answer = _answer(evolved)
    default_choice = args.default_choice
    cheap = _cheap_choice(baseline_answer, evolved_answer, default_choice)
    selector_reason = ""
    selector_raw = ""
    selector_error = None
    if cheap is not None:
        choice, selector_reason = cheap
    else:
        question = str(evolved.get("question") or baseline.get("question") or "")
        last_error: Exception | None = None
        choice = default_choice
        for attempt in range(args.max_retries + 1):
            try:
                response = _client(args.base_url, args.api_key, args.timeout).chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _selector_user_prompt(question, baseline_answer, evolved_answer)},
                    ],
                    temperature=0,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                selector_raw = response.choices[0].message.content or ""
                parsed = _json_from_text(selector_raw)
                parsed_choice = str(parsed.get("choice", "")).strip().upper()
                selected_answer = str(parsed.get("selected_answer", "")).strip()
                answer_choice = _choice_from_selected_answer(selected_answer, baseline_answer, evolved_answer)
                if parsed_choice in {"A", "BASELINE"}:
                    choice = "baseline"
                elif parsed_choice in {"B", "EVOLVED"}:
                    choice = "evolved"
                else:
                    raise ValueError(f"invalid selector choice: {parsed_choice!r}")
                if answer_choice is not None:
                    choice = answer_choice
                selector_reason = str(parsed.get("reason", ""))[:500]
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < args.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    selector_error = f"{type(last_error).__name__}: {last_error}"
                    selector_reason = f"selector failed; used default {default_choice}"

    if choice == "same":
        source = evolved if default_choice == "evolved" else baseline
        selected_answer = evolved_answer if default_choice == "evolved" else baseline_answer
    elif choice == "baseline":
        source = baseline
        selected_answer = baseline_answer
    else:
        source = evolved
        selected_answer = evolved_answer

    record = dict(source)
    record["predicted"] = selected_answer
    scores = score_answer(selected_answer, record.get("expected"))
    record["correct"] = scores["correct"]
    record["exact"] = scores["exact"]
    record["f1"] = scores["f1"]
    record["selector"] = {
        "choice": choice,
        "reason": selector_reason,
        "error": selector_error,
        "model": args.model,
        "source_runs": {
            "baseline": str(args.baseline_run),
            "evolved": str(args.evolved_run),
        },
        "candidate_baseline": baseline_answer,
        "candidate_evolved": evolved_answer,
        "raw": selector_raw[:1000] if args.save_selector_raw else None,
    }
    return record


def select(args: argparse.Namespace) -> dict[str, Any]:
    baseline_records = _load_run(Path(args.baseline_run))
    evolved_records = _load_run(Path(args.evolved_run))
    baseline_by_id = {str(record["id"]): record for record in baseline_records}
    evolved_by_id = {str(record["id"]): record for record in evolved_records}
    ids = [str(record["id"]) for record in evolved_records if str(record["id"]) in baseline_by_id]
    if args.limit is not None:
        ids = ids[: args.limit]
    if not ids:
        raise ValueError("No overlapping record ids between baseline and evolved runs.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "baseline_run": str(args.baseline_run),
        "evolved_run": str(args.evolved_run),
        "out": str(out),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "default_choice": args.default_choice,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
    }
    (out / "selector_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    results: list[dict[str, Any]] = []

    def run_one(record_id: str) -> dict[str, Any]:
        return _choose_one(baseline_by_id[record_id], evolved_by_id[record_id], args)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_one, record_id) for record_id in ids]
        for future in track(as_completed(futures), total=len(futures), description="2wiki selector"):
            results.append(future.result())

    by_id = {str(record["id"]): record for record in results}
    ordered = [by_id[record_id] for record_id in ids]
    (out / "runs.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in ordered) + "\n",
        encoding="utf-8",
    )

    choices = Counter(str(record.get("selector", {}).get("choice")) for record in ordered)
    n_correct = sum(1 for record in ordered if record.get("correct") is True)
    exact_correct = sum(1 for record in ordered if record.get("exact") is True)
    f1_values = [float(record["f1"]) for record in ordered if isinstance(record.get("f1"), (int, float))]
    summary = {
        "task": "2wiki",
        "mode": "selector",
        "n": len(ordered),
        "accuracy": n_correct / len(ordered) if ordered else 0.0,
        "exact_match": exact_correct / len(ordered) if ordered else 0.0,
        "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "selector_model": args.model,
        "baseline_run": str(args.baseline_run),
        "evolved_run": str(args.evolved_run),
        "choices": dict(choices),
        "config": config,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--evolved-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8004/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "sii-opd-v13-merged-sglang"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--default-choice", choices=["baseline", "evolved"], default="evolved")
    parser.add_argument("--save-selector-raw", action="store_true")
    args = parser.parse_args()
    print("SELECTOR_CONFIG=" + json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(json.dumps(select(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
