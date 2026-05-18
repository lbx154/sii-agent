"""Judge benchmark runs with an OpenAI-compatible semantic evaluator."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.progress import track


SYSTEM_PROMPT = """You are a strict semantic judge for 2WikiMultihopQA answers.

Decide whether the predicted answer correctly answers the question given the gold answer.
Use semantic equivalence, not exact string matching:
- Accept aliases, alternate names, spelling variants, articles, honorifics, and harmless punctuation/casing differences.
- Accept demonym/country wording when it is a valid answer to the question.
- Accept slightly shorter or longer location names only when no essential disambiguating information is lost.
- Reject different entities, wrong family relations, wrong comparison choices, wrong dates, and answers that are too broad/narrow to satisfy the gold answer.
- Reject empty, unknown, refusal, or malformed answers.

Return JSON only: {"correct": true|false, "reason": "brief"}.
"""


def _original_question(question: str) -> str:
    if "Question: " in question:
        return question.split("Question: ", 1)[1].split("\n", 1)[0].strip()
    if "Provided context:" in question:
        return question.split("Provided context:", 1)[0].strip()
    return question.strip()


def _json_from_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_run(path: Path) -> list[dict[str, Any]]:
    runs_jsonl = path / "runs.jsonl" if path.is_dir() else path
    records = []
    with runs_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            existing[str(item["id"])] = item
    return existing


def _judge_one(client: Any, model: str, record: dict[str, Any], max_retries: int) -> dict[str, Any]:
    predicted = record.get("predicted")
    expected = record.get("expected")
    payload = {
        "question": _original_question(str(record.get("question") or "")),
        "gold_answer": expected,
        "predicted_answer": predicted,
    }
    if predicted is None or str(predicted).strip() == "":
        return {
            "id": record.get("id"),
            "local_correct": record.get("correct"),
            "expected": expected,
            "predicted": predicted,
            "judge_correct": False,
            "judge_reason": "empty prediction",
        }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                temperature=0,
                max_tokens=128,
                timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            parsed = _json_from_text(content)
            if parsed is None or "correct" not in parsed:
                raise ValueError(f"could not parse judge JSON: {content[:300]}")
            correct = parsed.get("correct")
            if isinstance(correct, str):
                correct = correct.strip().lower() in {"true", "yes", "1", "correct"}
            return {
                "id": record.get("id"),
                "local_correct": record.get("correct"),
                "expected": expected,
                "predicted": predicted,
                "judge_correct": bool(correct),
                "judge_reason": str(parsed.get("reason", ""))[:500],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    return {
        "id": record.get("id"),
        "local_correct": record.get("correct"),
        "expected": expected,
        "predicted": predicted,
        "judge_correct": None,
        "judge_reason": f"judge_error: {type(last_error).__name__}: {last_error}",
    }


def _write_summary(out_jsonl: Path, items: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    judged = [item for item in items if isinstance(item.get("judge_correct"), bool)]
    correct = sum(1 for item in judged if item["judge_correct"])
    local_correct = sum(1 for item in items if item.get("local_correct") is True)
    summary = {
        **meta,
        "n": len(items),
        "n_judged": len(judged),
        "n_errors": len(items) - len(judged),
        "judge_correct": correct,
        "judge_accuracy": correct / len(judged) if judged else 0.0,
        "local_correct": local_correct,
        "local_accuracy": local_correct / len(items) if items else 0.0,
    }
    out_jsonl.with_name(out_jsonl.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def judge_run(args: argparse.Namespace, run_path: Path) -> dict[str, Any]:
    from openai import OpenAI

    records = _load_run(run_path)
    if args.limit is not None:
        records = records[: args.limit]
    out_dir = run_path if run_path.is_dir() else run_path.parent
    out_jsonl = out_dir / f"{args.out_prefix}.jsonl"
    existing = _load_existing(out_jsonl) if args.resume else {}
    results = [existing[str(record["id"])] for record in records if str(record["id"]) in existing]
    todo = [record for record in records if str(record["id"]) not in existing]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if todo:
        with out_jsonl.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                futures = {
                    pool.submit(_judge_one, client, args.model, record, args.max_retries): record
                    for record in todo
                }
                for future in track(as_completed(futures), total=len(futures), description=f"judge {run_path.name}"):
                    item = future.result()
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                    results.append(item)

    by_id = {str(item["id"]): item for item in results}
    ordered = [by_id[str(record["id"])] for record in records if str(record["id"]) in by_id]
    out_jsonl.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in ordered) + ("\n" if ordered else ""),
        encoding="utf-8",
    )
    meta = {
        "run": str(run_path),
        "judge_model": args.model,
        "judge_base_url": args.base_url,
        "concurrency": args.concurrency,
        "limit": args.limit,
    }
    _write_summary(out_jsonl, ordered, meta)
    return json.loads(out_jsonl.with_name(out_jsonl.stem + "_summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories containing runs.jsonl, or runs.jsonl files.")
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:8005/v1"))
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "Qwen3-32B"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-prefix", default="semantic_judge_qwen32")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = {
        "run_dirs": args.run_dirs,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "out_prefix": args.out_prefix,
        "max_retries": args.max_retries,
        "resume": args.resume,
    }
    print("SEMANTIC_JUDGE_CONFIG=" + json.dumps(config, ensure_ascii=False, indent=2))
    summaries = [judge_run(args, Path(run_dir)) for run_dir in args.run_dirs]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
