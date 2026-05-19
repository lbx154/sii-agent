"""Judge run outputs with a remote OpenAI-compatible Responses API model.

The script intentionally reads credentials from the environment and never stores
API keys in output files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.progress import track

from evaluation.judge_semantic import (
    SYSTEM_PROMPT,
    _judge_from_text,
    _load_existing,
    _load_run,
    _original_question,
    _write_summary,
)


DEFAULT_BASE_URL = "https://ai4m6.openai.azure.com/openai/v1/"
DEFAULT_MODEL = "gpt-5.4"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    try:
        payload = response.model_dump()
    except Exception:  # noqa: BLE001
        payload = response if isinstance(response, dict) else {}

    parts: list[str] = []
    for item in payload.get("output", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            value = content.get("text") or content.get("output_text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _judge_one(client: Any, args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    predicted = record.get("predicted")
    expected = record.get("expected")
    if predicted is None or str(predicted).strip() == "":
        return {
            "id": record.get("id"),
            "local_correct": record.get("correct"),
            "expected": expected,
            "predicted": predicted,
            "judge_correct": False,
            "judge_reason": "empty prediction",
        }

    payload = {
        "question": _original_question(str(record.get("question") or "")),
        "gold_answer": expected,
        "predicted_answer": predicted,
    }
    user_input = (
        "Judge this 2WikiMultihopQA answer. Return JSON only.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    last_error: BaseException | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=user_input,
                reasoning={"effort": args.reasoning_effort},
                max_output_tokens=args.max_output_tokens,
                store=False,
                timeout=args.timeout,
            )
            content = _response_text(response)
            parsed = _judge_from_text(content)
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
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))

    return {
        "id": record.get("id"),
        "local_correct": record.get("correct"),
        "expected": expected,
        "predicted": predicted,
        "judge_correct": None,
        "judge_reason": f"judge_error: {type(last_error).__name__}: {last_error}",
    }


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
                futures = {pool.submit(_judge_one, client, args, record): record for record in todo}
                for future in track(as_completed(futures), total=len(futures), description=f"remote judge {run_path.name}"):
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
        "judge_api": "responses",
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "limit": args.limit,
    }
    _write_summary(out_jsonl, ordered, meta)
    return json.loads(out_jsonl.with_name(out_jsonl.stem + "_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic judge with a remote Responses API model.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories containing runs.jsonl, or runs.jsonl files.")
    parser.add_argument("--base-url", default=os.getenv("REMOTE_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("REMOTE_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default=os.getenv("REMOTE_JUDGE_REASONING_EFFORT", "high"))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("REMOTE_JUDGE_CONCURRENCY", "32")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-prefix", default=os.getenv("REMOTE_JUDGE_OUT_PREFIX", "semantic_judge_gpt54_remote"))
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required. Export it in the shell or pass --api-key.")

    config = {
        "run_dirs": args.run_dirs,
        "base_url": args.base_url,
        "model": args.model,
        "api": "responses",
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "out_prefix": args.out_prefix,
        "max_retries": args.max_retries,
        "resume": args.resume,
    }
    print("REMOTE_JUDGE_CONFIG=" + json.dumps(config, ensure_ascii=False, indent=2))
    summaries = [judge_run(args, Path(run_dir)) for run_dir in args.run_dirs]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
