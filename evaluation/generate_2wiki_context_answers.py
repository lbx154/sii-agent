"""Generate direct 2Wiki answers from the provided context without tools."""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.progress import track

from agent.runner import _postprocess_2wiki_evolved_answer
from agent.scoring import score_answer


_LOCAL = threading.local()
_JSON_RE = re.compile(r"\{.*\}", re.S)

SYSTEM_PROMPT = """You answer 2WikiMultihopQA questions using only the provided context.

Resolve the two-hop chain from the context:
1. identify the question entity,
2. identify the intermediate entity if needed,
3. answer the requested attribute/relation/comparison.

Return only JSON: {"final_answer": "concise answer span", "reason": "brief evidence summary"}.

Rules:
- Do not use outside knowledge.
- If the answer is present, copy the shortest context-supported span with the right granularity.
- For yes/no questions, answer exactly "yes" or "no".
- For country/nationality questions, preserve the context wording.
- For dates, preserve the context granularity.
- For places, avoid extra parenthetical/country details unless needed to disambiguate.
"""


def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache_key = (base_url, api_key, timeout)
    cached = getattr(_LOCAL, "client", None)
    if cached is None or getattr(_LOCAL, "client_key", None) != cache_key:
        _LOCAL.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _LOCAL.client_key = cache_key
    return _LOCAL.client


def _json_from_text(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads((text or "").strip())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_source(path: Path, limit: int | None) -> list[dict[str, Any]]:
    source = path / "runs.jsonl" if path.is_dir() else path
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _generate_one(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = str(record.get("question") or "")
    last_error: Exception | None = None
    raw = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = _client(args.base_url, args.api_key, args.timeout).chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            parsed = _json_from_text(raw)
            if parsed is None:
                raise ValueError(f"could not parse context-answer JSON: {raw[:300]}")
            answer = str(parsed.get("final_answer") or "").strip()
            if not answer:
                raise ValueError(f"empty final_answer: {raw[:300]}")
            answer = _postprocess_2wiki_evolved_answer(answer, question)
            scores = score_answer(answer, record.get("expected"))
            out = dict(record)
            out.update(
                {
                    "predicted": answer,
                    "rationale": str(parsed.get("reason") or "")[:1000],
                    "correct": scores["correct"],
                    "exact": scores["exact"],
                    "f1": scores["f1"],
                    "steps": 1,
                    "tool_calls": 0,
                    "tool_call_counts": {},
                    "stop_reason": "context_direct",
                    "finish_reasons": {},
                    "short_memory_stats": {},
                    "elapsed": 0.0,
                    "reflection": None,
                    "context_direct": {"raw": raw[:1000], "error": None},
                }
            )
            return out
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(1.5 * (attempt + 1))
    answer = str(record.get("predicted") or "")
    scores = score_answer(answer, record.get("expected"))
    out = dict(record)
    out.update(
        {
            "predicted": answer,
            "correct": scores["correct"],
            "exact": scores["exact"],
            "f1": scores["f1"],
            "steps": 1,
            "tool_calls": 0,
            "tool_call_counts": {},
            "stop_reason": "context_direct_error",
            "context_direct": {"raw": raw[:1000], "error": f"{type(last_error).__name__}: {last_error}"},
        }
    )
    return out


def generate(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = _load_source(Path(args.source_run), args.limit)
    config = vars(args)
    (out / "context_direct_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(_generate_one, record, args) for record in records]
        for future in track(as_completed(futures), total=len(futures), description="2wiki context-direct"):
            results.append(future.result())

    by_id = {str(record["id"]): record for record in results}
    ordered = [by_id[str(record["id"])] for record in records]
    (out / "runs.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in ordered) + "\n",
        encoding="utf-8",
    )
    n_correct = sum(1 for record in ordered if record.get("correct") is True)
    exact_correct = sum(1 for record in ordered if record.get("exact") is True)
    f1_values = [float(record["f1"]) for record in ordered if isinstance(record.get("f1"), (int, float))]
    summary = {
        "task": "2wiki",
        "mode": "context_direct",
        "n": len(ordered),
        "accuracy": n_correct / len(ordered) if ordered else 0.0,
        "exact_match": exact_correct / len(ordered) if ordered else 0.0,
        "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_run": str(args.source_run),
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "errors": sum(1 for record in ordered if record.get("stop_reason") == "context_direct_error"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8004/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "sii-opd-v13-merged-sglang"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    print("CONTEXT_DIRECT_CONFIG=" + json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(json.dumps(generate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
