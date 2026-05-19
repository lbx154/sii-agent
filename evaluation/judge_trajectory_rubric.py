"""Score agent trajectories with an OpenAI-compatible GRM/Judge and rubric."""
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


RUBRIC_PROMPT = """You are a strict GRM/rubric judge for tool-using QA agent trajectories.

You must score both the final answer and the process quality. The goal is to decide whether
this trajectory is safe to use as a positive SFT trajectory for an agent. Do NOT reward a
trajectory merely because the answer looks plausible; require evidence support and good tool
behavior.

Scoring rubric, total 10 points:
- answer_correctness: 0-3. Gold may be provided. If gold is present, judge semantic equivalence
  to the gold answer. If gold is absent, judge whether the answer actually answers the question.
- evidence_support: 0-2. The final answer must be supported by provided context, visual/OCR
  evidence, search results, browsed pages, or tool observations in the trajectory.
- task_procedure: 0-1.5. For 2Wiki, prefer the provided context and solve the two-hop chain
  before broad external search. For SimpleVQA, use visual/OCR/image/web verification before
  final answer.
- tool_efficiency: 0-1. Use focused, necessary tool calls; penalize repeated queries, loops,
  broad over-search, wrong tools, and unnecessary browsing.
- stop_quality: 0-1. Stop with final_answer once evidence is sufficient; penalize premature
  final answers, max-step endings, no-tool loops, or continuing to search after enough evidence.
- format_quality: 0-1.5. Final answer should be concise and dataset-formatted: no explanation,
  yes/no exactly when needed, correct date/place/entity granularity, no multiple candidates.

Bad pattern labels you may use:
unsupported_final, wrong_visual_entity, tool_loop, repeated_query, premature_final,
search_before_context, no_visual_verification, verbose_or_wrong_format_answer, wrong_tool,
over_search, context_ignored, evidence_contradiction.

Critical bad patterns that should usually prevent positive SFT use:
unsupported_final, wrong_visual_entity, tool_loop, evidence_contradiction,
search_before_context, no_visual_verification.

Return STRICT JSON only, with this schema:
{
  "score": 0.0,
  "success": true,
  "confidence": "high|medium|low",
  "answer_correctness": 0.0,
  "evidence_support": 0.0,
  "task_procedure": 0.0,
  "tool_efficiency": 0.0,
  "stop_quality": 0.0,
  "format_quality": 0.0,
  "bad_patterns": ["..."],
  "failure_mode": "none|...",
  "feedback": "brief actionable feedback"
}
No markdown, no prose outside JSON.
"""

CRITICAL_BAD_PATTERNS = {
    "unsupported_final",
    "wrong_visual_entity",
    "tool_loop",
    "evidence_contradiction",
    "search_before_context",
    "no_visual_verification",
}

BAD_PATTERN_ALIASES = {
    "repeated query": "repeated_query",
    "tool loop": "tool_loop",
    "premature final": "premature_final",
    "unsupported final": "unsupported_final",
    "over search": "over_search",
    "over-search": "over_search",
    "wrong tool": "wrong_tool",
    "wrong visual entity": "wrong_visual_entity",
    "no visual verification": "no_visual_verification",
    "verbose answer": "verbose_or_wrong_format_answer",
    "wrong format": "verbose_or_wrong_format_answer",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_run(path: Path) -> list[dict[str, Any]]:
    runs_jsonl = path / "runs.jsonl" if path.is_dir() else path
    return _read_jsonl(runs_jsonl)


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for item in _read_jsonl(path):
        if item.get("id") is not None:
            existing[str(item["id"])] = item
    return existing


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


def _clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _original_question(question: str) -> str:
    if "Question: " in question:
        return question.split("Question: ", 1)[1].split("\n", 1)[0].strip()
    if "Provided context:" in question:
        return question.split("Provided context:", 1)[0].strip()
    return question.strip()


def _tool_name(event: dict[str, Any]) -> str:
    return str(event.get("name") or "")


def _tool_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args")
    return args if isinstance(args, dict) else {}


def _assistant_tool_names(event: dict[str, Any]) -> list[str]:
    names = []
    for call in event.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        if name:
            names.append(str(name))
    return names


def _first_tool_name(record: dict[str, Any]) -> str:
    for event in record.get("trajectory") or []:
        if event.get("role") == "assistant":
            names = _assistant_tool_names(event)
            if names:
                return names[0]
    return ""


def _rule_bad_patterns(record: dict[str, Any]) -> list[str]:
    task = str(record.get("task") or "").lower()
    trajectory = record.get("trajectory") or []
    patterns: set[str] = set()
    seen_calls: dict[str, int] = {}
    search_calls = 0
    visual_calls = 0

    for event in trajectory:
        if event.get("role") != "tool":
            continue
        name = _tool_name(event)
        args = _tool_args(event)
        if name in {"web_search", "wiki_search", "wiki_page", "browse", "browse_many", "image_search"}:
            key = json.dumps({"name": name, "args": args}, sort_keys=True, ensure_ascii=False)
            seen_calls[key] = seen_calls.get(key, 0) + 1
            if seen_calls[key] > 1:
                patterns.add("repeated_query")
        if name in {"web_search", "wiki_search", "wiki_page", "browse", "browse_many", "image_search"}:
            search_calls += 1
        if name in {"visual_web_search", "image_to_text", "image_search"}:
            visual_calls += 1

    stop_reason = str(record.get("stop_reason") or "")
    if stop_reason.startswith("loop"):
        patterns.add("tool_loop")
    if stop_reason in {"max_steps", "timeout"}:
        patterns.add("tool_loop")
    if task == "2wiki" and search_calls > 4:
        patterns.add("over_search")
    if task in {"simplevqa", "simple-vqa"} and visual_calls == 0:
        patterns.add("no_visual_verification")

    answer = str(record.get("predicted") or "").strip()
    if not answer:
        patterns.add("premature_final")
    if len(answer) > 120 or "\n" in answer:
        patterns.add("verbose_or_wrong_format_answer")
    if task == "2wiki" and _first_tool_name(record) in {"web_search", "wiki_search"}:
        patterns.add("search_before_context")
    return sorted(patterns)


def _trajectory_summary(record: dict[str, Any], max_events: int = 18) -> list[dict[str, Any]]:
    events = record.get("trajectory") or []
    if len(events) > max_events:
        head_count = max_events // 2
        tail_count = max_events - head_count
        selected = events[:head_count] + [{"role": "omitted", "count": len(events) - max_events}] + events[-tail_count:]
    else:
        selected = events

    summary = []
    for event in selected:
        role = event.get("role")
        if role == "assistant":
            summary.append(
                {
                    "role": "assistant",
                    "content": _clip(event.get("content"), 1000),
                    "tool_calls": event.get("tool_calls") or [],
                    "reasoning_content": _clip(event.get("reasoning_content"), 1200),
                }
            )
        elif role == "tool":
            summary.append(
                {
                    "role": "tool",
                    "name": event.get("name"),
                    "args": event.get("args"),
                    "content": _clip(event.get("content"), 1600),
                }
            )
        else:
            summary.append(event)
    return summary


def _payload_for_record(record: dict[str, Any], use_gold: bool) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "task": record.get("task"),
        "split": record.get("split"),
        "question": _original_question(str(record.get("question") or "")),
        "full_question_excerpt": _clip(record.get("question"), 5000),
        "gold_answer": record.get("expected") if use_gold else None,
        "predicted_answer": record.get("predicted"),
        "local_exact_or_f1_correct": record.get("correct") if use_gold else None,
        "steps": record.get("steps"),
        "tool_calls": record.get("tool_calls"),
        "tool_call_counts": record.get("tool_call_counts"),
        "stop_reason": record.get("stop_reason"),
        "finish_reasons": record.get("finish_reasons"),
        "rule_bad_pattern_suspicions": _rule_bad_patterns(record),
        "trajectory": _trajectory_summary(record),
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_confidence(value: Any) -> str:
    confidence = str(value or "low").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else "low"


def _normalize_bad_patterns(items: Any) -> list[str]:
    if not isinstance(items, list):
        items = [items] if items else []
    patterns: set[str] = set()
    for item in items:
        raw = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        raw = BAD_PATTERN_ALIASES.get(raw, raw)
        if raw:
            patterns.add(raw)
    return sorted(patterns)


def _normalize_score(parsed: dict[str, Any], rule_patterns: list[str], threshold: float) -> dict[str, Any]:
    answer = max(0.0, min(3.0, _as_float(parsed.get("answer_correctness"))))
    evidence = max(0.0, min(2.0, _as_float(parsed.get("evidence_support"))))
    procedure = max(0.0, min(1.5, _as_float(parsed.get("task_procedure"))))
    efficiency = max(0.0, min(1.0, _as_float(parsed.get("tool_efficiency"))))
    stop = max(0.0, min(1.0, _as_float(parsed.get("stop_quality"))))
    fmt = max(0.0, min(1.5, _as_float(parsed.get("format_quality"))))
    component_sum = answer + evidence + procedure + efficiency + stop + fmt
    score = _as_float(parsed.get("score"), component_sum)
    if abs(score - component_sum) > 1.5:
        score = component_sum
    patterns = sorted(set(_normalize_bad_patterns(parsed.get("bad_patterns"))) | set(rule_patterns))
    has_critical = bool(set(patterns) & CRITICAL_BAD_PATTERNS)
    success = parsed.get("success")
    if isinstance(success, str):
        success = success.strip().lower() in {"true", "yes", "1", "correct", "success"}
    success = bool(success)
    positive_candidate = (
        success
        and score >= threshold
        and answer >= 2.5
        and evidence >= 1.5
        and fmt >= 1.0
        and efficiency >= 0.7
        and not has_critical
    )
    return {
        "score": round(max(0.0, min(10.0, score)), 3),
        "success": success,
        "confidence": _normalize_confidence(parsed.get("confidence")),
        "answer_correctness": answer,
        "evidence_support": evidence,
        "task_procedure": procedure,
        "tool_efficiency": efficiency,
        "stop_quality": stop,
        "format_quality": fmt,
        "bad_patterns": patterns,
        "failure_mode": str(parsed.get("failure_mode") or ("none" if success else "unknown"))[:120],
        "feedback": str(parsed.get("feedback") or "")[:1200],
        "has_critical_bad_pattern": has_critical,
        "positive_candidate": positive_candidate,
    }


def _judge_one(client: Any, args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_for_record(record, use_gold=args.use_gold)
    rule_patterns = list(payload["rule_bad_pattern_suspicions"])
    last_error: BaseException | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": RUBRIC_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            parsed = _json_from_text(raw)
            if parsed is None:
                raise ValueError(f"could not parse judge JSON: {raw[:500]}")
            normalized = _normalize_score(parsed, rule_patterns, args.positive_threshold)
            return {
                "id": record.get("id"),
                "task": record.get("task"),
                "split": record.get("split"),
                "predicted": record.get("predicted"),
                "expected": record.get("expected") if args.use_gold else None,
                "local_correct": record.get("correct") if args.use_gold else None,
                "steps": record.get("steps"),
                "tool_calls": record.get("tool_calls"),
                "tool_call_counts": record.get("tool_call_counts"),
                "stop_reason": record.get("stop_reason"),
                "rule_bad_patterns": rule_patterns,
                **normalized,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(1.5 * (attempt + 1))
    return {
        "id": record.get("id"),
        "task": record.get("task"),
        "split": record.get("split"),
        "predicted": record.get("predicted"),
        "expected": record.get("expected") if args.use_gold else None,
        "local_correct": record.get("correct") if args.use_gold else None,
        "score": None,
        "success": False,
        "confidence": "low",
        "bad_patterns": sorted(set(rule_patterns)),
        "failure_mode": "judge_error",
        "feedback": f"judge_error: {type(last_error).__name__}: {last_error}",
        "has_critical_bad_pattern": bool(set(rule_patterns) & CRITICAL_BAD_PATTERNS),
        "positive_candidate": False,
    }


def _write_summary(out_jsonl: Path, items: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    scored = [item for item in items if isinstance(item.get("score"), (int, float))]
    positives = [item for item in scored if item.get("positive_candidate") is True]
    successes = [item for item in scored if item.get("success") is True]
    critical = [item for item in scored if item.get("has_critical_bad_pattern") is True]
    bad_counts: dict[str, int] = {}
    for item in scored:
        for pattern in item.get("bad_patterns") or []:
            bad_counts[str(pattern)] = bad_counts.get(str(pattern), 0) + 1
    summary = {
        **meta,
        "n": len(items),
        "n_scored": len(scored),
        "n_errors": len(items) - len(scored),
        "avg_score": sum(float(item["score"]) for item in scored) / len(scored) if scored else 0.0,
        "success_count": len(successes),
        "success_rate": len(successes) / len(scored) if scored else 0.0,
        "positive_candidates": len(positives),
        "positive_rate": len(positives) / len(scored) if scored else 0.0,
        "critical_bad_pattern_count": len(critical),
        "bad_pattern_counts": dict(sorted(bad_counts.items())),
    }
    summary_path = out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


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
                for future in track(as_completed(futures), total=len(futures), description=f"rubric {run_path.name}"):
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
        "use_gold": args.use_gold,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "positive_threshold": args.positive_threshold,
    }
    return _write_summary(out_jsonl, ordered, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score agent trajectories with a rubric GRM/Judge.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories or runs.jsonl files.")
    parser.add_argument("--base-url", default=os.getenv("GRM_BASE_URL", "http://127.0.0.1:8004/v1"))
    parser.add_argument("--api-key", default=os.getenv("GRM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("GRM_MODEL", "qwen35-27b-sglang"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-prefix", default="trajectory_rubric_grm")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--positive-threshold", type=float, default=8.0)
    parser.add_argument("--use-gold", action="store_true", help="Include gold answer for train/validation analysis.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = {
        "run_dirs": args.run_dirs,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "out_prefix": args.out_prefix,
        "positive_threshold": args.positive_threshold,
        "use_gold": args.use_gold,
        "resume": args.resume,
    }
    print("TRAJECTORY_RUBRIC_CONFIG=" + json.dumps(config, ensure_ascii=False, indent=2))
    summaries = [judge_run(args, Path(run_dir)) for run_dir in args.run_dirs]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
