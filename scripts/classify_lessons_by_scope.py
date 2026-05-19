"""Classify existing lessons by reuse scope without overwriting the source file."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You classify exactly ONE existing memory lesson.

Do not rewrite the lesson. Decide how it should be routed for future use.

Important principle: active memory must generalize across the intended target
distribution, not across every possible task. The target distribution is
search-heavy factual QA similar to SimpleVQA and 2Wiki: visual/image QA,
OCR/entity identification, visual or web search, evidence verification,
multi-hop text QA, comparison, relation-chain reasoning, and concise answer
formatting.

Do NOT require a lesson to be universal. Good lessons for SimpleVQA-like visual
search or 2Wiki-like multi-hop search should be preserved as active candidates.
Only reject lessons that are unsafe, leaky, memorized, one-off, contradictory,
not actionable, or outside the target family with no reusable value.

Before returning JSON, perform a strict self-audit:
1. Validity check: the lesson must be actionable, non-contradictory, not a vague
   platitude, not just "be careful", and usable as future behavior guidance.
2. Generalization check: the lesson must apply to multiple future questions in
   the target family, not to only this entity, passage, image, answer, or exact
   query.
If either check fails, set decision to reject_bad.

Use one decision:
- keep_general: reusable across both visual/search QA and text multi-hop/search QA, such as verification, query reformulation, tool recovery, evidence triage, or answer formatting.
- keep_simplevqa: reusable for SimpleVQA-like visual QA/search/OCR/image-evidence tasks.
- keep_task_specific: reusable for 2Wiki-like multi-hop text QA, comparison, family-relation chains, or search reasoning.
- reject_bad: answer leakage, benchmark/gold shortcut, exact answer memorization, entity-specific trivia, duplicate/low-information, contradictory, or not actionable.

Return JSON only:
{
  "decision": "keep_simplevqa|keep_general|keep_task_specific|reject_bad",
  "category": "global_verification|query_reformulation|evidence_triage|format|multilingual|tool_recovery|generic_reasoning|ocr|visual_search|entity_identification|multihop_text|comparison|family_relation|browser_navigation|task_specific|reject",
  "task_family": "target_general|visual_qa|vqa_search|multihop_text|web_research|ocr|comparison|relation_chain|other|reject",
  "validity_check": "pass|fail",
  "generalization_check": "pass|fail",
  "reason": "brief",
  "confidence": "high|medium|low"
}
"""

VALID_DECISIONS = {"keep_simplevqa", "keep_general", "keep_task_specific", "reject_bad"}
DEFAULT_APPEND_DECISIONS = ("keep_general", "keep_simplevqa", "keep_task_specific")
PASS_VALUES = {"pass", "passed", "true", "yes", "ok"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_obj(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def normalize_decision(value: object) -> str:
    decision = str(value or "").strip().lower()
    return decision if decision in VALID_DECISIONS else "reject_bad"


def audit_pass(value: object) -> bool:
    return str(value or "").strip().lower() in PASS_VALUES


def lesson_payload(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": row.get("id"),
        "question_pattern": row.get("question"),
        "failure_mode": row.get("failure_mode"),
        "root_cause": row.get("root_cause"),
        "corrective_strategy": row.get("corrective_strategy"),
        "reusable_lesson": row.get("reusable_lesson"),
        "tags": row.get("tags"),
        "outcome": row.get("outcome"),
        "score": row.get("score"),
    }


def call_model(client: Any, args: argparse.Namespace, row: dict[str, Any], index: int) -> dict[str, Any]:
    user = json.dumps(lesson_payload(row, index), ensure_ascii=False, indent=2)
    last_error: BaseException | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content or ""
            parsed = parse_json_obj(raw)
            if parsed is None:
                raise ValueError(f"could not parse JSON: {raw[:300]}")
            decision = normalize_decision(parsed.get("decision"))
            validity_check = str(parsed.get("validity_check") or "").strip().lower()
            generalization_check = str(parsed.get("generalization_check") or "").strip().lower()
            audit_reasons: list[str] = []
            if decision in {"keep_simplevqa", "keep_general", "keep_task_specific"} and not audit_pass(validity_check):
                audit_reasons.append("validity_check_failed")
            if decision in {"keep_simplevqa", "keep_general", "keep_task_specific"} and not audit_pass(generalization_check):
                audit_reasons.append("generalization_check_failed")
            if audit_reasons:
                decision = "reject_bad"
            return {
                "index": index,
                "id": row.get("id"),
                "decision": decision,
                "category": str(parsed.get("category") or "reject")[:100],
                "task_family": str(parsed.get("task_family") or "reject")[:100],
                "validity_check": validity_check,
                "generalization_check": generalization_check,
                "audit_reasons": audit_reasons,
                "reason": str(parsed.get("reason") or "")[:600],
                "confidence": str(parsed.get("confidence") or "")[:40],
                "raw": raw[:1200],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    return {
        "index": index,
        "id": row.get("id"),
        "decision": "reject_bad",
        "category": "reject",
        "task_family": "reject",
        "validity_check": "fail",
        "generalization_check": "fail",
        "audit_reasons": ["judge_error"],
        "reason": f"judge_error: {type(last_error).__name__}: {last_error}",
        "confidence": "low",
        "raw": "",
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def env_from_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    allowed = {"VLLM_BASE_URL", "VLLM_MODEL", "VLLM_API_KEY", "OPENAI_API_KEY"}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key not in allowed or os.getenv(key):
            continue
        value = value.strip()
        if value:
            try:
                parts = shlex.split(value, posix=True)
                value = parts[0] if parts else ""
            except ValueError:
                value = value.strip("\"'")
        os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--memory-root", default="logs/memory")
    parser.add_argument("--append-to-memory", action="store_true")
    parser.add_argument("--append-decisions", default=",".join(DEFAULT_APPEND_DECISIONS))
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8004/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-dotenv", action="store_true")
    args = parser.parse_args()

    if args.load_dotenv:
        env_from_dotenv()
        args.base_url = os.getenv("VLLM_BASE_URL", args.base_url)
        args.model = os.getenv("VLLM_MODEL", args.model)
        args.api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or args.api_key

    from openai import OpenAI

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(source)
    if args.limit is not None:
        rows = rows[: args.limit]

    decisions_path = out_dir / "lesson_scope_decisions.jsonl"
    simplevqa_path = out_dir / "lessons_keep_simplevqa.jsonl"
    general_path = out_dir / "lessons_keep_general.jsonl"
    task_specific_path = out_dir / "lessons_keep_task_specific.jsonl"
    active_path = out_dir / "lessons_active_append_candidates.jsonl"
    rejected_path = out_dir / "lessons_reject_bad.jsonl"
    report_path = out_dir / "lesson_scope_report.json"
    append_decisions = {item.strip() for item in args.append_decisions.split(",") if item.strip()}

    existing: dict[int, dict[str, Any]] = {}
    if args.resume and decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[int(item["index"])] = item

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    decisions: list[dict[str, Any]] = list(existing.values())
    todo = [(idx, row) for idx, row in enumerate(rows, 1) if idx not in existing]

    print(
        json.dumps(
            {
                "source": str(source),
                "out_dir": str(out_dir),
                "input_rows": len(rows),
                "todo": len(todo),
                "model": args.model,
                "base_url": args.base_url,
                "one_lesson_per_call": True,
                "concurrency": args.concurrency,
                "append_to_memory": args.append_to_memory,
                "append_decisions": sorted(append_decisions),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if todo:
        with decisions_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                futures = {pool.submit(call_model, client, args, row, idx): idx for idx, row in todo}
                for count, future in enumerate(as_completed(futures), 1):
                    decision = future.result()
                    handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
                    handle.flush()
                    decisions.append(decision)
                    if count % 25 == 0 or count == len(todo):
                        counts: dict[str, int] = {}
                        for item in decisions:
                            counts[item["decision"]] = counts.get(item["decision"], 0) + 1
                        print(
                            "scope_progress "
                            f"{count}/{len(todo)} "
                            + " ".join(f"{key}={counts.get(key, 0)}" for key in sorted(VALID_DECISIONS)),
                            flush=True,
                        )

    by_index = {int(item["index"]): item for item in decisions}
    buckets = {
        "keep_simplevqa": [],
        "keep_general": [],
        "keep_task_specific": [],
        "reject_bad": [],
    }
    for idx, row in enumerate(rows, 1):
        decision = by_index[idx]
        annotated = dict(row)
        annotated["scope_filter"] = {
            key: decision.get(key)
            for key in ("decision", "category", "task_family", "validity_check", "generalization_check", "audit_reasons", "reason", "confidence", "error")
        }
        buckets[decision["decision"]].append(annotated)

    active = [
        row
        for decision in append_decisions
        for row in buckets.get(decision, [])
        if decision in {"keep_simplevqa", "keep_general", "keep_task_specific"}
    ]
    write_jsonl(simplevqa_path, buckets["keep_simplevqa"])
    write_jsonl(general_path, buckets["keep_general"])
    write_jsonl(task_specific_path, buckets["keep_task_specific"])
    write_jsonl(rejected_path, buckets["reject_bad"])
    write_jsonl(active_path, active)
    if args.append_to_memory and active:
        append_jsonl(Path(args.memory_root) / "lessons.jsonl", active)

    category_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    for item in decisions:
        decision_counts[item["decision"]] = decision_counts.get(item["decision"], 0) + 1
        category = str(item.get("category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source),
        "out_dir": str(out_dir),
        "input_rows": len(rows),
        "decision_counts": decision_counts,
        "category_counts": category_counts,
        "judge_errors": sum(1 for item in decisions if item.get("error")),
        "append_to_memory": args.append_to_memory,
        "append_decisions": sorted(append_decisions),
        "active_rows": len(active),
        "paths": {
            "decisions": str(decisions_path),
            "simplevqa": str(simplevqa_path),
            "general": str(general_path),
            "task_specific": str(task_specific_path),
            "active": str(active_path),
            "rejected": str(rejected_path),
        },
        "generalization_policy": "Target-family active memory defaults to keep_general, keep_simplevqa, and keep_task_specific for SimpleVQA-like VQA search plus 2Wiki-like multi-hop search tasks.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
