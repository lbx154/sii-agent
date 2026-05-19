"""Generate high-quality memory lessons offline from saved run traces.

This script reads benchmark ``runs.jsonl`` files, selects low-quality or
high-value trajectories, asks an OpenAI-compatible model to write reusable
lessons, filters leakage/low-quality outputs, classifies useful lessons by
scope, and optionally appends selected scopes to an active memory directory.
It never runs the agent and never modifies memory unless ``--append-to-memory``
is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


LEAKAGE_RE = re.compile(
    r"\b(gold answer|expected answer|reference answer|answer key|hidden gold|ground truth)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(verify|check|compare|search|use|extract|avoid|do not|don't|prefer|confirm|cross-reference|return)\b",
    re.I,
)

SYSTEM_PROMPT = """You write reusable memory lessons for a target-family QA/search agent.

You receive exactly one saved trajectory. Use the gold answer only to diagnose
the failure offline; do NOT reveal or copy the gold answer into any lesson field.

Write a lesson only if the trajectory teaches a reusable procedural pattern.
Optimize for transfer across the intended target distribution, not for a single
benchmark item. The target distribution is search-heavy factual QA similar to
SimpleVQA and 2Wiki: visual/image QA, OCR/entity identification, web or visual
search, evidence verification, concise answer formatting, multi-hop text QA,
comparison, and relation-chain reasoning.

Do NOT require a lesson to be universal across all possible tasks. A lesson is
useful if it generalizes across this VQA/search + multi-hop-search family. Good
target-family lessons should be active candidates; only reject genuinely unsafe,
leaky, memorized, one-off, or low-quality lessons.

Before returning JSON, perform a strict self-audit:
1. Validity check: the lesson must be actionable, derived from the observed
   failure, non-contradictory, not a vague platitude, and not just "be careful".
2. Generalization check: the lesson must apply to multiple future questions in
   the target family, not to only this entity, image, passage, answer, or one
   exact query.
If either check fails, set decision to reject_bad.

Use one of these decisions:
- keep_general: reusable across both visual/search QA and text multi-hop/search QA; strong active-memory candidate.
- keep_simplevqa: reusable for SimpleVQA-like visual QA/search/OCR/image-evidence tasks; active for the target VQA-search family.
- keep_task_specific: reusable for 2Wiki-like multi-hop text QA, comparison, relation chains, or search reasoning; active for the target multi-hop-search family.
- reject_bad: answer leakage, benchmark/gold shortcut, one-off fact, exact answer memorization, entity-specific trivia, duplicate/low-information, or not actionable.

Return STRICT JSON only:
{
  "decision": "keep_simplevqa|keep_general|keep_task_specific|reject_bad",
  "failure_mode": "short_snake_case",
  "root_cause": "1-2 sentences, no gold answer",
  "corrective_strategy": "actionable future procedure, no gold answer",
  "reusable_lesson": "<=240 chars, generic and actionable, no gold answer",
  "category": "global_verification|query_reformulation|evidence_triage|format|multilingual|tool_recovery|generic_reasoning|ocr|visual_search|entity_identification|multihop_text|comparison|browser_navigation|task_specific|reject",
  "task_family": "target_general|visual_qa|vqa_search|multihop_text|web_research|ocr|comparison|relation_chain|other|reject",
  "validity_check": "pass|fail",
  "generalization_check": "pass|fail",
  "reason": "brief reason for the decision",
  "confidence": "high|medium|low"
}
"""

KEEP_DECISIONS = {"keep_simplevqa", "keep_general", "keep_task_specific"}
DEFAULT_APPEND_DECISIONS = ("keep_general", "keep_simplevqa", "keep_task_specific")
PASS_VALUES = {"pass", "passed", "true", "yes", "ok"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    source = path / "runs.jsonl" if path.is_dir() else path
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
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


def original_question(text: str) -> str:
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*?)(?:\n\s*\n|$)", text or "", flags=re.I | re.S)
    value = match.group(1) if match else text or ""
    return " ".join(value.split())[:800]


def clip(text: object, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def summarize_trajectory(record: dict[str, Any], max_events: int = 18) -> list[dict[str, Any]]:
    traj = record.get("trajectory") or []
    selected: list[dict[str, Any]] = []
    if len(traj) <= max_events:
        events = traj
    else:
        head = traj[:6]
        tail = traj[-(max_events - len(head)) :]
        events = [*head, {"role": "system", "name": "omitted", "content": f"omitted {len(traj) - len(head) - len(tail)} middle events"}, *tail]
    for event in events:
        if not isinstance(event, dict):
            continue
        item: dict[str, Any] = {"role": event.get("role")}
        if event.get("name"):
            item["name"] = event.get("name")
        if event.get("args"):
            item["args"] = event.get("args")
        if event.get("tool_calls"):
            item["tool_calls"] = event.get("tool_calls")
        if event.get("reasoning_content"):
            item["reasoning"] = clip(event.get("reasoning_content"), 700)
        if event.get("content"):
            item["content"] = clip(event.get("content"), 1200)
        selected.append(item)
    return selected


def is_low_confidence_answer(answer: object) -> bool:
    text = str(answer or "").lower()
    return any(term in text for term in ("unknown", "not sure", "cannot determine", "无法确定", "不确定", "可能"))


def select_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        f1 = row.get("f1")
        exact = row.get("exact")
        correct = row.get("correct")
        steps = int(row.get("steps") or 0)
        tool_calls = int(row.get("tool_calls") or 0)
        low_f1 = isinstance(f1, (int, float)) and float(f1) < args.max_f1
        hard = (
            correct is False
            or exact is False
            or low_f1
            or row.get("stop_reason") != "final"
            or is_low_confidence_answer(row.get("predicted"))
            or steps >= args.min_steps
            or tool_calls >= args.min_tool_calls
        )
        if hard:
            selected.append(row)
        if args.limit and len(selected) >= args.limit:
            break
    return selected


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


def normalize_key(text: object) -> str:
    value = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value).strip()


def stable_id(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "question": row.get("question"),
            "failure_mode": row.get("failure_mode"),
            "corrective_strategy": row.get("corrective_strategy"),
            "reusable_lesson": row.get("reusable_lesson"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "offline_lesson_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def normalize_decision(candidate: dict[str, Any]) -> str:
    decision = str(candidate.get("decision") or "").strip().lower()
    if decision in KEEP_DECISIONS or decision == "reject_bad":
        return decision
    keep = str(candidate.get("keep")).strip().lower()
    if keep in {"true", "1", "yes", "keep"}:
        return "keep_general"
    return "reject_bad"


def audit_pass(value: object) -> bool:
    return str(value or "").strip().lower() in PASS_VALUES


def leakage_or_quality_reasons(candidate: dict[str, Any], expected: object, decision: str) -> list[str]:
    text = " ".join(str(candidate.get(k) or "") for k in ("failure_mode", "root_cause", "corrective_strategy", "reusable_lesson", "reason"))
    reasons: list[str] = []
    if LEAKAGE_RE.search(text):
        reasons.append("gold_artifact")
    exp = str(expected or "").strip()
    if len(exp) >= 3 and re.search(re.escape(exp), text, flags=re.I):
        reasons.append("copies_expected_answer")
    if len(str(candidate.get("reusable_lesson") or "").strip()) < 40:
        reasons.append("lesson_too_short")
    if not ACTION_RE.search(text):
        reasons.append("not_actionable")
    if decision in KEEP_DECISIONS and not audit_pass(candidate.get("validity_check")):
        reasons.append("validity_check_failed")
    if decision in KEEP_DECISIONS and not audit_pass(candidate.get("generalization_check")):
        reasons.append("generalization_check_failed")
    if decision == "reject_bad":
        reasons.append("model_rejected")
    return reasons


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "question": original_question(str(record.get("question") or "")),
        "gold_answer_for_diagnosis_only": record.get("expected"),
        "predicted_answer": record.get("predicted"),
        "local_exact": record.get("exact"),
        "local_f1": record.get("f1"),
        "local_correct": record.get("correct"),
        "stop_reason": record.get("stop_reason"),
        "steps": record.get("steps"),
        "tool_calls": record.get("tool_calls"),
        "tool_call_counts": record.get("tool_call_counts"),
        "trajectory": summarize_trajectory(record),
    }


def call_model(client: Any, model: str, record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, str | None]:
    user = json.dumps(build_payload(record), ensure_ascii=False, indent=2)
    last_error: BaseException | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
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
            return parsed, raw, None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(1.5 * (attempt + 1))
    return None, "", f"{type(last_error).__name__}: {last_error}"


def existing_lesson_keys(memory_root: Path) -> set[str]:
    keys: set[str] = set()
    path = memory_root / "lessons.jsonl"
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        keys.add(normalize_key(f"{row.get('failure_mode')} {row.get('corrective_strategy')} {row.get('reusable_lesson')}"))
    return keys


def lesson_row_from_decision(item: dict[str, Any], source_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else None
    if item.get("decision") not in KEEP_DECISIONS or item.get("reasons") or not candidate:
        return None
    source_id = str(item.get("source_id") or "")
    source_record = source_by_id.get(source_id, {})
    row = {
        "ts": time.time(),
        "question": "Question pattern: " + original_question(str(source_record.get("question") or source_id)),
        "failure_mode": clip(candidate.get("failure_mode"), 120) or "target_family_qa_failure",
        "root_cause": clip(candidate.get("root_cause"), 600),
        "corrective_strategy": clip(candidate.get("corrective_strategy"), 700),
        "reusable_lesson": clip(candidate.get("reusable_lesson"), 360),
        "outcome": "offline_filtered_failure_lesson",
        "score": 0.85,
        "tags": [
            "offline",
            str(item.get("decision")),
            str(candidate.get("category") or "generic_reasoning"),
            str(candidate.get("task_family") or "target_general"),
        ],
        "source_run_id": source_id,
        "source_train_index": item.get("source_train_index"),
        "filter_reason": clip(candidate.get("reason"), 300),
        "filter_confidence": clip(candidate.get("confidence"), 40),
        "validity_check": clip(candidate.get("validity_check"), 40),
        "generalization_check": clip(candidate.get("generalization_check"), 40),
        "memory_scope": str(item.get("decision")),
        "task_family": clip(candidate.get("task_family"), 80) or "target_general",
    }
    if item.get("decision") == "keep_simplevqa":
        row["tags"].append("simplevqa")
    elif item.get("decision") == "keep_general":
        row["tags"].append("general")
    else:
        row["tags"].append("task_specific")
    row["id"] = stable_id(row)
    return row


def split_lesson_outputs(
    ordered: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    append_decisions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    simplevqa: list[dict[str, Any]] = []
    general: list[dict[str, Any]] = []
    task_specific: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in ordered:
        row = lesson_row_from_decision(item, source_by_id)
        if row is None:
            rejected.append(item)
            continue
        if item.get("decision") == "keep_simplevqa":
            simplevqa.append(row)
        elif item.get("decision") == "keep_general":
            general.append(row)
        else:
            task_specific.append(row)
        if str(item.get("decision")) in append_decisions:
            active.append(row)
    return active, simplevqa, general, task_specific, rejected


def write_memory_checkpoint(
    checkpoint_dir: Path,
    active: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    base_memory_root: Path,
    include_base_memory: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lessons = active
    if include_base_memory:
        base_lessons = load_jsonl(base_memory_root / "lessons.jsonl") if (base_memory_root / "lessons.jsonl").exists() else []
        lessons = [*base_lessons, *active]
        for name in ("skills.jsonl", "episodes.jsonl", "policies.jsonl"):
            source = base_memory_root / name
            if source.exists():
                (checkpoint_dir / name).write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    write_jsonl(checkpoint_dir / "lessons.jsonl", lessons)
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "readable_memory_root": str(checkpoint_dir),
                "lessons_jsonl": str(checkpoint_dir / "lessons.jsonl"),
                "active_lessons": len(active),
                "total_lessons": len(lessons),
                "include_base_memory": include_base_memory,
                "base_memory_root": str(base_memory_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run dirs or runs.jsonl files to mine.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--memory-root", default="logs/memory")
    parser.add_argument("--append-to-memory", action="store_true")
    parser.add_argument(
        "--append-decisions",
        default=",".join(DEFAULT_APPEND_DECISIONS),
        help="Comma-separated decisions to append to target active memory. Defaults to all target-family keep decisions.",
    )
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8004/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-f1", type=float, default=0.9)
    parser.add_argument("--min-steps", type=int, default=12)
    parser.add_argument("--min-tool-calls", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Write readable memory checkpoints every N source training records.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for readable memory checkpoints. Defaults to <out-dir>/memory_checkpoints.")
    parser.add_argument("--checkpoint-include-base-memory", action="store_true", help="Include existing memory-root JSONL files in each checkpoint.")
    args = parser.parse_args()

    from openai import OpenAI

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "offline_memory_decisions.jsonl"
    active_path = out_dir / "offline_memory_lessons_active.jsonl"
    simplevqa_path = out_dir / "offline_memory_lessons_simplevqa.jsonl"
    general_path = out_dir / "offline_memory_lessons_general.jsonl"
    task_specific_path = out_dir / "offline_memory_lessons_task_specific.jsonl"
    rejected_path = out_dir / "offline_memory_lessons_rejected.jsonl"
    report_path = out_dir / "offline_memory_report.json"
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "memory_checkpoints"
    memory_root = Path(args.memory_root)
    append_decisions = {
        item.strip()
        for item in str(args.append_decisions or "").split(",")
        if item.strip()
    }

    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        for row in load_jsonl(Path(run_dir)):
            copied = dict(row)
            copied["_offline_source_train_index"] = len(rows) + 1
            rows.append(copied)
    selected = select_records(rows, args)

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[str(item.get("source_id"))] = item

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    prior_keys = existing_lesson_keys(memory_root)
    decisions: list[dict[str, Any]] = list(existing.values())
    todo = [row for row in selected if str(row.get("id")) not in existing]
    source_by_id = {str(row.get("id")): row for row in selected}
    selected_by_source_id = {str(row.get("id")): row for row in selected}
    checkpoint_thresholds = (
        list(range(args.checkpoint_every, len(rows) + 1, args.checkpoint_every))
        if args.checkpoint_every and args.checkpoint_every > 0
        else []
    )
    written_checkpoints = {
        int(path.name.rsplit("_", 1)[-1])
        for path in checkpoint_root.glob("memory_after_*")
        if path.is_dir() and path.name.rsplit("_", 1)[-1].isdigit()
    } if checkpoint_thresholds and checkpoint_root.exists() else set()

    print(
        json.dumps(
            {
                "input_records": len(rows),
                "selected_records": len(selected),
                "todo": len(todo),
                "out_dir": str(out_dir),
                "memory_root": str(memory_root),
                "append_to_memory": args.append_to_memory,
                "append_decisions": sorted(append_decisions),
                "model": args.model,
                "base_url": args.base_url,
                "checkpoint_every": args.checkpoint_every,
                "checkpoint_dir": str(checkpoint_root),
                "checkpoint_include_base_memory": args.checkpoint_include_base_memory,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if todo:
        with decisions_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                futures = {pool.submit(call_model, client, args.model, row, args): row for row in todo}
                completed = 0
                for future in as_completed(futures):
                    row = futures[future]
                    candidate, raw, error = future.result()
                    decision = {
                        "source_id": row.get("id"),
                        "source_correct": row.get("correct"),
                        "source_exact": row.get("exact"),
                        "source_f1": row.get("f1"),
                        "source_train_index": row.get("_offline_source_train_index"),
                        "expected_redacted": bool(row.get("expected")),
                        "raw": raw[:2000],
                        "error": error,
                    }
                    if candidate is None:
                        decision.update({"decision": "reject_bad", "append_to_active": False, "reasons": ["model_error"], "candidate": None})
                    else:
                        scope_decision = normalize_decision(candidate)
                        reasons = leakage_or_quality_reasons(candidate, row.get("expected"), scope_decision)
                        key = normalize_key(f"{candidate.get('failure_mode')} {candidate.get('corrective_strategy')} {candidate.get('reusable_lesson')}")
                        if key in prior_keys:
                            reasons.append("duplicate_existing_memory")
                        append_to_active = not reasons and scope_decision in append_decisions
                        decision.update(
                            {
                                "decision": scope_decision,
                                "append_to_active": append_to_active,
                                "reasons": reasons,
                                "candidate": candidate,
                            }
                        )
                        if not reasons:
                            prior_keys.add(key)
                    handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
                    handle.flush()
                    decisions.append(decision)
                    completed += 1
                    if completed % 10 == 0 or completed == len(todo):
                        active_so_far = sum(1 for item in decisions if item.get("append_to_active"))
                        archived_so_far = sum(1 for item in decisions if item.get("decision") in KEEP_DECISIONS and not item.get("reasons"))
                        print(
                            f"offline_memory_progress {completed}/{len(todo)} "
                            f"active_total={active_so_far} archived_good_total={archived_so_far}",
                            flush=True,
                        )
                    if checkpoint_thresholds:
                        completed_ids = {str(item.get("source_id")) for item in decisions}
                        by_id_now = {str(item.get("source_id")): item for item in decisions}
                        for threshold in checkpoint_thresholds:
                            if threshold in written_checkpoints:
                                continue
                            required_ids = {
                                source_id
                                for source_id, source_row in selected_by_source_id.items()
                                if int(source_row.get("_offline_source_train_index") or 0) <= threshold
                            }
                            if not required_ids.issubset(completed_ids):
                                continue
                            ordered_now = [
                                by_id_now[str(row.get("id"))]
                                for row in selected
                                if int(row.get("_offline_source_train_index") or 0) <= threshold
                                and str(row.get("id")) in by_id_now
                            ]
                            active_now, _, _, _, _ = split_lesson_outputs(ordered_now, source_by_id, append_decisions)
                            checkpoint_dir = checkpoint_root / f"memory_after_{threshold:04d}"
                            write_memory_checkpoint(
                                checkpoint_dir,
                                active_now,
                                {
                                    "source_records_seen": threshold,
                                    "selected_records_seen": len(ordered_now),
                                    "append_decisions": sorted(append_decisions),
                                    "run_dirs": args.run_dirs,
                                },
                                base_memory_root=memory_root,
                                include_base_memory=args.checkpoint_include_base_memory,
                            )
                            written_checkpoints.add(threshold)
                            print(f"memory_checkpoint_written threshold={threshold} path={checkpoint_dir}", flush=True)

    by_id = {str(item.get("source_id")): item for item in decisions}
    ordered = [by_id[str(row.get("id"))] for row in selected if str(row.get("id")) in by_id]

    active, simplevqa, general, task_specific, rejected = split_lesson_outputs(ordered, source_by_id, append_decisions)
    if checkpoint_thresholds:
        by_id_all = {str(item.get("source_id")): item for item in ordered}
        completed_ids = set(by_id_all)
        for threshold in checkpoint_thresholds:
            if threshold in written_checkpoints:
                continue
            required_ids = {
                source_id
                for source_id, source_row in selected_by_source_id.items()
                if int(source_row.get("_offline_source_train_index") or 0) <= threshold
            }
            if not required_ids.issubset(completed_ids):
                continue
            ordered_now = [
                by_id_all[str(row.get("id"))]
                for row in selected
                if int(row.get("_offline_source_train_index") or 0) <= threshold
                and str(row.get("id")) in by_id_all
            ]
            active_now, _, _, _, _ = split_lesson_outputs(ordered_now, source_by_id, append_decisions)
            checkpoint_dir = checkpoint_root / f"memory_after_{threshold:04d}"
            write_memory_checkpoint(
                checkpoint_dir,
                active_now,
                {
                    "source_records_seen": threshold,
                    "selected_records_seen": len(ordered_now),
                    "append_decisions": sorted(append_decisions),
                    "run_dirs": args.run_dirs,
                },
                base_memory_root=memory_root,
                include_base_memory=args.checkpoint_include_base_memory,
            )
            written_checkpoints.add(threshold)

    write_jsonl(active_path, active)
    write_jsonl(simplevqa_path, simplevqa)
    write_jsonl(general_path, general)
    write_jsonl(task_specific_path, task_specific)
    write_jsonl(rejected_path, rejected)
    if args.append_to_memory and active:
        append_jsonl(memory_root / "lessons.jsonl", active)

    reason_counts: dict[str, int] = {}
    for item in ordered:
        for reason in item.get("reasons") or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dirs": args.run_dirs,
        "input_records": len(rows),
        "selected_records": len(selected),
        "decisions": len(ordered),
        "active_rows": len(active),
        "simplevqa_rows": len(simplevqa),
        "general_rows": len(general),
        "task_specific_rows": len(task_specific),
        "rejected_rows": len(rejected),
        "reason_counts": reason_counts,
        "decision_counts": {
            decision: sum(1 for item in ordered if item.get("decision") == decision)
            for decision in sorted({str(item.get("decision")) for item in ordered})
        },
        "memory_root": str(memory_root),
        "appended_to_memory": bool(args.append_to_memory),
        "append_decisions": sorted(append_decisions),
        "active_path": str(active_path),
        "simplevqa_path": str(simplevqa_path),
        "general_path": str(general_path),
        "task_specific_path": str(task_specific_path),
        "rejected_path": str(rejected_path),
        "decisions_path": str(decisions_path),
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_dir": str(checkpoint_root),
        "checkpoint_include_base_memory": args.checkpoint_include_base_memory,
        "written_checkpoints": sorted(written_checkpoints),
        "generalization_policy": "Target-family active memory defaults to keep_general, keep_simplevqa, and keep_task_specific for SimpleVQA-like VQA search plus 2Wiki-like multi-hop search tasks.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
