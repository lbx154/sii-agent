"""Run the answered browser benchmark with local scoring and submission traces.

The benchmark protocol here is intentionally test-only:
- no reflection/retry runner
- no train-time gold verification
- no memory writes
- baseline has no memory tools
- memory mode can query read-only memory and receives overall guidance
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.react import _overall_memory_guidance, run_react  # noqa: E402
from agent.scoring import score_answer  # noqa: E402
from harness.controller import HarnessConfig  # noqa: E402


csv.field_size_limit(sys.maxsize)

BASELINE_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "final_answer",
)

MEMORY_QUERY_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "memory_search",
    "memory_stats",
    "memory_list",
    "memory_get",
    "final_answer",
)

VISUAL_TOOLS = {
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
}

BASELINE_EXTRA = (
    "Evaluation mode. No memory tools are available in this run. Use live search and browser tools as needed. "
    "For slow pages, browser_open/browser_open_many may use timeout_ms up to 120000. "
    "Cross-check current evidence before final_answer. No reflection retry is available; solve carefully within "
    "this single ReAct attempt. Only use visual/image tools when the prompt provides an actual image file path "
    "or direct image URL; never pass an ordinary web page or search-result URL to image tools."
)

MEMORY_EXTRA = (
    "Evaluation/test mode with read-only memory. You can and should call memory_search early when it may help: "
    "query the full question plus 2-6 focused phrases/entities/patterns. Treat memory_search guidance and overall "
    "memory guidance only as procedural advice, not evidence. Do not create/update/delete memory. Verify the final "
    "answer using current web_search/browser evidence. For slow pages, browser_open/browser_open_many may use "
    "timeout_ms up to 120000. No reflection retry is available; solve carefully within this single ReAct attempt. "
    "Only use visual/image tools when the prompt provides an actual image file path or direct image URL; never pass "
    "an ordinary web page or search-result URL to image tools."
)


def _set_default_env() -> None:
    os.environ["LLM_BACKEND"] = os.getenv("LLM_BACKEND") or "vllm"
    os.environ["VLLM_BASE_URL"] = os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8004/v1"
    if os.environ["VLLM_BASE_URL"].rstrip("/") == "http://127.0.0.1:8000/v1":
        os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8004/v1"
    os.environ["VLLM_MODEL"] = os.getenv("VLLM_MODEL") or "Qwen3.5-9B"
    if os.environ["VLLM_MODEL"] == "Qwen/Qwen3.5-9B":
        os.environ["VLLM_MODEL"] = "Qwen3.5-9B"
    os.environ["VLLM_API_KEY"] = os.getenv("VLLM_API_KEY") or "EMPTY"
    os.environ["VLLM_ENABLE_THINKING"] = os.getenv("VLLM_ENABLE_THINKING") or "0"
    os.environ["SII_AGENT_RUNTIME_MODE"] = os.getenv("SII_AGENT_RUNTIME_MODE") or "test"
    os.environ.setdefault("SII_AGENT_MEMORY_ROOT", "logs/memory")
    os.environ.setdefault("SII_MEMORY_OVERALL_IN_PROMPT", "1")
    os.environ.setdefault("SII_MEMORY_OVERALL_PROMPT_MAX_CHARS", "2200")
    os.environ.setdefault("SII_MEMORY_SEARCH_LLM_SUMMARY", "1")
    os.environ.setdefault("SII_MEMORY_SEARCH_SUMMARY_TIMEOUT", "240")
    os.environ.setdefault("SEARCH_PROXY_TIMEOUT", "300")
    os.environ.setdefault("SEARCH_PROXY_UPLOAD_TIMEOUT", "300")


def _load_rows(csv_path: Path, n: int, offset: int) -> tuple[list[str], list[dict[str, str]], list[tuple[int, dict[str, str]]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        all_rows = [dict(row) for row in reader]
    selected: list[tuple[int, dict[str, str]]] = []
    limit = len(all_rows) if n <= 0 else n
    for idx, row in enumerate(all_rows):
        if idx < offset:
            continue
        if len(selected) >= limit:
            break
        selected.append((idx, row))
    return fieldnames, all_rows, selected


def _build_question(row: dict[str, str]) -> str:
    problem = " ".join(str(row.get("problem") or "").split())
    image = str(row.get("_image_ref") or "").strip()
    parts = [problem]
    if image:
        parts.append(f"Image file/source: {image}")
        parts.append(
            "If the image is needed, call image_to_text, image_to_search_queries, visual_web_search, "
            "or reverse_image_search with this source path/URL. Do not copy raw image data into tool arguments."
        )
    parts.append("Return only the concise answer requested by the problem.")
    return "\n\n".join(parts)


def _guess_image_ext(mime: str | None, data: bytes) -> str:
    if mime:
        guessed = mimetypes.guess_extension(mime.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _decode_image_payload(image: str) -> tuple[bytes, str | None, str]:
    value = "".join(image.strip().split())
    if value.startswith("data:image/"):
        header, sep, encoded = value.partition(",")
        if not sep or not encoded:
            raise ValueError("data image URL is missing a base64 payload")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        return base64.b64decode(encoded, validate=True), mime, "data_url"
    return base64.b64decode(value, validate=True), None, "raw_base64"


def _prepare_image(row: dict[str, str], idx: int, images_dir: Path) -> dict[str, Any]:
    image = str(row.get("image") or "").strip()
    if not image:
        row["_image_ref"] = ""
        row["_image_meta"] = {"kind": "none", "original_chars": 0}
        return row["_image_meta"]

    if image.startswith(("http://", "https://")):
        row["_image_ref"] = image
        row["_image_meta"] = {"kind": "url", "original_chars": len(image), "source": image}
        return row["_image_meta"]

    if len(image) < 4096:
        try:
            image_path = Path(image).expanduser()
            if image_path.exists() and image_path.is_file():
                row["_image_ref"] = str(image_path)
                row["_image_meta"] = {"kind": "path", "original_chars": len(image), "source": str(image_path)}
                return row["_image_meta"]
        except OSError:
            pass

    try:
        data, mime, kind = _decode_image_payload(image)
    except (binascii.Error, ValueError) as exc:
        row["_image_ref"] = ""
        row["_image_meta"] = {
            "kind": "invalid_image_payload",
            "original_chars": len(image),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return row["_image_meta"]

    digest = hashlib.sha1(data).hexdigest()[:12]
    ext = _guess_image_ext(mime, data)
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"benchmark-csv-{idx}-{digest}{ext}"
    if not path.exists():
        path.write_bytes(data)
    row["_image_ref"] = str(path)
    row["_image_meta"] = {
        "kind": kind,
        "original_chars": len(image),
        "bytes": len(data),
        "mime": mime,
        "path": str(path),
    }
    return row["_image_meta"]


def _run_one(
    mode: str,
    tools: tuple[str, ...],
    extra_system: str,
    idx: int,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_tools = tools if row.get("_image_ref") else tuple(name for name in tools if name not in VISUAL_TOOLS)
    row_extra_system = extra_system
    if not row.get("_image_ref"):
        row_extra_system += " This row has no image input; use web_search/browser tools for web pages, not visual/image tools."
    cfg = HarnessConfig(
        max_steps=args.max_steps,
        max_wall_seconds=args.max_wall_seconds,
        max_llm_tokens=args.max_llm_tokens,
        max_llm_call_seconds=args.max_llm_call_seconds,
        min_llm_call_seconds=args.min_llm_call_seconds,
        allowed_tools=row_tools,
    )
    expected = str(row.get("answer") or "").strip()
    started = time.time()
    try:
        result = run_react(
            _build_question(row),
            cfg=cfg,
            extra_system=row_extra_system,
            expected=None,
            task="benchmark_csv",
        )
        scores = score_answer(result.final_answer, expected)
        return {
            "id": f"benchmark-csv-{idx}",
            "index": idx,
            "mode": mode,
            "problem": row.get("problem", ""),
            "image": row.get("_image_ref", ""),
            "image_meta": row.get("_image_meta", {}),
            "answer": result.final_answer or "",
            "expected": expected,
            "correct": bool(scores.get("correct")),
            "exact": bool(scores.get("exact")),
            "f1": float(scores.get("f1") or 0.0),
            "rationale": result.rationale,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "tool_call_counts": result.tool_call_counts,
            "stop_reason": result.stop_reason,
            "finish_reasons": result.finish_reasons,
            "elapsed": result.elapsed,
            "trajectory": result.trajectory if args.save_trace else [],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        scores = score_answer(None, expected)
        return {
            "id": f"benchmark-csv-{idx}",
            "index": idx,
            "mode": mode,
            "problem": row.get("problem", ""),
            "image": row.get("_image_ref", ""),
            "image_meta": row.get("_image_meta", {}),
            "answer": "",
            "expected": expected,
            "correct": bool(scores.get("correct")),
            "exact": bool(scores.get("exact")),
            "f1": float(scores.get("f1") or 0.0),
            "rationale": "",
            "steps": 0,
            "tool_calls": 0,
            "tool_call_counts": {},
            "stop_reason": f"error: {type(exc).__name__}: {exc}",
            "finish_reasons": {},
            "elapsed": time.time() - started,
            "trajectory": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _submission_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["index"],
        "problem": record["problem"],
        "image": record["image"],
        "image_meta": record.get("image_meta", {}),
        "answer": record["answer"],
        "steps": record["steps"],
        "tool_call_counts": record["tool_call_counts"],
        "stop_reason": record["stop_reason"],
        "elapsed": record["elapsed"],
        "trajectory": record["trajectory"],
    }


def _write_mode_outputs(
    run_root: Path,
    mode: str,
    records: list[dict[str, Any]],
    fieldnames: list[str],
    input_rows: list[dict[str, str]],
) -> dict[str, str]:
    mode_root = run_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    group_name = f"group_{mode}"
    trace_path = mode_root / f"{group_name}.json"
    answer_path = mode_root / f"{group_name}.csv"
    zip_path = mode_root / f"{group_name}.zip"

    ordered = sorted(records, key=lambda item: int(item["index"]))
    trace_payload = [_submission_trace_record(record) for record in ordered]
    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    answer_by_index = {int(record["index"]): str(record.get("answer") or "") for record in ordered}
    output_rows: list[dict[str, str]] = []
    for idx, row in enumerate(input_rows):
        output = {field: row.get(field, "") for field in fieldnames}
        output["answer"] = answer_by_index.get(idx, "")
        output_rows.append(output)
    with answer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(trace_path, arcname=trace_path.name)
        zf.write(answer_path, arcname=answer_path.name)
    return {"trace": str(trace_path), "answers_csv": str(answer_path), "zip": str(zip_path)}


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "accuracy": sum(1 for record in records if record["correct"]) / len(records),
        "exact_match": sum(1 for record in records if record["exact"]) / len(records),
        "avg_f1": sum(float(record["f1"]) for record in records) / len(records),
        "avg_steps": sum(int(record["steps"]) for record in records) / len(records),
        "avg_tool_calls": sum(int(record["tool_calls"]) for record in records) / len(records),
        "stop_reasons": dict(Counter(str(record["stop_reason"]) for record in records)),
        "tool_call_counts": dict(sum((Counter(record["tool_call_counts"]) for record in records), Counter())),
        "errors": sum(1 for record in records if record.get("error")),
    }


def _mode_specs(requested: str) -> list[tuple[str, tuple[str, ...], str]]:
    specs = [
        ("baseline_no_memory", BASELINE_TOOLS, BASELINE_EXTRA),
        ("memory_query_only", MEMORY_QUERY_TOOLS, MEMORY_EXTRA),
    ]
    if requested == "both":
        return specs
    return [spec for spec in specs if spec[0] == requested]


def _run_mode(
    run_root: Path,
    mode: str,
    tools: tuple[str, ...],
    extra_system: str,
    selected_rows: list[tuple[int, dict[str, str]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jsonl_path = run_root / f"{mode}.jsonl"
    existing_records: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("index"), int):
                existing_records.append(record)
                completed_indices.add(int(record["index"]))
    remaining_rows = [(idx, row) for idx, row in selected_rows if idx not in completed_indices]
    print(
        f"START_MODE {mode} rows={len(selected_rows)} completed={len(completed_indices)} "
        f"remaining={len(remaining_rows)} concurrency={args.concurrency}",
        flush=True,
    )
    records: list[dict[str, Any]] = list(existing_records)
    lock = threading.Lock()
    file_mode = "a" if args.resume and jsonl_path.exists() else "w"
    with jsonl_path.open(file_mode, encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_run_one, mode, tools, extra_system, idx, row, args): idx
            for idx, row in remaining_rows
        }
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            records.append(record)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "done": done,
                        "index": record["index"],
                        "predicted": record["answer"],
                        "expected": record["expected"],
                        "correct": record["correct"],
                        "f1": record["f1"],
                        "steps": record["steps"],
                        "stop_reason": record["stop_reason"],
                        "tools": record["tool_call_counts"],
                        "elapsed": record["elapsed"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return sorted(records, key=lambda item: int(item["index"]))


def _write_summary(
    run_root: Path,
    config: dict[str, Any],
    mode_records: dict[str, list[dict[str, Any]]],
    outputs: dict[str, dict[str, str]],
    selected_rows: list[tuple[int, dict[str, str]]],
) -> dict[str, Any]:
    mode_summaries = {mode: _summarize(records) for mode, records in mode_records.items()}
    baseline = {int(record["index"]): record for record in mode_records.get("baseline_no_memory", [])}
    memory = {int(record["index"]): record for record in mode_records.get("memory_query_only", [])}
    comparison = []
    for idx, row in selected_rows:
        b_record = baseline.get(idx)
        m_record = memory.get(idx)
        comparison.append(
            {
                "index": idx,
                "expected": str(row.get("answer") or ""),
                "baseline_predicted": b_record.get("answer") if b_record else None,
                "baseline_correct": b_record.get("correct") if b_record else None,
                "baseline_f1": b_record.get("f1") if b_record else None,
                "memory_predicted": m_record.get("answer") if m_record else None,
                "memory_correct": m_record.get("correct") if m_record else None,
                "memory_f1": m_record.get("f1") if m_record else None,
            }
        )
    summary = {
        "run_root": str(run_root),
        "config": config,
        "mode_summaries": mode_summaries,
        "delta_memory_minus_baseline": {
            "accuracy": mode_summaries.get("memory_query_only", {}).get("accuracy", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("accuracy", 0.0),
            "exact_match": mode_summaries.get("memory_query_only", {}).get("exact_match", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("exact_match", 0.0),
            "avg_f1": mode_summaries.get("memory_query_only", {}).get("avg_f1", 0.0)
            - mode_summaries.get("baseline_no_memory", {}).get("avg_f1", 0.0),
        },
        "outputs": outputs,
        "comparison": comparison,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/root/harness-sii-browser-service/benchmark_answered.csv")
    parser.add_argument("--out", default="logs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--mode", choices=["both", "baseline_no_memory", "memory_query_only"], default="both")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to run. Use 0 for all rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=26)
    parser.add_argument("--max-llm-tokens", type=int, default=120000)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--max-llm-call-seconds", type=float, default=600.0)
    parser.add_argument("--min-llm-call-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in mode JSONL files.")
    parser.add_argument("--no-save-trace", dest="save_trace", action="store_false")
    parser.set_defaults(save_trace=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_default_env()
    csv_path = Path(args.csv)
    run_name = args.run_name or f"benchmark_answered_special_26s_120k_c{args.concurrency}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.out) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    fieldnames, input_rows, selected_rows = _load_rows(csv_path, args.n, args.offset)
    images_dir = run_root.resolve() / "images"
    image_metas = []
    for idx, row in selected_rows:
        image_metas.append({"index": idx, **_prepare_image(row, idx, images_dir)})
    overall = _overall_memory_guidance()
    config = {
        "run_root": str(run_root),
        "csv_path": str(csv_path),
        "n": len(selected_rows),
        "offset": args.offset,
        "concurrency": args.concurrency,
        "backend": os.getenv("LLM_BACKEND"),
        "base_url": os.getenv("VLLM_BASE_URL"),
        "model": os.getenv("VLLM_MODEL"),
        "enable_thinking": os.getenv("VLLM_ENABLE_THINKING"),
        "runtime_mode": os.getenv("SII_AGENT_RUNTIME_MODE"),
        "memory_root": os.getenv("SII_AGENT_MEMORY_ROOT"),
        "memory_read_only": True,
        "reflection_retry": False,
        "max_steps": args.max_steps,
        "max_llm_tokens": args.max_llm_tokens,
        "max_wall_seconds": args.max_wall_seconds,
        "max_llm_call_seconds": args.max_llm_call_seconds,
        "min_llm_call_seconds": args.min_llm_call_seconds,
        "search_proxy_timeout": os.getenv("SEARCH_PROXY_TIMEOUT"),
        "memory_search_summary_timeout": os.getenv("SII_MEMORY_SEARCH_SUMMARY_TIMEOUT"),
        "output_format_reference": str(ROOT / "benchmarkreadme.md"),
        "submission_format": "group_{mode}.json trace, group_{mode}.csv answers, group_{mode}.zip bundle",
        "overall_in_prompt": bool(overall),
        "overall_preview": overall[:700],
        "save_trace": args.save_trace,
        "image_handling": {
            "images_dir": str(images_dir),
            "nonempty": sum(1 for meta in image_metas if meta.get("kind") != "none"),
            "decoded": sum(1 for meta in image_metas if meta.get("kind") in {"raw_base64", "data_url"}),
            "invalid": sum(1 for meta in image_metas if meta.get("kind") == "invalid_image_payload"),
            "max_original_chars": max((int(meta.get("original_chars") or 0) for meta in image_metas), default=0),
        },
    }
    (run_root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    mode_records: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, dict[str, str]] = {}
    for mode, tools, extra_system in _mode_specs(args.mode):
        records = _run_mode(run_root, mode, tools, extra_system, selected_rows, args)
        mode_records[mode] = records
        outputs[mode] = _write_mode_outputs(run_root, mode, records, fieldnames, input_rows)
        print(json.dumps({"mode_summary": mode, **_summarize(records), "outputs": outputs[mode]}, ensure_ascii=False), flush=True)

    summary = _write_summary(run_root, config, mode_records, outputs, selected_rows)
    print("FINAL_SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
