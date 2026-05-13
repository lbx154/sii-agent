"""Merge multiple BrowseComp-Plus run directories with an LLM answer selector."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


_LOCAL = threading.local()
_JSON_RE = re.compile(r"\{.*\}", re.S)

REFUSAL_PHRASES = (
    "cannot find",
    "could not identify",
    "not enough information",
    "insufficient",
    "unable to",
    "i do not have",
    "i don't have",
    "no definitive",
    "cannot determine",
    "not found",
    "do not know",
)


def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    cache_key = (base_url, api_key, timeout)
    cached = getattr(_LOCAL, "client", None)
    if cached is None or getattr(_LOCAL, "client_key", None) != cache_key:
        _LOCAL.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _LOCAL.client_key = cache_key
    return _LOCAL.client


def _safe_qid(query_id: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(query_id))


def _load_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            questions[str(record["query_id"])] = str(record.get("query") or record.get("question") or "")
    return questions


def _load_run(run_dir: Path, query_id: str) -> dict | None:
    path = run_dir / "runs" / f"{_safe_qid(query_id)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _answer(run: dict | None) -> str:
    if not run:
        return ""
    result = run.get("result") or []
    if not result:
        return ""
    return str(result[-1].get("output") or "").strip()


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _parse_selector_json(text: str) -> dict:
    match = _JSON_RE.search(text)
    return json.loads(match.group(0) if match else text)


def _selector_prompt(question: str, candidates: list[tuple[str, str, str]]) -> str:
    blocks = [
        f"Candidate {label} ({description}):\n{answer}"
        for label, description, answer in candidates
    ]
    labels = "|".join(label for label, _, _ in candidates)
    return (
        "You are a strict answer selector for a fixed-corpus question-answering benchmark.\n\n"
        f"Question:\n{question}\n\n"
        + "\n\n".join(blocks)
        + "\n\nChoose the candidate most likely to be the exact answer. Prefer concise, "
        "specific answers that directly satisfy all constraints. Penalize refusals, "
        "caveats, broad explanations, and unsupported guesses. If two candidates are "
        "equivalent, choose the shorter exact answer. Do not use any hidden gold answer; "
        "use only the question and candidates.\n\n"
        f'Return only JSON: {{"choice":"{labels}", "final_answer":"..."}} with the selected label.'
    )


def _choose(
    question: str,
    candidates: list[tuple[str, str, str]],
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    retries: int,
) -> tuple[str, str]:
    nonempty = [(label, answer) for label, _, answer in candidates if answer]
    unique_answers = {answer for _, answer in nonempty}
    if len(unique_answers) == 1:
        return "same", next(iter(unique_answers))

    non_refusal = [(label, answer) for label, answer in nonempty if not _is_refusal(answer)]
    if len(non_refusal) == 1:
        return non_refusal[0]

    labels = {label for label, _, _ in candidates}
    prompt = _selector_prompt(question, candidates)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = _client(base_url, api_key, timeout).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            obj = _parse_selector_json(response.choices[0].message.content or "")
            choice = str(obj.get("choice", "")).strip().upper()
            final_answer = str(obj.get("final_answer", "")).strip()
            if choice in labels:
                fallback = next(answer for label, _, answer in candidates if label == choice)
                return choice, final_answer or fallback
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 8))

    for label, answer in non_refusal:
        return f"{label}_fallback", answer
    if nonempty:
        return "fallback", nonempty[0][1]
    raise RuntimeError(f"selector failed and no candidates were available: {last_error}")


def _merge_counts(runs: list[dict | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for run in runs:
        if not run:
            continue
        for name, count in (run.get("tool_call_counts") or {}).items():
            counts[str(name)] += int(count)
    return dict(counts)


def _merge_docids(runs: list[dict | None]) -> list[str]:
    docids: set[str] = set()
    for run in runs:
        if run:
            docids.update(str(docid) for docid in (run.get("retrieved_docids") or []))
    return sorted(docids)


def merge(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    runs_out = out / "runs"
    runs_out.mkdir(parents=True, exist_ok=True)

    source_dirs = [Path(path) for path in args.run_dirs]
    if not 2 <= len(source_dirs) <= 26:
        raise ValueError("Provide between 2 and 26 run directories.")
    labels = [chr(ord("A") + i) for i in range(len(source_dirs))]

    gt_path = Path(args.ground_truth) if args.ground_truth else source_dirs[0] / "browsecomp_plus_decrypted.jsonl"
    qrel_path = Path(args.qrel_evidence) if args.qrel_evidence else source_dirs[0] / "qrel_evidence.txt"
    shutil.copyfile(gt_path, out / "browsecomp_plus_decrypted.jsonl")
    if qrel_path.exists():
        shutil.copyfile(qrel_path, out / "qrel_evidence.txt")

    descriptions = args.descriptions or [path.name for path in source_dirs]
    if len(descriptions) != len(source_dirs):
        raise ValueError("--descriptions must match --run-dirs length.")

    questions = _load_questions(gt_path)
    choices: list[str] = []

    def run_one(query_id: str) -> str:
        loaded_runs = [_load_run(source_dir, query_id) for source_dir in source_dirs]
        candidates = [
            (label, description, _answer(run))
            for label, description, run in zip(labels, descriptions, loaded_runs, strict=True)
        ]
        choice, final_answer = _choose(
            questions[query_id],
            candidates,
            args.model,
            args.base_url,
            args.api_key,
            args.timeout,
            args.retries,
        )
        record = {
            "metadata": {
                "model": f"{args.model} ({len(source_dirs)}-run selector)",
                "mode": "sii-agent-selector",
                "selector_choice": choice,
                "source_runs": [str(path) for path in source_dirs],
            },
            "query_id": query_id,
            "tool_call_counts": _merge_counts(loaded_runs),
            "status": "completed" if final_answer else "incomplete",
            "retrieved_docids": _merge_docids(loaded_runs),
            "result": [{"type": "output_text", "output": final_answer}],
        }
        (runs_out / f"{_safe_qid(query_id)}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return choice

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_one, query_id) for query_id in sorted(questions, key=lambda x: int(x))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="merge/select", unit="query"):
            choices.append(future.result())

    summary = {
        "task": "browsecomp-plus",
        "mode": "selector",
        "n": len(questions),
        "source_runs": [str(path) for path in source_dirs],
        "selector_model": args.model,
        "runs_dir": str(runs_out),
        "ground_truth": str(out / "browsecomp_plus_decrypted.jsonl"),
        "qrel_evidence": str(out / "qrel_evidence.txt") if qrel_path.exists() else None,
        "choices": dict(Counter(choices)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="BrowseComp run directories to merge.")
    parser.add_argument("--descriptions", nargs="+", default=None, help="Human-readable labels for run dirs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--qrel-evidence", default=None)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(merge(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
