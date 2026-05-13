"""Evaluate baseline vs evolved on a benchmark; write per-run JSONL + summary."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import track

from agent.runner import RunOutcome, run_baseline, run_evolved
from agent.scoring import score_answer
from harness.controller import HarnessConfig
from memory.store import MemoryStore
from .datasets import DEFAULT_SPLITS, LOADERS, load_examples

console = Console()


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _record(ex: dict, outcome: RunOutcome, save_traces: bool = False) -> dict:
    scores = score_answer(outcome.result.final_answer, ex["answer"])
    record = {
        "id": ex["id"],
        "task": ex.get("task"),
        "split": ex.get("split"),
        "question": ex["question"],
        "expected": ex["answer"],
        "predicted": outcome.result.final_answer,
        "correct": outcome.correct,
        "exact": scores["exact"],
        "f1": scores["f1"],
        "steps": outcome.result.steps,
        "tool_calls": outcome.result.tool_calls,
        "tool_call_counts": outcome.result.tool_call_counts,
        "stop_reason": outcome.result.stop_reason,
        "finish_reasons": outcome.result.finish_reasons,
        "elapsed": outcome.result.elapsed,
        "reflection": outcome.reflection,
    }
    if save_traces:
        record["trajectory"] = outcome.result.trajectory
    return record


def _run_one(
    ex: dict,
    mode: str,
    cfg: HarnessConfig,
    memory: MemoryStore | None,
    lesson_context: str | None = None,
    allow_reflection: bool = True,
    use_gold_for_reflection: bool = False,
    save_traces: bool = False,
) -> dict:
    if mode == "baseline":
        outcome = run_baseline(ex["question"], expected=ex["answer"], cfg=cfg)
    else:
        outcome = run_evolved(
            ex["question"],
            expected=ex["answer"],
            cfg=cfg,
            memory=memory,
            lesson_context=lesson_context,
            allow_reflection=allow_reflection,
            use_gold_for_reflection=use_gold_for_reflection,
        )
    return _record(ex, outcome, save_traces=save_traces)


def _run_parallel_batch(
    examples: list[dict],
    mode: str,
    cfg: HarnessConfig,
    memory: MemoryStore | None,
    concurrency: int,
    description: str,
    lesson_contexts: dict[str, str] | None = None,
    allow_reflection: bool = True,
    use_gold_for_reflection: bool = False,
    save_traces: bool = False,
    batch_index: int | None = None,
) -> Iterable[dict]:
    max_workers = max(1, min(concurrency, len(examples)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _run_one,
                ex,
                mode,
                cfg,
                memory,
                (lesson_contexts or {}).get(ex["id"]),
                allow_reflection,
                use_gold_for_reflection,
                save_traces,
            )
            for ex in examples
        ]
        for future in track(as_completed(futures), total=len(futures), description=description):
            record = future.result()
            if batch_index is not None:
                record["batch_index"] = batch_index
            yield record


def _prepare_memory(
    mode: str,
    out: Path,
    memory_root: str | None,
    memory_mode: str,
) -> MemoryStore | None:
    if mode != "evolved":
        return None

    root = Path(memory_root) if memory_root else out / "memory"
    if memory_mode == "fresh" and root.exists():
        shutil.rmtree(root)
    return MemoryStore(root=root, read_only=(memory_mode == "read_only"))


def run(
    task: str,
    mode: str,
    split: str | None,
    n: int,
    out_dir: str,
    max_steps: int,
    concurrency: int,
    evolve_batch_size: int | None,
    max_wall_seconds: float,
    max_llm_tokens: int,
    max_llm_call_seconds: float,
    min_llm_call_seconds: float,
    offset: int,
    memory_root: str | None,
    memory_mode: str,
    allow_reflection: bool,
    use_gold_for_reflection: bool,
    save_traces: bool,
):
    assert mode in {"baseline", "evolved"}
    cfg = HarnessConfig(
        max_steps=max_steps,
        max_wall_seconds=max_wall_seconds,
        max_llm_tokens=max_llm_tokens,
        max_llm_call_seconds=max_llm_call_seconds,
        min_llm_call_seconds=min_llm_call_seconds,
    )
    out = Path(out_dir) / f"{task}_{mode}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    f_jsonl = (out / "runs.jsonl").open("w", encoding="utf-8")

    memory = _prepare_memory(mode, out, memory_root, memory_mode)
    n_correct = 0
    n_total = 0
    sum_steps = sum_tools = 0
    exact_correct = 0
    sum_f1 = 0.0
    n_f1 = 0
    split = split or DEFAULT_SPLITS[task]
    examples = load_examples(task, n, offset, split=split)
    t0 = time.time()

    if mode == "baseline":
        result_iter = _run_parallel_batch(
            examples,
            mode,
            cfg,
            memory,
            concurrency,
            f"{task}/{mode}",
            allow_reflection=allow_reflection,
            use_gold_for_reflection=use_gold_for_reflection,
            save_traces=save_traces,
        )
    else:
        batch_size = evolve_batch_size or concurrency
        batch_size = max(1, batch_size)

        def evolved_records() -> Iterable[dict]:
            total_batches = (len(examples) + batch_size - 1) // batch_size
            for i, batch in enumerate(_chunks(examples, batch_size), 1):
                assert memory is not None
                lesson_contexts = {
                    ex["id"]: memory.render_for_prompt(ex["question"])
                    for ex in batch
                }
                yield from _run_parallel_batch(
                    batch,
                    mode,
                    cfg,
                    memory,
                    concurrency,
                    f"{task}/{mode} batch {i}/{total_batches}",
                    lesson_contexts,
                    allow_reflection=allow_reflection,
                    use_gold_for_reflection=use_gold_for_reflection,
                    save_traces=save_traces,
                    batch_index=i,
                )

        result_iter = evolved_records()

    for record in result_iter:
        n_total += 1
        if record["correct"]:
            n_correct += 1
        if record.get("exact"):
            exact_correct += 1
        if isinstance(record.get("f1"), (int, float)):
            sum_f1 += float(record["f1"])
            n_f1 += 1
        sum_steps += record["steps"]
        sum_tools += record["tool_calls"]
        f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_jsonl.flush()

    f_jsonl.close()
    summary = {
        "task": task, "split": split, "mode": mode, "n": n_total,
        "offset": offset,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "exact_match": exact_correct / n_total if n_total else 0.0,
        "avg_f1": sum_f1 / n_f1 if n_f1 else 0.0,
        "avg_steps": sum_steps / n_total if n_total else 0.0,
        "avg_tool_calls": sum_tools / n_total if n_total else 0.0,
        "concurrency": concurrency,
        "evolve_batch_size": evolve_batch_size if mode == "evolved" else None,
        "max_wall_seconds": max_wall_seconds,
        "max_llm_tokens": max_llm_tokens,
        "max_llm_call_seconds": max_llm_call_seconds,
        "min_llm_call_seconds": min_llm_call_seconds,
        "memory_root": str(memory.root) if memory else None,
        "memory_mode": memory_mode if mode == "evolved" else None,
        "allow_reflection": allow_reflection if mode == "evolved" else None,
        "use_gold_for_reflection": use_gold_for_reflection if mode == "evolved" else None,
        "save_traces": save_traces,
        "wall_seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(LOADERS), required=True)
    p.add_argument("--mode", choices=["baseline", "evolved"], required=True)
    p.add_argument(
        "--split",
        default=None,
        help="Dataset split. Defaults to test for SimpleQA and validation for 2Wiki. 2Wiki also supports train/test; SimpleQA has no public train split.",
    )
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-wall-seconds", type=float, default=120.0)
    p.add_argument("--max-llm-tokens", type=int, default=1536)
    p.add_argument("--max-llm-call-seconds", type=float, default=60.0)
    p.add_argument("--min-llm-call-seconds", type=float, default=20.0)
    p.add_argument("--concurrency", type=int, default=40)
    p.add_argument("--memory-root", default=None)
    p.add_argument(
        "--memory-mode",
        choices=["fresh", "read_write", "read_only"],
        default="fresh",
        help="Evolved mode only. fresh clears the memory root before running; read_write reuses and appends; read_only reuses without writing.",
    )
    p.add_argument("--no-reflection", action="store_true")
    p.add_argument(
        "--gold-reflection",
        action="store_true",
        help="Pass gold answers into reflection. Off by default to avoid answer leakage in memory.",
    )
    p.add_argument("--save-traces", action="store_true")
    p.add_argument(
        "--evolve-batch-size",
        type=int,
        default=None,
        help="Evolved mode only: freeze memory per batch, then make new lessons visible to the next batch. Defaults to --concurrency.",
    )
    p.add_argument("--out", default="logs/eval")
    args = p.parse_args()
    run(
        args.task,
        args.mode,
        args.split,
        args.n,
        args.out,
        args.max_steps,
        args.concurrency,
        args.evolve_batch_size,
        args.max_wall_seconds,
        args.max_llm_tokens,
        args.max_llm_call_seconds,
        args.min_llm_call_seconds,
        args.offset,
        args.memory_root,
        args.memory_mode,
        not args.no_reflection,
        args.gold_reflection,
        args.save_traces,
    )


if __name__ == "__main__":
    main()
