"""Evaluate baseline vs evolved on a benchmark; write per-run JSONL + summary."""
from __future__ import annotations
import argparse
import json
import os
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


def _attempt_summary(result) -> dict | None:
    if result is None:
        return None
    return {
        "final_answer": result.final_answer,
        "rationale": result.rationale,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "tool_call_counts": result.tool_call_counts,
        "stop_reason": result.stop_reason,
        "finish_reasons": result.finish_reasons,
        "elapsed": result.elapsed,
    }


def _memory_context_refs(context: str | None) -> list[str]:
    if not context:
        return []
    refs = set()
    import re

    for match in re.finditer(r"\[([A-Za-z0-9_:-]+)\]", context):
        refs.add(match.group(1))
    for match in re.finditer(r"\(past\s+([^:()]+):\s*([^()]+)\)", context):
        outcome = match.group(1).strip().replace(" ", "_")
        mode = match.group(2).strip().replace(" ", "_")
        refs.add(f"lesson:{outcome}:{mode}")
    return sorted(refs)


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _record(
    ex: dict,
    outcome: RunOutcome,
    save_traces: bool = False,
    lesson_context: str | None = None,
) -> dict:
    scores = score_answer(outcome.result.final_answer, ex["answer"])
    record = {
        "id": ex["id"],
        "task": ex.get("task"),
        "split": ex.get("split"),
        "question": ex["question"],
        "expected": ex["answer"],
        "predicted": outcome.result.final_answer,
        "rationale": outcome.result.rationale,
        "correct": outcome.correct,
        "exact": scores["exact"],
        "f1": scores["f1"],
        "steps": outcome.result.steps,
        "tool_calls": outcome.result.tool_calls,
        "tool_call_counts": outcome.result.tool_call_counts,
        "stop_reason": outcome.result.stop_reason,
        "finish_reasons": outcome.result.finish_reasons,
        "short_memory_stats": outcome.result.short_memory_stats,
        "elapsed": outcome.result.elapsed,
        "reflection": outcome.reflection,
        "final_refinement": outcome.final_refinement,
    }
    if outcome.first_result is not None:
        record["first_attempt"] = _attempt_summary(outcome.first_result)
        if outcome.retry_result is not None:
            record["retry_attempt"] = _attempt_summary(outcome.retry_result)
        record["selected_attempt"] = outcome.selected_attempt
        record["retry_selected"] = outcome.retry_selected
        record["retry_reason"] = outcome.retry_reason
        record["reflection_useful"] = outcome.reflection_useful
    if save_traces:
        record["trajectory"] = outcome.result.trajectory
        if lesson_context:
            record["lesson_context"] = lesson_context
            record["memory_context_refs"] = _memory_context_refs(lesson_context)
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
        outcome = run_baseline(ex["question"], expected=ex["answer"], cfg=cfg, task=ex.get("task"))
    else:
        outcome = run_evolved(
            ex["question"],
            expected=ex["answer"],
            cfg=cfg,
            memory=memory,
            lesson_context=lesson_context,
            allow_reflection=allow_reflection,
            use_gold_for_reflection=use_gold_for_reflection,
            task=ex.get("task"),
        )
    return _record(ex, outcome, save_traces=save_traces, lesson_context=lesson_context)


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
    tool_profile: str | None = None,
    short_memory: bool = False,
    short_memory_max_chars: int = 2500,
):
    assert mode in {"baseline", "evolved"}
    if tool_profile:
        os.environ["SII_AGENT_TOOL_PROFILE"] = tool_profile
    cfg = HarnessConfig(
        max_steps=max_steps,
        max_wall_seconds=max_wall_seconds,
        max_llm_tokens=max_llm_tokens,
        max_llm_call_seconds=max_llm_call_seconds,
        min_llm_call_seconds=min_llm_call_seconds,
        use_short_memory=short_memory,
        short_memory_max_chars=short_memory_max_chars,
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
                    ex["id"]: memory.render_for_prompt(
                        ex["question"],
                        k=3,
                        task=task,
                        include_successes=(task != "2wiki"),
                    )
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
        "tool_profile": tool_profile or os.getenv("SII_AGENT_TOOL_PROFILE", "benchmark"),
        "short_memory": short_memory,
        "short_memory_max_chars": short_memory_max_chars if short_memory else None,
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
    p.add_argument("--short-memory", action="store_true", help="Enable per-attempt compact working memory.")
    p.add_argument("--short-memory-max-chars", type=int, default=2500)
    p.add_argument(
        "--tool-profile",
        choices=["benchmark", "default", "visual", "rich", "full", "all"],
        default=None,
        help="Tool set exposed to the agent. 'visual' is tuned for image QA; 'all' exposes every registered tool.",
    )
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
        args.tool_profile,
        args.short_memory,
        args.short_memory_max_chars,
    )


if __name__ == "__main__":
    main()
