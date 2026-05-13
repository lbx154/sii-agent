"""Evaluate baseline vs evolved on a benchmark; write per-run JSONL + summary."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import track

from agent.runner import run_baseline, run_evolved
from harness.controller import HarnessConfig
from memory.store import MemoryStore
from .datasets import LOADERS

console = Console()


def run(task: str, mode: str, n: int, out_dir: str, max_steps: int):
    assert mode in {"baseline", "evolved"}
    cfg = HarnessConfig(max_steps=max_steps)
    out = Path(out_dir) / f"{task}_{mode}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    f_jsonl = (out / "runs.jsonl").open("w", encoding="utf-8")

    memory = MemoryStore(root=out / "memory") if mode == "evolved" else None
    n_correct = 0
    n_total = 0
    sum_steps = sum_tools = 0

    for ex in track(list(LOADERS[task](n)), description=f"{task}/{mode}"):
        if mode == "baseline":
            outcome = run_baseline(ex["question"], expected=ex["answer"], cfg=cfg)
        else:
            outcome = run_evolved(ex["question"], expected=ex["answer"], cfg=cfg, memory=memory)
        n_total += 1
        if outcome.correct:
            n_correct += 1
        sum_steps += outcome.result.steps
        sum_tools += outcome.result.tool_calls
        f_jsonl.write(json.dumps({
            "id": ex["id"],
            "question": ex["question"],
            "expected": ex["answer"],
            "predicted": outcome.result.final_answer,
            "correct": outcome.correct,
            "steps": outcome.result.steps,
            "tool_calls": outcome.result.tool_calls,
            "stop_reason": outcome.result.stop_reason,
            "elapsed": outcome.result.elapsed,
            "reflection": outcome.reflection,
        }, ensure_ascii=False) + "\n")
        f_jsonl.flush()

    f_jsonl.close()
    summary = {
        "task": task, "mode": mode, "n": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "avg_steps": sum_steps / n_total if n_total else 0.0,
        "avg_tool_calls": sum_tools / n_total if n_total else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(LOADERS), required=True)
    p.add_argument("--mode", choices=["baseline", "evolved"], required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--out", default="logs/eval")
    args = p.parse_args()
    run(args.task, args.mode, args.n, args.out, args.max_steps)


if __name__ == "__main__":
    main()
