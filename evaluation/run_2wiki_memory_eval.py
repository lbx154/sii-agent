"""Two-phase 2Wiki evaluation with train-distilled, read-only memory.

This wrapper keeps gold reflection off the scored validation/test phase. It
first distills lessons from the 2Wiki train split, then evaluates with that
memory mounted read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from .run_eval import run


console = Console()

MODEL_ENV_KEYS = (
    "LLM_BACKEND",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "VLLM_ENABLE_THINKING",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "SII_AGENT_TOOL_PROFILE",
)

SII_ENV_KEYS = (
    "SII_2WIKI_ENABLE_REFLECTION",
    "SII_2WIKI_ENABLE_LESSONS",
    "SII_2WIKI_ENABLE_SKILLS",
    "SII_2WIKI_ENABLE_TYPED_POLICIES",
)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _memory_counts(root: Path) -> dict[str, int]:
    return {
        "lessons": _jsonl_count(root / "lessons.jsonl"),
        "episodes": _jsonl_count(root / "episodes.jsonl"),
        "skills": _jsonl_count(root / "skills.jsonl"),
    }


def _env_snapshot(keys: tuple[str, ...]) -> dict[str, str | None]:
    return {key: os.getenv(key) for key in keys}


def _resolved_sii_env(*, enable_lessons: bool) -> dict[str, str | None]:
    env = _env_snapshot(SII_ENV_KEYS)
    env["SII_2WIKI_ENABLE_REFLECTION"] = "1"
    env["SII_2WIKI_ENABLE_LESSONS"] = "1" if enable_lessons else "0"
    env["SII_2WIKI_ENABLE_SKILLS"] = None
    env["SII_2WIKI_ENABLE_TYPED_POLICIES"] = None
    return env


def _apply_sii_env(*, enable_lessons: bool) -> None:
    os.environ["SII_2WIKI_ENABLE_REFLECTION"] = "1"
    os.environ["SII_2WIKI_ENABLE_LESSONS"] = "1" if enable_lessons else "0"
    os.environ.pop("SII_2WIKI_ENABLE_SKILLS", None)
    os.environ.pop("SII_2WIKI_ENABLE_TYPED_POLICIES", None)


def _nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _positive_int(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{name} must be > 0")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Distill 2Wiki lessons from train with gold reflection, then evaluate "
            "validation/test with the distilled memory mounted read-only."
        )
    )
    p.add_argument("--out", default="logs/opd_eval")
    p.add_argument("--memory-root", default=None, help="Memory directory to create/reuse. Defaults to <run-root>/memory.")
    p.add_argument(
        "--distill-memory-mode",
        choices=["fresh", "read_write"],
        default="fresh",
        help="fresh clears the distill memory root; read_write appends to an existing memory root.",
    )
    p.add_argument(
        "--force-fresh-memory",
        action="store_true",
        help="Allow --distill-memory-mode fresh to delete an existing non-empty --memory-root.",
    )
    p.add_argument("--memory-split", choices=["train"], default="train")
    p.add_argument("--memory-n", type=int, default=500)
    p.add_argument("--memory-offset", type=int, default=0)
    p.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--eval-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-wall-seconds", type=float, default=600.0)
    p.add_argument("--max-llm-tokens", type=int, default=12000)
    p.add_argument("--max-llm-call-seconds", type=float, default=600.0)
    p.add_argument("--min-llm-call-seconds", type=float, default=20.0)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument(
        "--evolve-batch-size",
        type=int,
        default=None,
        help="Passed to both distill and evolved eval. Defaults to --concurrency inside run_eval.",
    )
    p.add_argument("--memory-k", type=int, default=3)
    p.add_argument(
        "--include-success-memory",
        action="store_true",
        help="Include train successful episode memories during read-only eval. Lessons are always enabled for eval.",
    )
    p.add_argument("--tool-profile", choices=["benchmark", "default", "visual", "rich", "full", "all"], default=None)
    p.add_argument("--short-memory", action="store_true")
    p.add_argument("--short-memory-max-chars", type=int, default=2500)
    p.add_argument(
        "--eval-no-reflection",
        action="store_true",
        help="Disable no-gold self-reflection/retry during the scored eval phase.",
    )
    p.add_argument(
        "--eval-force-reflection",
        action="store_true",
        help="Run no-gold reflection on every scored eval sample so retry can trigger without writing memory.",
    )
    p.add_argument("--save-eval-traces", action="store_true")
    p.add_argument("--run-baseline", action="store_true", help="Also run a baseline on the eval split for comparison.")
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and exit without running.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _positive_int(parser, "--memory-n", args.memory_n)
    _positive_int(parser, "--eval-n", args.eval_n)
    _positive_int(parser, "--concurrency", args.concurrency)
    if args.evolve_batch_size is not None:
        _positive_int(parser, "--evolve-batch-size", args.evolve_batch_size)
    _positive_int(parser, "--memory-k", args.memory_k)

    if args.tool_profile:
        os.environ["SII_AGENT_TOOL_PROFILE"] = args.tool_profile

    ts = int(time.time())
    run_root = Path(args.out) / f"2wiki_memory_fair_{ts}"
    memory_root = Path(args.memory_root) if args.memory_root else run_root / "memory"
    if args.memory_root and memory_root.exists() and not memory_root.is_dir():
        parser.error(f"--memory-root {memory_root} exists but is not a directory.")
    if (
        args.memory_root
        and args.distill_memory_mode == "fresh"
        and _nonempty_dir(memory_root)
        and not args.force_fresh_memory
    ):
        parser.error(
            f"--memory-root {memory_root} is non-empty; pass --force-fresh-memory "
            "or use --distill-memory-mode read_write."
        )

    common_run_args = {
        "max_steps": args.max_steps,
        "concurrency": args.concurrency,
        "evolve_batch_size": args.evolve_batch_size,
        "max_wall_seconds": args.max_wall_seconds,
        "max_llm_tokens": args.max_llm_tokens,
        "max_llm_call_seconds": args.max_llm_call_seconds,
        "min_llm_call_seconds": args.min_llm_call_seconds,
        "tool_profile": args.tool_profile,
        "short_memory": args.short_memory,
        "short_memory_max_chars": args.short_memory_max_chars,
    }
    workflow_config: dict[str, Any] = {
        "workflow": "2wiki_train_distilled_read_only_memory_eval",
        "run_root": str(run_root),
        "memory_root": str(memory_root),
        "fairness_contract": {
            "gold_reflection_split": args.memory_split,
            "scored_eval_split": args.eval_split,
            "eval_uses_gold_reflection": False,
            "eval_memory_mode": "read_only",
            "notes": (
                "Gold answers are only passed to reflection on the 2Wiki train split. "
                "The scored validation/test phase mounts the resulting memory read-only. "
                "Literal train answers are redacted before lesson persistence, but near-duplicate "
                "train/eval examples can still transfer task-specific hints."
            ),
        },
        "distill": {
            "split": args.memory_split,
            "n": args.memory_n,
            "offset": args.memory_offset,
            "memory_mode": args.distill_memory_mode,
            "use_gold_for_reflection": True,
            "force_reflection": False,
            "save_traces": True,
            "lesson_retrieval_enabled_during_distill": False,
        },
        "eval": {
            "split": args.eval_split,
            "n": args.eval_n,
            "offset": args.eval_offset,
            "memory_mode": "read_only",
            "use_gold_for_reflection": False,
            "force_reflection": args.eval_force_reflection,
            "allow_reflection": not args.eval_no_reflection,
            "save_traces": args.save_eval_traces,
            "include_success_memory": args.include_success_memory,
        },
        "baseline": {"enabled": args.run_baseline},
        "common_run_args": common_run_args,
        "model_env": _env_snapshot(MODEL_ENV_KEYS),
        "distill_sii_env": _resolved_sii_env(enable_lessons=False),
        "eval_sii_env": _resolved_sii_env(enable_lessons=True),
    }
    console.print_json(data=workflow_config)
    if args.dry_run:
        return

    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "workflow_config.json", workflow_config)

    counts_before = _memory_counts(memory_root)
    _apply_sii_env(enable_lessons=False)
    distill_summary = run(
        task="2wiki",
        mode="evolved",
        split=args.memory_split,
        n=args.memory_n,
        out_dir=str(run_root / "distill"),
        offset=args.memory_offset,
        memory_root=str(memory_root),
        memory_mode=args.distill_memory_mode,
        memory_k=args.memory_k,
        include_success_memory=False,
        allow_reflection=True,
        use_gold_for_reflection=True,
        force_reflection=False,
        save_traces=True,
        **common_run_args,
    )
    counts_after_distill = _memory_counts(memory_root)
    if counts_after_distill["lessons"] == 0:
        console.print("[yellow]No lessons were distilled; eval will still run with an empty lesson bank.[/yellow]")

    baseline_summary = None
    if args.run_baseline:
        baseline_summary = run(
            task="2wiki",
            mode="baseline",
            split=args.eval_split,
            n=args.eval_n,
            out_dir=str(run_root / "baseline"),
            offset=args.eval_offset,
            memory_root=None,
            memory_mode="fresh",
            memory_k=args.memory_k,
            include_success_memory=False,
            allow_reflection=False,
            use_gold_for_reflection=False,
            force_reflection=False,
            save_traces=args.save_eval_traces,
            **common_run_args,
        )

    _apply_sii_env(enable_lessons=True)
    eval_summary = run(
        task="2wiki",
        mode="evolved",
        split=args.eval_split,
        n=args.eval_n,
        out_dir=str(run_root / "eval"),
        offset=args.eval_offset,
        memory_root=str(memory_root),
        memory_mode="read_only",
        memory_k=args.memory_k,
        include_success_memory=args.include_success_memory,
        allow_reflection=not args.eval_no_reflection,
        use_gold_for_reflection=False,
        force_reflection=args.eval_force_reflection,
        save_traces=args.save_eval_traces,
        **common_run_args,
    )
    counts_after_eval = _memory_counts(memory_root)

    summary: dict[str, Any] = {
        "workflow": workflow_config["workflow"],
        "run_root": str(run_root),
        "memory_root": str(memory_root),
        "memory_counts_before": counts_before,
        "memory_counts_after_distill": counts_after_distill,
        "memory_counts_after_eval": counts_after_eval,
        "distill_summary": distill_summary,
        "baseline_summary": baseline_summary,
        "eval_summary": eval_summary,
        "fairness_contract": workflow_config["fairness_contract"],
    }
    _write_json(run_root / "workflow_summary.json", summary)
    console.print_json(data=summary)


if __name__ == "__main__":
    main()
