"""Optional shell command tool for controlled experiments."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .registry import register


_DANGEROUS_PATTERNS = (" eval ", "\neval ", ";eval ", " rm -rf /", "mkfs.")
_DANGEROUS_REGEXES = (
    re.compile(r"\$\{[^}]+@P\}"),
    re.compile(r"\$\{![^}]+\}"),
)


def shell_tool_enabled() -> bool:
    return os.getenv("SII_AGENT_ENABLE_SHELL_TOOL", "").strip().lower() in {"1", "true", "yes"}


def _clamp_int(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _safe_cwd(cwd: str) -> Path:
    root = Path(os.getenv("SII_AGENT_SHELL_ROOT", os.getcwd())).resolve()
    target = Path(cwd or ".").expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if root not in [target, *target.parents]:
        raise ValueError(f"cwd must stay under shell root: {root}")
    return target


def _validate_command(command: str) -> None:
    lowered = f" {command.lower()} "
    if any(pattern in command or pattern in lowered for pattern in _DANGEROUS_PATTERNS):
        raise ValueError("command contains blocked shell expansion/eval/destructive pattern")
    if any(pattern.search(command) for pattern in _DANGEROUS_REGEXES):
        raise ValueError("command contains blocked shell expansion/eval/destructive pattern")


@register(
    "bash_exec",
    "Run a bash command in the repository shell environment. Disabled unless SII_AGENT_ENABLE_SHELL_TOOL=1. "
    "Use only for explicit command-line inspection or memory-file searches; prefer specialized tools when available.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string", "default": "."},
            "timeout_seconds": {"type": "integer", "default": 30, "minimum": 1, "maximum": 120},
            "max_output_chars": {"type": "integer", "default": 12000, "minimum": 1000, "maximum": 50000},
        },
        "required": ["command"],
    },
)
def bash_exec(command: str, cwd: str = ".", timeout_seconds: int = 30, max_output_chars: int = 12000) -> str:
    if not shell_tool_enabled():
        return "ERROR: bash_exec is disabled. Set SII_AGENT_ENABLE_SHELL_TOOL=1 and use a shell-enabled profile to expose it."
    _validate_command(command)
    timeout_seconds = _clamp_int(timeout_seconds, 30, 1, 120)
    max_output_chars = _clamp_int(max_output_chars, 12000, 1000, 50000)
    target_cwd = _safe_cwd(cwd)
    completed = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=target_cwd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout[-max_output_chars:]
    stderr = completed.stderr[-max_output_chars:]
    return json.dumps(
        {
            "cwd": str(target_cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated_from_left": len(completed.stdout) > len(stdout),
            "stderr_truncated_from_left": len(completed.stderr) > len(stderr),
        },
        ensure_ascii=False,
        indent=2,
    )
