"""
Trajectory recording and replay for the Kimi Agent Harness.

Each task produces a JSONL file where every line is one interaction turn,
tagged with role, timestamp, step_id, and optional tool metadata.
"""

import json
import time
from pathlib import Path
from typing import Optional

from roles import Role


class Trajectory:
    """
    Append-only JSONL trajectory store.

    File layout (one JSON object per line):
    {
        "timestamp":    float,          # unix epoch
        "step_id":      int | None,     # agent loop step number
        "role":         str,            # Role enum value
        "content":      str | dict,     # message content
        "tool_call_id": str | None,     # links tool result to tool call
        ...extra fields...
    }
    """

    def __init__(self, task_id: str, output_dir: str = "trajectories"):
        self.task_id = task_id
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.path = output_path / f"{task_id}.jsonl"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        role: Role,
        content,
        step_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Append one turn to the trajectory file."""
        entry = {
            "timestamp":    time.time(),
            "step_id":      step_id,
            "role":         role.value,
            "content":      content,
            "tool_call_id": tool_call_id,
        }
        if extra:
            entry.update(extra)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict]:
        """Return all recorded turns as a list of dicts."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def to_messages(self) -> list[dict]:
        """
        Convert trajectory to OpenAI-compatible messages list.

        - tool_calls field is re-attached to assistant turns when present
        """
        messages = []
        for entry in self.read_all():
            role = entry["role"]

            msg: dict = {"role": role, "content": entry["content"] or ""}

            # Re-attach tool_calls list so the LLM can continue the loop
            if role == "assistant" and entry.get("tool_calls"):
                msg["tool_calls"] = entry["tool_calls"]

            # Required by OpenAI spec for tool-result messages
            if entry.get("tool_call_id"):
                msg["tool_call_id"] = entry["tool_call_id"]

            messages.append(msg)
        return messages

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return high-level stats about this trajectory."""
        entries = self.read_all()
        role_counts: dict[str, int] = {}
        for e in entries:
            r = e["role"]
            role_counts[r] = role_counts.get(r, 0) + 1
        return {
            "task_id":    self.task_id,
            "total_turns": len(entries),
            "role_counts": role_counts,
            "path":        str(self.path),
        }

    def export_json(self) -> str:
        """Export the full trajectory as a pretty-printed JSON string."""
        return json.dumps(self.read_all(), ensure_ascii=False, indent=2)
