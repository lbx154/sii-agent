"""Per-agent-run context visible to tools during dispatch."""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_TOOL_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("sii_tool_context", default={})


def set_tool_context(**kwargs: Any) -> None:
    current = dict(_TOOL_CONTEXT.get() or {})
    current.update(kwargs)
    _TOOL_CONTEXT.set(current)


def get_tool_context() -> dict[str, Any]:
    return dict(_TOOL_CONTEXT.get() or {})
