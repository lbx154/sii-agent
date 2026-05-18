"""Tool registry — OpenAI function-calling spec + dispatcher."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Callable

TOOL_REGISTRY: dict[str, dict] = {}


def register(name: str, description: str, parameters: dict):
    def deco(fn: Callable):
        TOOL_REGISTRY[name] = {
            "fn": fn,
            "spec": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return fn
    return deco


def tool_specs(names: Iterable[str] | None = None) -> list[dict]:
    allowed = set(names) if names is not None else None
    return [
        t["spec"]
        for name, t in TOOL_REGISTRY.items()
        if allowed is None or name in allowed
    ]


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    if name not in TOOL_REGISTRY:
        return f"ERROR: unknown tool '{name}'. Available: {list(TOOL_REGISTRY)}"
    try:
        result = TOOL_REGISTRY[name]["fn"](**arguments)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR while running {name}: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else str(result)


# import side-effects: register built-in tools
from . import wiki, search, browser, vision, browsecomp, memory_tools, verify, shell, finalize  # noqa: E402,F401
