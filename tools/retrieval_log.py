"""Per-run retrieval capture for benchmark output."""
from __future__ import annotations

from contextvars import ContextVar


_RETRIEVED_DOCIDS: ContextVar[tuple[str, ...]] = ContextVar(
    "retrieved_docids",
    default=(),
)


def reset_retrieved_docids() -> None:
    _RETRIEVED_DOCIDS.set(())


def record_retrieved_docids(docids: list[str] | tuple[str, ...]) -> None:
    if not docids:
        return
    current = list(_RETRIEVED_DOCIDS.get())
    seen = set(current)
    for docid in docids:
        docid_str = str(docid)
        if docid_str not in seen:
            current.append(docid_str)
            seen.add(docid_str)
    _RETRIEVED_DOCIDS.set(tuple(current))


def get_retrieved_docids() -> list[str]:
    return list(_RETRIEVED_DOCIDS.get())
