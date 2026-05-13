"""Search tools — DuckDuckGo online search + local wiki25 fallback."""
from __future__ import annotations
import os
from .registry import register
from .wiki import wiki_search


@register(
    "web_search",
    "Search the web and return top results (title, url, snippet). "
    "Use for any factual / up-to-date question.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search query"},
            "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
)
def web_search(query: str, k: int = 5) -> str:
    backends = [
        b.strip().lower()
        for b in os.getenv("SEARCH_BACKENDS", "ddg,wiki").split(",")
        if b.strip()
    ]
    sections: list[str] = []

    if os.getenv("TAVILY_API_KEY"):
        backends = ["tavily" if b == "ddg" else b for b in backends]

    for backend in backends:
        if backend == "tavily":
            result = _guarded_search("Tavily", lambda: _tavily(query, k))
        elif backend == "ddg":
            result = _guarded_search("DuckDuckGo", lambda: _ddg(query, k))
        elif backend == "wiki":
            result = _guarded_search("Offline Wikipedia", lambda: wiki_search(query, k))
        else:
            result = f"ERROR: unknown search backend '{backend}'"
        if result and result != "(no results)":
            sections.append(f"## {backend}\n{result}")

    return "\n\n".join(sections) if sections else "(no results)"


def _guarded_search(label: str, fn) -> str:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {label} search failed: {type(e).__name__}: {e}"


def _ddg(query: str, k: int) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "ERROR: ddgs not installed"
    out = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=k), 1):
            out.append(f"[{i}] {r.get('title','')}\n    {r.get('href','')}\n    {r.get('body','')[:300]}")
    return "\n".join(out) if out else "(no results)"


def _tavily(query: str, k: int) -> str:
    import httpx
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": k},
        timeout=20,
    )
    r.raise_for_status()
    items = r.json().get("results", [])
    return "\n".join(
        f"[{i}] {it['title']}\n    {it['url']}\n    {it.get('content','')[:300]}"
        for i, it in enumerate(items, 1)
    ) or "(no results)"
