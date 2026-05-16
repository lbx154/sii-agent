"""Search tools — DuckDuckGo online search + local wiki25 fallback."""
from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from .registry import register
from .wiki import wiki_search

_MAX_BACKEND_WORKERS = 3


def _clamp_int(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _configured_backends() -> list[str]:
    backends = [
        b.strip().lower()
        for b in os.getenv("SEARCH_BACKENDS", "ddg,wiki").split(",")
        if b.strip()
    ]
    if os.getenv("TAVILY_API_KEY"):
        backends = ["tavily" if b == "ddg" else b for b in backends]
    return list(dict.fromkeys(backends))


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
    k = _clamp_int(k, 5, 1, 10)
    backends = _configured_backends()
    sections: list[str] = []

    if len(backends) <= 1:
        results = {
            backend: _cached_backend_search(backend, query, k)
            for backend in backends
        }
    else:
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(_MAX_BACKEND_WORKERS, len(backends))) as pool:
            futures = {
                pool.submit(_cached_backend_search, backend, query, k): backend
                for backend in backends
            }
            for future in as_completed(futures):
                backend = futures[future]
                results[backend] = future.result()

    for backend in backends:
        result = results.get(backend, "")
        if result and result != "(no results)":
            sections.append(f"## {backend}\n{result}")

    return "\n\n".join(sections) if sections else "(no results)"


@register(
    "image_search",
    "Search web images and return JSON image/page results. Use when visual evidence, image sources, "
    "or image-to-page lookup may help answer a question.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "image search query"},
            "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    },
)
def image_search(query: str, k: int = 5) -> str:
    k = _clamp_int(k, 5, 1, 20)
    return _guarded_search("DuckDuckGo image", lambda: _ddg_images(query, k))


@lru_cache(maxsize=256)
def _cached_backend_search(backend: str, query: str, k: int) -> str:
    if backend == "tavily":
        return _guarded_search("Tavily", lambda: _tavily(query, k))
    if backend == "ddg":
        return _guarded_search("DuckDuckGo", lambda: _ddg(query, k))
    if backend == "wiki":
        return _guarded_search("Offline Wikipedia", lambda: wiki_search(query, k))
    return f"ERROR: unknown search backend '{backend}'"


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


def _ddg_images(query: str, k: int) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "ERROR: ddgs not installed"
    out = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.images(query, max_results=k), 1):
            out.append(
                {
                    "rank": i,
                    "title": r.get("title", ""),
                    "page_url": r.get("url") or r.get("href") or "",
                    "image_url": r.get("image") or r.get("thumbnail") or "",
                    "source": r.get("source", ""),
                }
            )
    return json.dumps({"query": query, "results": out}, ensure_ascii=False, indent=2) if out else "(no results)"


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
