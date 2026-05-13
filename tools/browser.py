"""Lightweight browser tool — fetch + extract main text. Playwright path is optional."""
from __future__ import annotations
import httpx
from bs4 import BeautifulSoup
from .registry import register

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


@register(
    "browse",
    "Fetch a URL and return its main readable text (truncated). "
    "Use after web_search to read a specific page.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000, "minimum": 200, "maximum": 20000},
        },
        "required": ["url"],
    },
)
def browse(url: str, max_chars: int = 4000) -> str:
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"ERROR fetching {url}: {e}"
    soup = BeautifulSoup(r.text, "lxml")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    text = " ".join(soup.get_text(" ").split())
    title = (soup.title.string or "").strip() if soup.title else ""
    return f"TITLE: {title}\nURL: {url}\n\n{text[:max_chars]}"
