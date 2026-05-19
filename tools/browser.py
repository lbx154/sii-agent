"""Browser tools backed by Playwright sessions."""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv

from .registry import register

load_dotenv()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
_PLAYWRIGHT_INSTALL_HINT = "Run `python -m playwright install chromium` to enable browser_* tools."
_BROWSER_MAX_SESSIONS = 4
_BROWSER_IDLE_SECONDS = 15 * 60
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "both", "but", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "why", "with", "you", "your", "answer", "answering", "candidate", "context",
    "first", "paragraphs", "provided", "question", "ranked", "relevance", "use",
    "using", "wiki", "wikipedia",
}
_TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿĀ-žḀ-ỿ]+(?:['’.-][\wÀ-ÖØ-öø-ÿĀ-žḀ-ỿ]+)*", re.UNICODE)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _default_extract_chars() -> int:
    try:
        value = int(os.getenv("BROWSER_EXTRACT_MAX_CHARS", "12000"))
    except ValueError:
        value = 12000
    return _clamp(value, 1000, 50000)


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _canonical_extract_query(query: str) -> str:
    query = str(query or "").strip()
    if not query:
        return ""
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*)", query, flags=re.IGNORECASE | re.DOTALL)
    if match:
        query = match.group(1)
    query = re.split(
        r"\n\s*\n|\s+Use the relevance-ranked\b|\s+Use the relevant\b|\s+Provided context\b",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _normalize_text(query)[:1000]


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query.lower()):
        token = raw.strip("'’.-_")
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens[:48]


def _query_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    for size in (4, 3, 2):
        for i in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[i:i + size])
            if len(phrase) >= 8:
                phrases.append(phrase)
    return phrases[:32]


def _chunk_text(text: str, chunk_chars: int = 900, overlap: int = 180) -> list[tuple[int, str]]:
    if len(text) <= chunk_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_chars]
        if chunk:
            chunks.append((start, chunk))
        if start + chunk_chars >= len(text):
            break
    return chunks


def _score_chunk(chunk: str, tokens: list[str], phrases: list[str]) -> int:
    text = chunk.lower()
    score = 0
    for token in tokens:
        count = text.count(token)
        if count:
            score += min(count, 3) * (3 if len(token) >= 6 else 1)
    for phrase in phrases:
        if phrase in text:
            score += 10
    return score


def _extract_relevant_text(text: str, query: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    budget = max_chars if max_chars > 0 else _default_extract_chars()
    if len(text) <= budget:
        return text, {
            "mode": "full_text_short_enough",
            "selected_chunks": [{"start": 0, "end": len(text), "score": None}],
        }

    tokens = _query_tokens(query)
    if not tokens:
        return text[:budget], {
            "mode": "head_fallback_no_query_terms",
            "selected_chunks": [{"start": 0, "end": min(len(text), budget), "score": None}],
        }

    phrases = _query_phrases(tokens)
    scored = [
        (score, start, chunk)
        for start, chunk in _chunk_text(text)
        if (score := _score_chunk(chunk, tokens, phrases)) > 0
    ]
    if not scored:
        return text[:budget], {
            "mode": "head_fallback_no_relevant_match",
            "query_terms": tokens,
            "selected_chunks": [{"start": 0, "end": min(len(text), budget), "score": 0}],
        }

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, int, int, str]] = []
    used_ranges: list[tuple[int, int]] = []
    used_chars = 0
    for score, start, chunk in scored:
        end = start + len(chunk)
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue
        separator_cost = 24 if selected else 0
        if used_chars + len(chunk) + separator_cost > budget and selected:
            continue
        selected.append((start, end, score, chunk))
        used_ranges.append((start, end))
        used_chars += len(chunk) + separator_cost
        if used_chars >= budget:
            break

    selected.sort(key=lambda item: item[0])
    parts = [
        f"[excerpt {i}, chars {start}-{end}, score {score}]\n{chunk.strip()}"
        for i, (start, end, score, chunk) in enumerate(selected, 1)
    ]
    rendered = "\n\n---\n\n".join(parts)
    return rendered[:budget], {
        "mode": "query_focused_extract",
        "query_terms": tokens,
        "selected_chunks": [
            {"start": start, "end": end, "score": score}
            for start, end, score, _ in selected
        ],
    }


def _aio_base_url() -> str | None:
    base_url = (
        os.getenv("AIO_SANDBOX_BASE_URL", "")
        or os.getenv("SANDBOX_BASE_URL", "")
        or os.getenv("BROWSER_SERVICE_URL", "")
    ).strip().rstrip("/")
    return base_url or None


def _sandbox_headers() -> dict[str, str]:
    token = (
        os.getenv("SANDBOX_API_TOKEN", "")
        or os.getenv("BROWSER_API_TOKEN", "")
        or os.getenv("AIO_SANDBOX_API_TOKEN", "")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def _normalize_ws_url(cdp_url: str, base_url: str) -> str:
    parsed = urlparse(cdp_url)
    base = urlparse(base_url)
    if parsed.hostname == "0.0.0.0" and base.hostname:
        netloc = base.hostname
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))
    return cdp_url


def _aio_browser_info(base_url: str) -> dict[str, Any]:
    headers = _sandbox_headers()
    aio_error: BaseException | None = None
    try:
        response = httpx.get(f"{base_url}/v1/browser/info", headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"AIO browser info response must be an object: {payload}")
        if payload.get("success") is False:
            raise RuntimeError(payload.get("message") or "AIO browser info request failed")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if not isinstance(data, dict):
            raise RuntimeError(f"AIO browser info response missing data: {payload}")
        data = dict(data)
        data.setdefault("backend", "aio-sandbox")
    except Exception as exc:  # noqa: BLE001
        aio_error = exc
        try:
            response = httpx.get(f"{base_url}/browser/cdp_url", headers=headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"browser-service cdp response must be an object: {payload}")
            if payload.get("success") is False:
                raise RuntimeError(payload.get("message") or "browser-service cdp request failed")
            data = dict(payload)
            data["backend"] = "browser-service"
        except Exception as browser_service_error:  # noqa: BLE001
            raise RuntimeError(
                "remote browser discovery failed: "
                f"AIO endpoint error={type(aio_error).__name__}: {aio_error}; "
                "browser-service endpoint error="
                f"{type(browser_service_error).__name__}: {browser_service_error}"
            ) from browser_service_error
    cdp_url = data.get("cdp_url")
    if not cdp_url:
        raise RuntimeError(f"remote browser response missing cdp_url: {payload}")
    data["cdp_url"] = _normalize_ws_url(str(cdp_url), base_url)
    return data


def _format_playwright_error(exc: BaseException) -> str:
    message = str(exc)
    if "Executable doesn't exist" in message or "Looks like Playwright" in message:
        return f"ERROR: Chromium for Playwright is not installed. {_PLAYWRIGHT_INSTALL_HINT}"
    return f"ERROR: browser tool failed: {type(exc).__name__}: {message}"


def _wait_until(wait_until: str) -> str:
    allowed = {"commit", "domcontentloaded", "load", "networkidle"}
    return wait_until if wait_until in allowed else "domcontentloaded"


def _snapshot(
    page: Any,
    max_chars: int = 0,
    max_links: int = 20,
    max_controls: int = 20,
    extract_query: str = "",
) -> str:
    max_chars = max(0, int(max_chars))
    max_links = _clamp(max_links, 0, 100)
    max_controls = _clamp(max_controls, 0, 100)
    text = _normalize_text(page.locator("body").inner_text(timeout=5000) if page.locator("body").count() else "")
    links = page.evaluate(
        """(limit) => Array.from(document.querySelectorAll('a')).slice(0, limit).map((a) => ({
            text: (a.innerText || a.getAttribute('aria-label') || '').trim().slice(0, 160),
            href: a.href || a.getAttribute('href') || ''
        })).filter((a) => a.text || a.href)""",
        max_links,
    )
    controls = page.evaluate(
        """(limit) => Array.from(document.querySelectorAll('input, textarea, select, button')).slice(0, limit).map((el) => ({
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || '',
            name: el.getAttribute('name') || '',
            id: el.id || '',
            placeholder: el.getAttribute('placeholder') || '',
            value: el.value || '',
            text: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 160)
        }))""",
        max_controls,
    )
    focused_query = _canonical_extract_query(extract_query)
    if focused_query:
        rendered_text, extract_meta = _extract_relevant_text(text, focused_query, max_chars)
    else:
        rendered_text = text[:max_chars] if max_chars else text
        extract_meta = {
            "mode": "full_text" if not max_chars or len(text) <= max_chars else "manual_truncate",
            "selected_chunks": [{"start": 0, "end": len(rendered_text), "score": None}],
        }

    return _json({
        "title": page.title(),
        "url": page.url,
        "text": rendered_text,
        "text_mode": extract_meta["mode"],
        "extract_query": focused_query,
        "full_text_chars": len(text),
        "returned_text_chars": len(rendered_text),
        "omitted_text_chars": max(0, len(text) - len(rendered_text)),
        "query_terms": extract_meta.get("query_terms", []),
        "selected_chunks": extract_meta.get("selected_chunks", []),
        "links": links,
        "controls": controls,
    })


def _looks_like_text_target(target: str) -> bool:
    return any(ch.isspace() for ch in target) or target.startswith(("text=", "label="))


def _css_attr_equals(attr: str, value: str) -> str:
    return f"[{attr}={json.dumps(value)}]"


def _resolve_locator(page: Any, target: str, by: str = "auto") -> Any:
    mode = by.lower().strip()
    if target.startswith("css="):
        return page.locator(target[4:]).first
    if target.startswith(("text=", "label=")):
        return page.get_by_text(target.split("=", 1)[1], exact=False).first
    if mode == "css":
        return page.locator(target).first
    if mode in {"text", "label"}:
        return page.get_by_text(target, exact=False).first
    if mode != "auto":
        raise ValueError("by must be one of: auto, css, text")

    if _looks_like_text_target(target):
        text_locator = page.get_by_text(target.removeprefix("text=").removeprefix("label="), exact=False)
        if text_locator.count() > 0:
            return text_locator.first

    try:
        css_locator = page.locator(target)
        if css_locator.count() > 0:
            return css_locator.first
    except Exception:  # noqa: BLE001
        pass
    return page.get_by_text(target, exact=False).first


def _resolve_fill_locator(page: Any, target: str, by: str = "auto") -> Any:
    mode = by.lower().strip()
    if target.startswith("css="):
        return page.locator(target[4:]).first
    if target.startswith("label="):
        return page.get_by_label(target.split("=", 1)[1], exact=False).first
    if target.startswith("placeholder="):
        return page.get_by_placeholder(target.split("=", 1)[1], exact=False).first
    if target.startswith("name="):
        return page.locator(_css_attr_equals("name", target.split("=", 1)[1])).first
    if mode == "css":
        return page.locator(target).first
    if mode == "label":
        return page.get_by_label(target, exact=False).first
    if mode == "placeholder":
        return page.get_by_placeholder(target, exact=False).first
    if mode == "name":
        return page.locator(_css_attr_equals("name", target)).first
    if mode == "text":
        return page.get_by_text(target, exact=False).first
    if mode != "auto":
        raise ValueError("by must be one of: auto, css, text, label, placeholder, name")

    candidates = (
        lambda: page.get_by_label(target, exact=False),
        lambda: page.get_by_placeholder(target, exact=False),
        lambda: page.locator(_css_attr_equals("name", target)),
        lambda: page.locator(target),
        lambda: page.get_by_text(target, exact=False),
    )
    for candidate in candidates:
        try:
            locator = candidate()
            if locator.count() > 0:
                return locator.first
        except Exception:  # noqa: BLE001
            continue
    return page.get_by_label(target, exact=False).first


class _BrowserSession:
    """Owns one Playwright page on one thread; sync Playwright is not thread-safe."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.last_used = time.monotonic()
        self.closed = False
        self._ready = threading.Event()
        self._init_error: str | None = None
        self._jobs: queue.Queue[tuple[Callable[[Any], str] | None, queue.Queue[str]]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name=f"browser-session-{session_id[:32]}",
            daemon=True,
        )
        self._thread.start()

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def run(self, action: Callable[[Any], str], timeout_s: float) -> str:
        self.last_used = time.monotonic()
        if not self._ready.wait(timeout_s):
            return "ERROR: timed out starting browser session"
        if self._init_error:
            return self._init_error
        if self.closed:
            return f"ERROR: browser session '{self.session_id}' is closed"
        result_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._jobs.put((action, result_queue))
        try:
            return result_queue.get(timeout=timeout_s)
        except queue.Empty:
            return f"ERROR: timed out waiting for browser session '{self.session_id}'"

    def close(self, timeout_s: float = 10) -> str:
        self.closed = True
        if not self._ready.wait(timeout_s) or self._init_error:
            return f"OK: browser session '{self.session_id}' closed"
        result_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._jobs.put((None, result_queue))
        try:
            return result_queue.get(timeout=timeout_s)
        except queue.Empty:
            return f"OK: browser session '{self.session_id}' close requested"

    def _run(self) -> None:
        playwright = None
        browser = None
        context = None
        page = None
        using_aio = False
        owns_context = True
        try:
            from playwright.sync_api import sync_playwright

            playwright = sync_playwright().start()
            aio_url = _aio_base_url()
            if aio_url:
                using_aio = True
                info = _aio_browser_info(aio_url)
                browser = playwright.chromium.connect_over_cdp(info["cdp_url"])
                if browser.contexts:
                    context = browser.contexts[0]
                    owns_context = False
                else:
                    context = browser.new_context(
                        ignore_https_errors=True,
                        viewport={"width": 1366, "height": 768},
                        user_agent=str(info.get("user_agent") or _HEADERS["User-Agent"]),
                    )
            else:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context = browser.new_context(
                    ignore_https_errors=True,
                    viewport={"width": 1366, "height": 768},
                    user_agent=_HEADERS["User-Agent"],
                )
            page = context.new_page()
            self._ready.set()
            while True:
                action, result_queue = self._jobs.get()
                if action is None:
                    result_queue.put(f"OK: browser session '{self.session_id}' closed")
                    break
                try:
                    result_queue.put(action(page))
                except Exception as exc:  # noqa: BLE001
                    result_queue.put(_format_playwright_error(exc))
        except ImportError:
            self._init_error = f"ERROR: playwright package is not installed. {_PLAYWRIGHT_INSTALL_HINT}"
            self._ready.set()
        except Exception as exc:  # noqa: BLE001
            self._init_error = _format_playwright_error(exc)
            self._ready.set()
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass
            if context is not None and owns_context:
                try:
                    context.close()
                except Exception:  # noqa: BLE001
                    pass
            if browser is not None and not using_aio:
                try:
                    browser.close()
                except Exception:  # noqa: BLE001
                    pass
            if playwright is not None:
                try:
                    playwright.stop()
                except Exception:  # noqa: BLE001
                    pass
            self.closed = True


class _BrowserManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _BrowserSession] = {}

    def run(self, session_id: str, action: Callable[[Any], str], timeout_s: float) -> str:
        session_id = session_id or "default"
        to_close: list[_BrowserSession] = []
        with self._lock:
            now = time.monotonic()
            for key, session in list(self._sessions.items()):
                if session.closed or now - session.last_used > _BROWSER_IDLE_SECONDS:
                    self._sessions.pop(key, None)
                    to_close.append(session)
            session = self._sessions.get(session_id)
            if session is None or session.closed:
                while len(self._sessions) >= _BROWSER_MAX_SESSIONS:
                    key, old_session = min(self._sessions.items(), key=lambda item: item[1].last_used)
                    self._sessions.pop(key, None)
                    to_close.append(old_session)
                session = _BrowserSession(session_id)
                self._sessions[session_id] = session
        for session_to_close in to_close:
            session_to_close.close()
        result = session.run(action, timeout_s)
        if session.init_error:
            with self._lock:
                if self._sessions.get(session_id) is session:
                    self._sessions.pop(session_id, None)
        return result

    def close(self, session_id: str) -> str:
        session_id = session_id or "default"
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return f"OK: browser session '{session_id}' was not open"
        return session.close()


_BROWSER_MANAGER = _BrowserManager()


@register(
    "aio_sandbox_status",
    "Check whether a remote sandbox/browser-service backend is configured and reachable.",
    {
        "type": "object",
        "properties": {},
    },
)
def aio_sandbox_status() -> str:
    base_url = _aio_base_url()
    if not base_url:
        return _json(
            {
                "configured": False,
                "message": "Set SANDBOX_BASE_URL=http://127.0.0.1:8080 for browser-service, or AIO_SANDBOX_BASE_URL for All-in-One Sandbox.",
            }
        )
    try:
        info = _aio_browser_info(base_url)
    except Exception as exc:  # noqa: BLE001
        return _json(
            {
                "configured": True,
                "base_url": base_url,
                "reachable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return _json(
        {
            "configured": True,
            "base_url": base_url,
            "reachable": True,
            "backend": info.get("backend"),
            "cdp_url": info.get("cdp_url"),
            "vnc_url": info.get("vnc_url"),
            "viewport": info.get("viewport"),
            "page_viewport": info.get("page_viewport"),
        }
    )


@register(
    "browser_open",
    "Open an http(s) URL in a persistent sandbox browser session and return JSON text/links/forms. "
    "When extract_query is provided, the full page is read first and only relevant excerpts are returned.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "session_id": {"type": "string", "default": "default"},
            "wait_until": {
                "type": "string",
                "default": "domcontentloaded",
                "enum": ["commit", "domcontentloaded", "load", "networkidle"],
            },
            "timeout_ms": {"type": "integer", "default": 30000, "minimum": 1000, "maximum": 120000},
            "max_chars": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Maximum returned text characters; 0 uses full text unless extract_query is set, then uses the default extraction budget.",
            },
            "extract_query": {
                "type": "string",
                "default": "",
                "description": "Optional question/query used to return only relevant excerpts from the full page text.",
            },
        },
        "required": ["url"],
    },
)
def browser_open(
    url: str,
    session_id: str = "default",
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 30000,
    max_chars: int = 0,
    extract_query: str = "",
) -> str:
    if not _is_http_url(url):
        return "ERROR: url must be an http(s) URL"
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))

    def action(page: Any) -> str:
        page.goto(url, wait_until=_wait_until(wait_until), timeout=timeout_ms)
        return _snapshot(page, max_chars=max_chars, extract_query=extract_query)

    return _BROWSER_MANAGER.run(session_id, action, timeout_ms / 1000 + 5)


@register(
    "browser_open_many",
    "Open up to four http(s) URLs concurrently in separate sandbox browser sessions and return JSON snapshots. "
    "Use when several independent pages need to be read in parallel. When extract_query is provided, each full page "
    "is read first and only relevant excerpts are returned.",
    {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "session_prefix": {"type": "string", "default": "bulk"},
            "wait_until": {
                "type": "string",
                "default": "domcontentloaded",
                "enum": ["commit", "domcontentloaded", "load", "networkidle"],
            },
            "timeout_ms": {"type": "integer", "default": 30000, "minimum": 1000, "maximum": 120000},
            "max_chars": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Maximum returned text characters per page; 0 uses full text unless extract_query is set, then uses the default extraction budget.",
            },
            "extract_query": {
                "type": "string",
                "default": "",
                "description": "Optional shared question/query used to return only relevant excerpts from each full page.",
            },
            "concurrency": {"type": "integer", "default": 4, "minimum": 1, "maximum": 4},
        },
        "required": ["urls"],
    },
)
def browser_open_many(
    urls: list[str],
    session_prefix: str = "bulk",
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 30000,
    max_chars: int = 0,
    extract_query: str = "",
    concurrency: int = 4,
) -> str:
    if not urls:
        return "ERROR: urls must contain at least one URL"
    limited_urls = urls[:_BROWSER_MAX_SESSIONS]
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))
    concurrency = _clamp(concurrency, 1, _BROWSER_MAX_SESSIONS)
    prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_prefix) or "bulk"
    call_id = uuid.uuid4().hex[:8]
    results: list[dict[str, Any] | None] = [None] * len(limited_urls)

    def open_one(index: int, url: str) -> dict[str, Any]:
        session_id = f"{prefix}_{call_id}_{index}"
        raw = browser_open(
            url,
            session_id=session_id,
            wait_until=wait_until,
            timeout_ms=timeout_ms,
            max_chars=max_chars,
            extract_query=extract_query,
        )
        try:
            snapshot: Any = json.loads(raw)
        except json.JSONDecodeError:
            snapshot = raw
        return {
            "index": index,
            "session_id": session_id,
            "url": url,
            "ok": not raw.startswith("ERROR:"),
            "snapshot": snapshot,
        }

    with ThreadPoolExecutor(max_workers=min(concurrency, len(limited_urls))) as pool:
        futures = {
            pool.submit(open_one, index, url): index
            for index, url in enumerate(limited_urls)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return _json({"results": results, "truncated": len(urls) > len(limited_urls)})


@register(
    "browser_text",
    "Return the current sandbox browser page as JSON with title, URL, visible text, links, and controls. "
    "When extract_query is provided, the full page is read first and only relevant excerpts are returned.",
    {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "default": "default"},
            "max_chars": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Maximum returned text characters; 0 uses full text unless extract_query is set, then uses the default extraction budget.",
            },
            "extract_query": {
                "type": "string",
                "default": "",
                "description": "Optional question/query used to return only relevant excerpts from the current page text.",
            },
            "max_links": {"type": "integer", "default": 20, "minimum": 0, "maximum": 100},
            "max_controls": {"type": "integer", "default": 20, "minimum": 0, "maximum": 100},
        },
    },
)
def browser_text(
    session_id: str = "default",
    max_chars: int = 0,
    extract_query: str = "",
    max_links: int = 20,
    max_controls: int = 20,
) -> str:
    def action(page: Any) -> str:
        return _snapshot(
            page,
            max_chars=max_chars,
            max_links=max_links,
            max_controls=max_controls,
            extract_query=extract_query,
        )

    return _BROWSER_MANAGER.run(session_id, action, 15)


@register(
    "browser_click",
    "Click a CSS selector or visible text in a sandbox browser session, then return the updated page JSON. "
    "When extract_query is provided, the full updated page is read first and only relevant excerpts are returned.",
    {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "CSS selector, visible text, or prefixed target like css=#submit / text=Sign in",
            },
            "session_id": {"type": "string", "default": "default"},
            "by": {"type": "string", "default": "auto", "enum": ["auto", "css", "text"]},
            "wait_until": {
                "type": "string",
                "default": "domcontentloaded",
                "enum": ["commit", "domcontentloaded", "load", "networkidle"],
            },
            "timeout_ms": {"type": "integer", "default": 30000, "minimum": 1000, "maximum": 120000},
            "max_chars": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Maximum returned text characters after clicking; 0 uses full text unless extract_query is set, then uses the default extraction budget.",
            },
            "extract_query": {
                "type": "string",
                "default": "",
                "description": "Optional question/query used to return only relevant excerpts after clicking.",
            },
        },
        "required": ["target"],
    },
)
def browser_click(
    target: str,
    session_id: str = "default",
    by: str = "auto",
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 30000,
    max_chars: int = 0,
    extract_query: str = "",
) -> str:
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))

    def action(page: Any) -> str:
        locator = _resolve_locator(page, target, by=by)
        locator.click(timeout=timeout_ms)
        try:
            page.wait_for_load_state(_wait_until(wait_until), timeout=min(timeout_ms, 10000))
        except Exception:  # noqa: BLE001
            pass
        return _snapshot(page, max_chars=max_chars, extract_query=extract_query)

    return _BROWSER_MANAGER.run(session_id, action, timeout_ms / 1000 + 5)


@register(
    "browser_type",
    "Fill an input/textarea/contenteditable target in a sandbox browser session, optionally pressing Enter. "
    "Targets can be css=..., label=..., placeholder=..., name=..., visible text, or auto-resolved. "
    "When extract_query is provided, the full updated page is read first and only relevant excerpts are returned.",
    {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "CSS selector, visible text, or prefixed target like css=input[name=q] / text=Search",
            },
            "text": {"type": "string"},
            "session_id": {"type": "string", "default": "default"},
            "by": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "css", "text", "label", "placeholder", "name"],
            },
            "submit": {"type": "boolean", "default": False},
            "timeout_ms": {"type": "integer", "default": 30000, "minimum": 1000, "maximum": 120000},
            "max_chars": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Maximum returned text characters after typing; 0 uses full text unless extract_query is set, then uses the default extraction budget.",
            },
            "extract_query": {
                "type": "string",
                "default": "",
                "description": "Optional question/query used to return only relevant excerpts after typing.",
            },
        },
        "required": ["target", "text"],
    },
)
def browser_type(
    target: str,
    text: str,
    session_id: str = "default",
    by: str = "auto",
    submit: bool = False,
    timeout_ms: int = 30000,
    max_chars: int = 0,
    extract_query: str = "",
) -> str:
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))

    def action(page: Any) -> str:
        locator = _resolve_fill_locator(page, target, by=by)
        locator.fill(text, timeout=timeout_ms)
        if submit:
            page.keyboard.press("Enter")
            try:
                page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 10000))
            except Exception:  # noqa: BLE001
                pass
        return _snapshot(page, max_chars=max_chars, extract_query=extract_query)

    return _BROWSER_MANAGER.run(session_id, action, timeout_ms / 1000 + 5)


@register(
    "browser_close",
    "Close a sandbox browser session and free its Chromium resources.",
    {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "default": "default"},
        },
    },
)
def browser_close(session_id: str = "default") -> str:
    return _BROWSER_MANAGER.close(session_id)
