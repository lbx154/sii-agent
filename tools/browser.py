"""Browser tools backed by Playwright sessions."""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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


def _normalize_browser_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return value
    if value.startswith(("http://", "https://", "file://", "data:")) or value in {"about:blank", "about:newtab"}:
        return value
    return f"https://{value}"


def _looks_like_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or path.startswith("/pdf/") or "/pdf/" in path


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
            "mode": "head_extract_no_query_terms",
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
            "mode": "head_extract_no_relevant_match",
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


def _snapshot_from_text(
    *,
    title: str,
    url: str,
    text: str,
    max_chars: int = 0,
    extract_query: str = "",
    links: list[dict[str, Any]] | None = None,
    controls: list[dict[str, Any]] | None = None,
    source_type: str = "html",
) -> str:
    focused_query = _canonical_extract_query(extract_query)
    if focused_query:
        rendered_text, extract_meta = _extract_relevant_text(text, focused_query, max_chars)
    else:
        rendered_text = text[:max_chars] if max_chars else text
        extract_meta = {
            "mode": "full_text" if not max_chars or len(text) <= max_chars else "manual_truncate",
            "selected_chunks": [{"start": 0, "end": len(rendered_text), "score": None}],
        }
    return _json(
        {
            "title": title,
            "url": url,
            "source_type": source_type,
            "text": rendered_text,
            "text_mode": extract_meta["mode"],
            "extract_query": focused_query,
            "full_text_chars": len(text),
            "returned_text_chars": len(rendered_text),
            "omitted_text_chars": max(0, len(text) - len(rendered_text)),
            "query_terms": extract_meta.get("query_terms", []),
            "selected_chunks": extract_meta.get("selected_chunks", []),
            "links": links or [],
            "controls": controls or [],
        }
    )


def _pdf_snapshot(url: str, *, max_chars: int = 0, extract_query: str = "") -> str:
    timeout = float(os.getenv("BROWSER_PDF_TIMEOUT", "60"))
    max_bytes = int(os.getenv("BROWSER_PDF_MAX_BYTES", str(50 * 1024 * 1024)))
    response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    content_length = int(response.headers.get("content-length") or 0)
    if content_length > max_bytes:
        raise RuntimeError(f"PDF is too large: {content_length} bytes")
    data = response.content
    if len(data) > max_bytes:
        raise RuntimeError(f"PDF is too large: {len(data)} bytes")
    if "pdf" not in content_type and not data.startswith(b"%PDF"):
        raise RuntimeError(f"URL does not look like a PDF after fetch: content-type={content_type!r}")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(data)
        pdf_file.flush()
        proc = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", pdf_file.name, "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "pdftotext failed").strip())
    text = _normalize_text(proc.stdout)
    title = Path(urlparse(str(response.url)).path).name or "PDF"
    return _snapshot_from_text(
        title=title,
        url=str(response.url),
        text=text,
        max_chars=max_chars,
        extract_query=extract_query,
        source_type="pdf",
    )


_SERVICE_SESSION_LOCK = threading.Lock()
_SERVICE_SESSIONS: dict[str, str] = {}


def _browser_service_url() -> str:
    base_url = _aio_base_url()
    if not base_url:
        raise RuntimeError(
            "remote browser backend is required; set SANDBOX_BASE_URL, "
            "BROWSER_SERVICE_URL, or AIO_SANDBOX_BASE_URL."
        )
    return base_url


def _browser_service_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    base_url = _browser_service_url()
    response = httpx.request(
        method,
        f"{base_url}{path}",
        headers=_sandbox_headers(),
        json=json_body,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"browser-service returned non-object JSON: {payload!r}")
    if payload.get("success") is False:
        raise RuntimeError(str(payload.get("message") or "browser-service request failed"))
    return payload


def _service_session_id(logical_session_id: str) -> str:
    logical_session_id = logical_session_id or "default"
    with _SERVICE_SESSION_LOCK:
        cached = _SERVICE_SESSIONS.get(logical_session_id)
        if cached:
            return cached
        payload = _browser_service_request("POST", "/session/create", timeout=30)
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            raise RuntimeError(f"browser-service create_session returned empty session_id: {payload}")
        _SERVICE_SESSIONS[logical_session_id] = session_id
        return session_id


def _service_get_text(session_id: str, tab_id: str | None = None, selector: str | None = None) -> str:
    payload = _browser_service_request(
        "POST",
        "/browser/get_text",
        json_body={"session_id": session_id, "tab_id": tab_id, "selector": selector},
    )
    return _normalize_text(str(payload.get("text") or ""))


def _service_title(session_id: str, tab_id: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"session_id": session_id}
    if tab_id:
        params["tab_id"] = tab_id
    return _browser_service_request("GET", "/browser/title", params=params)


def _service_eval(session_id: str, script: str, tab_id: str | None = None) -> Any:
    payload = _browser_service_request(
        "POST",
        "/browser/eval",
        json_body={"session_id": session_id, "tab_id": tab_id, "script": script},
    )
    return payload.get("result")


def _service_snapshot(
    session_id: str,
    *,
    tab_id: str | None = None,
    max_chars: int = 0,
    max_links: int = 20,
    max_controls: int = 20,
    extract_query: str = "",
) -> str:
    max_chars = max(0, int(max_chars))
    max_links = _clamp(max_links, 0, 100)
    max_controls = _clamp(max_controls, 0, 100)
    meta = _service_title(session_id, tab_id=tab_id)
    text = _service_get_text(session_id, tab_id=tab_id)
    links = _service_eval(
        session_id,
        f"""() => Array.from(document.querySelectorAll('a')).slice(0, {max_links}).map((a) => ({{
            text: (a.innerText || a.getAttribute('aria-label') || '').trim().slice(0, 160),
            href: a.href || a.getAttribute('href') || ''
        }})).filter((a) => a.text || a.href)""",
        tab_id=tab_id,
    )
    controls = _service_eval(
        session_id,
        f"""() => Array.from(document.querySelectorAll('input, textarea, select, button'))
        .filter((el) => {{
            const style = window.getComputedStyle(el);
            return el.type !== 'hidden' && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        }})
        .slice(0, {max_controls}).map((el) => ({{
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || '',
            name: el.getAttribute('name') || '',
            id: el.id || '',
            placeholder: el.getAttribute('placeholder') || '',
            value: (el.value || '').slice(0, 160),
            text: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 160)
        }}))""",
        tab_id=tab_id,
    )
    return _snapshot_from_text(
        title=str(meta.get("title", "")),
        url=str(meta.get("url", "")),
        text=text,
        max_chars=max_chars,
        extract_query=extract_query,
        links=links if isinstance(links, list) else [],
        controls=controls if isinstance(controls, list) else [],
    )


def _service_error(exc: BaseException) -> str:
    return f"ERROR: browser-service request failed: {type(exc).__name__}: {exc}"


def _target_to_selector(target: str, by: str = "auto") -> str | None:
    value = str(target or "").strip()
    mode = by.lower().strip()
    if value.startswith("css="):
        return value[4:]
    if value.startswith("name="):
        return _css_attr_equals("name", value.split("=", 1)[1])
    if mode == "css":
        return value
    if mode == "name":
        return _css_attr_equals("name", value)
    if mode == "auto":
        if value.startswith("#") or value.startswith(".") or value.startswith("["):
            return value
        if re.match(r"^[A-Za-z][A-Za-z0-9_-]*(?:[#.\\[]|$)", value):
            return value
    return None


def _js_literal(value: str) -> str:
    return json.dumps(value)


def _service_click_target(session_id: str, target: str, by: str, tab_id: str | None = None) -> None:
    selector = _target_to_selector(target, by)
    if selector:
        _browser_service_request(
            "POST",
            "/browser/click",
            json_body={"session_id": session_id, "tab_id": tab_id, "selector": selector, "timeout_ms": 10000},
            timeout=20,
        )
        return
    text = str(target or "").removeprefix("text=").removeprefix("label=").strip()
    script = (
        "() => {"
        f" const needle = {_js_literal(text)}.toLowerCase();"
        " const els = Array.from(document.querySelectorAll('a,button,input,[role=button],label,summary'));"
        " const el = els.find((node) => ((node.innerText || node.value || node.getAttribute('aria-label') || '').trim().toLowerCase()).includes(needle));"
        " if (!el) return false;"
        " el.click(); return true;"
        "}"
    )
    if not _service_eval(session_id, script, tab_id=tab_id):
        raise RuntimeError(f"target {target!r} was not found")


def _service_type_target(
    session_id: str,
    target: str,
    text: str,
    by: str,
    submit: bool,
    tab_id: str | None = None,
) -> None:
    selector = _target_to_selector(target, by)
    if not selector:
        if str(target).startswith("placeholder="):
            selector = _css_attr_equals("placeholder", str(target).split("=", 1)[1])
        elif str(target).startswith("label="):
            label = str(target).split("=", 1)[1]
            selector = _service_eval(
                session_id,
                "() => {"
                f" const needle = {_js_literal(label)}.toLowerCase();"
                " const label = Array.from(document.querySelectorAll('label')).find((el) => (el.innerText || '').toLowerCase().includes(needle));"
                " if (!label) return '';"
                " if (label.htmlFor) return '#' + CSS.escape(label.htmlFor);"
                " const input = label.querySelector('input,textarea,[contenteditable=true]');"
                " if (!input) return '';"
                " if (!input.id) input.id = 'sii-browser-input-' + Math.random().toString(36).slice(2);"
                " return '#' + CSS.escape(input.id);"
                "}",
                tab_id=tab_id,
            )
        elif by.lower().strip() == "placeholder":
            selector = _css_attr_equals("placeholder", target)
        elif by.lower().strip() == "label":
            selector = None
    if not selector:
        raise RuntimeError(f"input target {target!r} could not be resolved to a CSS selector")
    _browser_service_request(
        "POST",
        "/browser/type",
        json_body={
            "session_id": session_id,
            "tab_id": tab_id,
            "selector": selector,
            "text": str(text),
            "clear": True,
            "press_enter": bool(submit),
            "delay_ms": 20,
        },
        timeout=30,
    )


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
        """(limit) => Array.from(document.querySelectorAll('input, textarea, select, button'))
        .filter((el) => {
            const style = window.getComputedStyle(el);
            return el.type !== 'hidden' && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        })
        .slice(0, limit).map((el) => ({
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || '',
            name: el.getAttribute('name') || '',
            id: el.id || '',
            placeholder: el.getAttribute('placeholder') || '',
            value: (el.value || '').slice(0, 160),
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
            if not aio_url:
                raise RuntimeError(
                    "remote browser backend is required; set SANDBOX_BASE_URL, "
                    "BROWSER_SERVICE_URL, or AIO_SANDBOX_BASE_URL."
                )
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
    "Open a URL in a persistent browser-service sandbox session and return JSON text/links/forms. "
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
    url = _normalize_browser_url(url)
    if not url:
        return "ERROR: url must not be empty"
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))
    try:
        pdf_error: BaseException | None = None
        if _looks_like_pdf_url(url):
            try:
                return _pdf_snapshot(url, max_chars=max_chars, extract_query=extract_query)
            except Exception as exc:  # noqa: BLE001
                pdf_error = exc
        service_session_id = _service_session_id(session_id)
        try:
            _browser_service_request(
                "POST",
                "/browser/navigate",
                json_body={
                    "session_id": service_session_id,
                    "url": url,
                    "wait_until": _wait_until(wait_until),
                    "timeout_ms": timeout_ms,
                },
                timeout=timeout_ms / 1000 + 15,
            )
        except Exception as exc:  # noqa: BLE001
            if "navigation started a download" in str(exc):
                try:
                    return _pdf_snapshot(url, max_chars=max_chars, extract_query=extract_query)
                except Exception as pdf_exc:  # noqa: BLE001
                    raise RuntimeError(f"{exc}; PDF text extraction failed: {pdf_exc}") from pdf_exc
            if pdf_error is not None:
                raise RuntimeError(f"PDF text extraction failed: {pdf_error}; browser navigation failed: {exc}") from exc
            raise
        return _service_snapshot(
            service_session_id,
            max_chars=max_chars,
            extract_query=extract_query,
        )
    except Exception as exc:  # noqa: BLE001
        return _service_error(exc)


@register(
    "browser_open_many",
    "Open up to four URLs concurrently through browser-service tabs and return JSON snapshots. "
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
    limited_urls = [_normalize_browser_url(url) for url in urls[:_BROWSER_MAX_SESSIONS]]
    timeout_ms = _clamp(timeout_ms, 1000, 120000)
    max_chars = max(0, int(max_chars))
    concurrency = _clamp(concurrency, 1, _BROWSER_MAX_SESSIONS)
    results: list[dict[str, Any] | None] = [None] * len(limited_urls)

    def open_one(index: int, url: str) -> dict[str, Any]:
        if not url:
            return {"index": index, "url": url, "ok": False, "error": "url must not be empty"}
        tab_id = ""
        service_session_id = ""
        close_error = ""
        try:
            service_session_id = _service_session_id(session_prefix or "bulk")
            new_tab = _browser_service_request(
                "POST",
                "/tab/new",
                json_body={"session_id": service_session_id, "url": "about:blank"},
            )
            tab_id = str(new_tab.get("tab_id") or "")
            if not tab_id:
                raise RuntimeError(f"new_tab returned empty tab_id: {new_tab}")
            _browser_service_request(
                "POST",
                "/browser/navigate",
                json_body={
                    "session_id": service_session_id,
                    "tab_id": tab_id,
                    "url": url,
                    "wait_until": _wait_until(wait_until),
                    "timeout_ms": timeout_ms,
                },
                timeout=timeout_ms / 1000 + 15,
            )
            snapshot = json.loads(
                _service_snapshot(
                    service_session_id,
                    tab_id=tab_id,
                    max_chars=max_chars,
                    extract_query=extract_query,
                )
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            snapshot = _service_error(exc)
            ok = False
        finally:
            if tab_id:
                try:
                    _browser_service_request(
                        "POST",
                        "/tab/close",
                        json_body={"session_id": service_session_id, "tab_id": tab_id},
                )
                except Exception as exc:  # noqa: BLE001
                    close_error = _service_error(exc)
        return {
            "index": index,
            "session_id": service_session_id,
            "url": url,
            "ok": ok,
            "snapshot": snapshot,
            **({"close_error": close_error} if close_error else {}),
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
    try:
        service_session_id = _service_session_id(session_id)
        return _service_snapshot(
            service_session_id,
            max_chars=max_chars,
            max_links=max_links,
            max_controls=max_controls,
            extract_query=extract_query,
        )
    except Exception as exc:  # noqa: BLE001
        return _service_error(exc)


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
    try:
        service_session_id = _service_session_id(session_id)
        _service_click_target(service_session_id, target, by)
        return _service_snapshot(service_session_id, max_chars=max_chars, extract_query=extract_query)
    except Exception as exc:  # noqa: BLE001
        return _service_error(exc)


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
    try:
        service_session_id = _service_session_id(session_id)
        _service_type_target(service_session_id, target, text, by, submit)
        return _service_snapshot(service_session_id, max_chars=max_chars, extract_query=extract_query)
    except Exception as exc:  # noqa: BLE001
        return _service_error(exc)


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
    logical_session_id = session_id or "default"
    with _SERVICE_SESSION_LOCK:
        service_session_id = _SERVICE_SESSIONS.pop(logical_session_id, "")
    if not service_session_id:
        return f"OK: browser session '{logical_session_id}' was not open"
    try:
        _browser_service_request("DELETE", f"/session/{service_session_id}", timeout=15)
    except Exception as exc:  # noqa: BLE001
        return _service_error(exc)
    return f"OK: browser session '{logical_session_id}' closed"
