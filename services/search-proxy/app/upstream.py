"""Outbound calls to Serper / Jina / 0x0.

This module is the only place that touches the real internet. It is intended
to run on the *CPU host* (which has internet access). The GPU host calls
this service over a private SSH-forwarded port.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from .config import settings

logger = logging.getLogger("search-proxy.upstream")

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_LENS_URL = "https://google.serper.dev/lens"
JINA_READER_BASE = "https://r.jina.ai/"
_THREAD_LOCAL = threading.local()


def _http() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=settings.http_pool_connections,
            pool_maxsize=settings.http_pool_maxsize,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------
def _serper_post(url: str, payload: dict) -> dict:
    if not settings.serper_api_key:
        raise RuntimeError("SERPER_API_KEY not set on the proxy host")
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    resp = _http().post(
        url, json=payload, headers=headers, timeout=settings.serper_timeout
    )
    resp.raise_for_status()
    return resp.json()


def serper_search(query: str, top_k: int) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    data = _serper_post(SERPER_SEARCH_URL, {"q": query, "num": top_k})
    return list(data.get("organic", []) or [])


def serper_lens(image_url: str, top_k: int) -> list[dict]:
    image_url = image_url.strip()
    if not image_url.startswith(("http://", "https://")):
        raise ValueError("image_url must be an http(s) URL")
    data = _serper_post(SERPER_LENS_URL, {"url": image_url})
    items = data.get("organic") or data.get("visual_matches") or data.get("images") or []
    return list(items[:top_k])


# ---------------------------------------------------------------------------
# Jina Reader
# ---------------------------------------------------------------------------
def jina_fetch(url: str, max_chars: int, timeout: float | None = None) -> tuple[str, bool]:
    """Return (content, truncated). On failure raises (caller decides format)."""
    if not url:
        return "", False
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be an http(s) URL")
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"
    resp = _http().get(reader_url, headers=headers, timeout=timeout or settings.jina_timeout)
    resp.raise_for_status()
    text = resp.text or ""
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n...[truncated at {max_chars} chars]"
        truncated = True
    return text, truncated


# ---------------------------------------------------------------------------
# Image hosting (used by the upload endpoint when GPU side has only a local file)
# ---------------------------------------------------------------------------
def upload_image(file_bytes: bytes, filename: str) -> str:
    if settings.image_uploader != "0x0":
        raise RuntimeError(
            f"Unsupported IMAGE_UPLOADER={settings.image_uploader!r}"
        )

    # Persist to a temp file so we can hand a name + content-type to multipart.
    suffix = Path(filename).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        mime, _ = mimetypes.guess_type(filename)
        mime = mime or "application/octet-stream"
        with open(tmp_path, "rb") as fh:
            files = {"file": (filename, fh, mime)}
            headers = {"User-Agent": "kimi-agent-harness/search-proxy"}
            resp = _http().post(
                "https://0x0.st",
                files=files,
                headers=headers,
                timeout=settings.upload_timeout,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        if not url.startswith("http"):
            raise RuntimeError(f"Unexpected 0x0.st response: {url!r}")
        logger.info("Uploaded %s -> %s", filename, url)
        return url
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
