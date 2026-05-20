"""
Web search tool — runs on the GPU host (no internet) by default.

Two operating modes are supported, picked automatically at import time:

1. **Proxy mode** (preferred when GPU host has no internet)
   Set ``SEARCH_PROXY_URL`` to the URL of the *search-proxy* FastAPI
   service running on a CPU host that DOES have internet access. With
   VS Code (or plain ``ssh -L``) port forwarding this can be a localhost
   address such as ``http://127.0.0.1:8090``. The GPU host then needs
   neither outbound internet nor any third-party API keys — keys live
   on the CPU host with the proxy.

2. **Direct mode** (legacy / local-dev)
   When ``SEARCH_PROXY_URL`` is empty, the tool talks directly to
   Serper / Jina / 0x0 from this process, which requires both internet
   access AND the original API keys (`SERPER_API_KEY`, `JINA_API_KEY`).

The two public tool functions keep identical signatures and return shapes
so the LLM tool schema does not change between modes::

    search_text(query, top_k=5, fetch=True, max_chars=500) -> list[dict]
    search_image(image, top_k=5, fetch=True, max_chars=500) -> list[dict]

Each result dict::

    {"rank": int, "title": str, "url": str, "snippet": str, "content"?: str}
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shlex
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("harness.tools.search")


def _load_tool_env() -> None:
    explicit = os.getenv("HARNESS_APIKEY_FILE", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(parent / "apikey.md" for parent in Path(__file__).resolve().parents)
    seen: set[Path] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            try:
                parts = shlex.split(raw, comments=True, posix=True)
            except ValueError:
                continue
            if parts and parts[0] == "export":
                parts = parts[1:]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) and not os.getenv(key):
                    os.environ[key] = value
        return


_load_tool_env()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Proxy mode (recommended for the air-gapped GPU host).
SEARCH_PROXY_URL    = os.getenv("SEARCH_PROXY_URL", "http://0.0.0.0:1227").rstrip("/")
SEARCH_PROXY_TOKEN  = os.getenv("SEARCH_PROXY_TOKEN", "") or os.getenv(
    "PROXY_API_TOKEN", ""
)
PROXY_HTTP_TIMEOUT  = float(os.getenv("SEARCH_PROXY_TIMEOUT", "120"))

# When SEARCH_PROXY_URL points at a 3rd-party tunnel (vscode.dev / Codespaces
# preview / cloudflared / ngrok), SSL verification or extra headers may be
# needed. These knobs let you control that without code changes.
#   * SEARCH_PROXY_VERIFY_SSL=false  -> skip TLS cert verify (cloudflared etc.)
#   * SEARCH_PROXY_EXTRA_HEADERS     -> JSON, e.g. '{"Cookie":"vscode-tkn=..."}'
SEARCH_PROXY_VERIFY_SSL = os.getenv("SEARCH_PROXY_VERIFY_SSL", "true").lower() not in (
    "0", "false", "no"
)
try:
    import json as _json
    SEARCH_PROXY_EXTRA_HEADERS: dict = _json.loads(
        os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}"
    )
    if not isinstance(SEARCH_PROXY_EXTRA_HEADERS, dict):
        SEARCH_PROXY_EXTRA_HEADERS = {}
except Exception:  # noqa: BLE001
    SEARCH_PROXY_EXTRA_HEADERS = {}

# Direct mode (only used when SEARCH_PROXY_URL is empty).
SERPER_API_KEY      = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY        = os.getenv("JINA_API_KEY", "")
IMAGE_UPLOADER      = os.getenv("IMAGE_UPLOADER", "0x0")

SERPER_SEARCH_URL   = "https://google.serper.dev/search"
SERPER_LENS_URL     = "https://google.serper.dev/lens"
JINA_READER_BASE    = "https://r.jina.ai/"

DEFAULT_TIMEOUT     = 30
JINA_TIMEOUT        = 45

QUERY_REPAIR_ENABLED = _env_bool("SEARCH_PROXY_QUERY_REPAIR", False)


def _proxy_enabled() -> bool:
    return bool(SEARCH_PROXY_URL)


def _proxy_headers(json_body: bool = True) -> dict:
    h: dict = {}
    if json_body:
        h["Content-Type"] = "application/json"
    if SEARCH_PROXY_TOKEN:
        h["Authorization"] = f"Bearer {SEARCH_PROXY_TOKEN}"
    # Extra headers (e.g. tunnel auth cookies) take precedence so users
    # can override Content-Type / Authorization if their tunnel needs it.
    if SEARCH_PROXY_EXTRA_HEADERS:
        h.update(SEARCH_PROXY_EXTRA_HEADERS)
    return h


# ---------------------------------------------------------------------------
# Proxy-mode helpers
# ---------------------------------------------------------------------------
def _proxy_post(path: str, payload: dict, timeout: float = PROXY_HTTP_TIMEOUT) -> dict:
    url = f"{SEARCH_PROXY_URL}{path}"
    resp = requests.post(
        url,
        json=payload,
        headers=_proxy_headers(json_body=True),
        timeout=timeout,
        verify=SEARCH_PROXY_VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()


def _proxy_search(path: str, payload: dict) -> list[dict]:
    """POST /search/text or /search/image; normalize to the legacy result shape."""
    data = _proxy_post(path, payload)
    if not data.get("ok", False):
        # Surface the proxy-side error as a single empty result so the LLM
        # can see what went wrong without us raising.
        err = data.get("error", "unknown proxy error")
        logger.warning("search-proxy %s failed: %s", path, err)
        return [{"rank": 1, "title": "", "url": "", "snippet": f"[proxy-error] {err}"}]

    out: list[dict] = []
    for hit in data.get("results", []) or []:
        entry = {
            "rank":    hit.get("rank", len(out) + 1),
            "title":   hit.get("title", ""),
            "url":     hit.get("url", ""),
            "snippet": hit.get("snippet", ""),
        }
        if hit.get("content") is not None:
            entry["content"] = hit["content"]
        out.append(entry)
    return out


def _has_usable_results(results: list[dict]) -> bool:
    for item in results or []:
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet") or "")
        if snippet.startswith("[proxy-error]"):
            continue
        if item.get("title") or item.get("url") or snippet or item.get("content"):
            return True
    return False


def _relaxed_search_query(query: str) -> str:
    value = " ".join(str(query or "").split())
    if not value:
        return ""
    if re.search(r"\b(?:site|filetype|intitle|inurl):", value, flags=re.IGNORECASE):
        return value
    value = re.sub(r"\b(?:JsonObject|json|answer|official answer|reference answer)\b", " ", value, flags=re.IGNORECASE)
    value = value.replace('"', " ")
    value = re.sub(r"\b(?:OR|AND)\b", " ", value)
    value = re.sub(r"[(){}\[\]]", " ", value)
    tokens = [token for token in value.split() if token]
    if len(tokens) <= 3:
        return " ".join(tokens)
    return " ".join(tokens[:12])[:220].strip()


def _annotate_query_repair(results: list[dict], original: str, relaxed: str) -> list[dict]:
    annotated: list[dict] = []
    note = f"[query-repair fallback: {original!r} -> {relaxed!r}]"
    for item in results:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        snippet = str(entry.get("snippet") or "")
        entry["snippet"] = f"{note} {snippet}".strip()
        annotated.append(entry)
    return annotated


def _proxy_upload_image(path: Path) -> str:
    """Stream a local image to the proxy's /upload_image and get a public URL."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"

    headers = _proxy_headers(json_body=False)

    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        resp = requests.post(
            f"{SEARCH_PROXY_URL}/upload_image",
            files=files,
            headers=headers,
            timeout=PROXY_HTTP_TIMEOUT,
            verify=SEARCH_PROXY_VERIFY_SSL,
        )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok", False):
        raise RuntimeError(f"upload_image failed: {data.get('error')}")
    url = data.get("url", "")
    if not url:
        raise RuntimeError(f"upload_image returned empty url: {data}")
    return url


# ---------------------------------------------------------------------------
# Direct-mode helpers (used only when SEARCH_PROXY_URL is empty)
# ---------------------------------------------------------------------------
def _require_serper_key() -> str:
    if not SERPER_API_KEY:
        raise RuntimeError(
            "SERPER_API_KEY not set. Either export it, or point "
            "SEARCH_PROXY_URL at a running search-proxy instance."
        )
    return SERPER_API_KEY


def _serper_post(url: str, payload: dict) -> dict:
    headers = {
        "X-API-KEY":    _require_serper_key(),
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _jina_fetch(url: str, max_chars: int) -> str:
    if not url:
        return ""
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = requests.get(reader_url, headers=headers, timeout=JINA_TIMEOUT)
        resp.raise_for_status()
        text = resp.text or ""
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n\n...[truncated at {max_chars} chars]"
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Jina fetch failed for %s: %s", url, exc)
        return f"[jina-error] {type(exc).__name__}: {exc}"


def _direct_upload_local_image(path: Path) -> str:
    if IMAGE_UPLOADER != "0x0":
        raise RuntimeError(
            f"Unsupported IMAGE_UPLOADER={IMAGE_UPLOADER!r}. "
            "Either set IMAGE_UPLOADER=0x0, host the image yourself "
            "and pass an http(s) URL, or run via SEARCH_PROXY_URL."
        )
    if not path.exists():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        headers = {"User-Agent": "kimi-agent-harness/1.0"}
        resp = requests.post(
            "https://0x0.st", files=files, headers=headers, timeout=DEFAULT_TIMEOUT,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected 0x0.st response: {url!r}")
    logger.info("Uploaded %s -> %s", path, url)
    return url


def _resolve_image_to_url_direct(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://"):
        return image
    p = Path(image).expanduser()
    if p.exists() and p.is_file():
        return _direct_upload_local_image(p)
    raise ValueError(
        f"search_image: {image!r} is neither an http(s) URL nor an existing local file."
    )


def _resolve_image_to_url_proxy(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://"):
        # Already public — proxy can feed it straight to /search/image.
        return image
    p = Path(image).expanduser()
    if p.exists() and p.is_file():
        return _proxy_upload_image(p)
    raise ValueError(
        f"search_image: {image!r} is neither an http(s) URL nor an existing local file."
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------
def search_text(
    query: str,
    top_k: int = 3,
    fetch: bool = True,
    max_chars: int = 500,
) -> list[dict]:
    """Text search on Google (via Serper) optionally enriched with full-text via Jina.

    Returns ``list[dict]`` with keys ``{rank, title, url, snippet, content?}``.
    """
    if not query or not query.strip():
        return []
    top_k = max(1, min(int(top_k), 10))
    if _proxy_enabled():
        logger.info("search_text(proxy) q=%r top_k=%d fetch=%s",
                    query, top_k, fetch)
        payload = {
            "query":     query,
            "top_k":     top_k,
            "fetch":     bool(fetch),
            "max_chars": int(max_chars),
        }
        results = _proxy_search(
            "/search/text",
            payload,
        )
        if QUERY_REPAIR_ENABLED and not _has_usable_results(results):
            relaxed_query = _relaxed_search_query(query)
            if relaxed_query and relaxed_query != " ".join(query.split()):
                retry_payload = dict(payload)
                retry_payload["query"] = relaxed_query
                retry_results = _proxy_search("/search/text", retry_payload)
                if _has_usable_results(retry_results):
                    return _annotate_query_repair(retry_results, query, relaxed_query)
        return results

    # Direct mode
    logger.info("search_text(direct) q=%r top_k=%d fetch=%s",
                query, top_k, fetch)
    payload = {"q": query, "num": top_k}
    data = _serper_post(SERPER_SEARCH_URL, payload)
    organic = data.get("organic", []) or []

    def rows_from_organic(items: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for rank, item in enumerate(items[:top_k], start=1):
            url = item.get("link") or ""
            entry = {
                "rank":    rank,
                "title":   item.get("title", ""),
                "url":     url,
                "snippet": item.get("snippet", ""),
            }
            if fetch and url:
                entry["content"] = _jina_fetch(url, max_chars)
            rows.append(entry)
        return rows

    results = rows_from_organic(organic)
    if QUERY_REPAIR_ENABLED and not _has_usable_results(results):
        relaxed_query = _relaxed_search_query(query)
        if relaxed_query and relaxed_query != " ".join(query.split()):
            retry_data = _serper_post(SERPER_SEARCH_URL, {"q": relaxed_query, "num": top_k})
            retry_results = rows_from_organic(retry_data.get("organic", []) or [])
            if _has_usable_results(retry_results):
                return _annotate_query_repair(retry_results, query, relaxed_query)
    return results


def search_image(
    image: str,
    top_k: int = 1,
    fetch: bool = True,
    max_chars: int = 500,
) -> list[dict]:
    """Reverse image search via Google Lens (Serper /lens).

    ``image`` may be an http(s) URL or a local file path. In proxy mode
    local files are streamed to the proxy's ``/upload_image`` endpoint;
    in direct mode they are pushed to a public host (default 0x0.st).
    """
    if not image or not image.strip():
        raise ValueError("search_image requires a non-empty `image` argument.")
    top_k = max(1, min(int(top_k), 10))

    if _proxy_enabled():
        image_url = _resolve_image_to_url_proxy(image.strip())
        logger.info("search_image(proxy) image_url=%s top_k=%d fetch=%s",
                    image_url, top_k, fetch)
        return _proxy_search(
            "/search/image",
            {
                "image_url": image_url,
                "top_k":     top_k,
                "fetch":     bool(fetch),
                "max_chars": int(max_chars),
            },
        )

    # Direct mode
    image_url = _resolve_image_to_url_direct(image.strip())
    logger.info("search_image(direct) image_url=%s top_k=%d fetch=%s",
                image_url, top_k, fetch)
    payload = {"url": image_url}
    data = _serper_post(SERPER_LENS_URL, payload)
    items = data.get("organic") or data.get("visual_matches") or []

    results: list[dict] = []
    for rank, item in enumerate(items[:top_k], start=1):
        url = item.get("link") or item.get("url") or ""
        entry = {
            "rank":    rank,
            "title":   item.get("title", ""),
            "url":     url,
            "snippet": item.get("snippet", "") or item.get("source", ""),
        }
        if fetch and url:
            entry["content"] = _jina_fetch(url, max_chars)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text")
    p_text.add_argument("query")
    p_text.add_argument("--top-k", type=int, default=3)
    p_text.add_argument("--no-fetch", action="store_true")

    p_img = sub.add_parser("image")
    p_img.add_argument("image", help="URL or local path")
    p_img.add_argument("--top-k", type=int, default=3)
    p_img.add_argument("--no-fetch", action="store_true")

    args = ap.parse_args()

    print(f"[mode] {'proxy via ' + SEARCH_PROXY_URL if _proxy_enabled() else 'direct'}")

    if args.cmd == "text":
        out = search_text(args.query, top_k=args.top_k, fetch=not args.no_fetch)
    else:
        out = search_image(args.image, top_k=args.top_k, fetch=not args.no_fetch)

    print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
