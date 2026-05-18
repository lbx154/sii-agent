"""Search tools backed by the configured harness search-proxy."""
from __future__ import annotations
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .registry import register

load_dotenv()


def _clamp_int(value: int, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _configured_backends() -> list[str]:
    return ["proxy"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _search_proxy_url() -> str:
    return os.getenv("SEARCH_PROXY_URL", "").strip().rstrip("/")


def _search_proxy_timeout() -> float:
    try:
        return float(os.getenv("SEARCH_PROXY_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def _search_proxy_upload_timeout() -> float:
    try:
        return float(os.getenv("SEARCH_PROXY_UPLOAD_TIMEOUT", str(_search_proxy_timeout())))
    except ValueError:
        return _search_proxy_timeout()


def _search_proxy_upload_retries() -> int:
    return _clamp_int(os.getenv("SEARCH_PROXY_IMAGE_UPLOAD_RETRIES", "2"), 2, 0, 5)


def _image_upload_backends() -> list[str]:
    raw = os.getenv("SEARCH_PROXY_IMAGE_UPLOAD_BACKENDS", "tmpfiles,catbox,proxy")
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(backends)) or ["tmpfiles", "catbox", "proxy"]


def _search_proxy_headers(json_body: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"} if json_body else {}
    token = os.getenv("SEARCH_PROXY_TOKEN", "") or os.getenv("PROXY_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        extra = json.loads(os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}")
    except json.JSONDecodeError:
        extra = {}
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _search_proxy_verify_ssl() -> bool:
    return _env_bool("SEARCH_PROXY_VERIFY_SSL", True)


@register(
    "web_search",
    "Search the web via the configured search-proxy and return top results (title, url, snippet/content). "
    "Use for any factual / up-to-date question.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search query"},
            "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
)
def web_search(query: str, k: int = 3) -> str:
    k = _clamp_int(k, 3, 1, 10)
    result = _guarded_search("search-proxy", lambda: _proxy_text_search(query, k))
    if result == "(no results)":
        return result
    return f"## proxy\n{result}"


@register(
    "reverse_image_search",
    "Use the configured search-proxy image/lens search to find web pages related to an image URL or local image path. "
    "Use when the image itself must be matched to pages/entities. Pass the user question as query so the tool can fall back to text search if image upload/lens is unavailable.",
    {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "http(s) image URL, data:image URL, or local image path",
            },
            "query": {
                "type": "string",
                "default": "",
                "description": "Optional text question/search hint used only as fallback if image search fails",
            },
            "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            "fetch": {"type": "boolean", "default": False},
            "max_chars": {"type": "integer", "default": 0, "minimum": 0, "maximum": 10000},
            "fallback_to_text": {"type": "boolean", "default": True},
        },
        "required": ["source"],
    },
)
def reverse_image_search(
    source: str,
    query: str = "",
    k: int = 3,
    fetch: bool = False,
    max_chars: int = 0,
    fallback_to_text: bool = True,
) -> str:
    k = _clamp_int(k, 3, 1, 10)
    max_chars = _clamp_int(max_chars, 0, 0, 10000)
    return _guarded_search(
        "search-proxy image",
        lambda: _proxy_image_search(
            source,
            query=query,
            k=k,
            fetch=bool(fetch),
            max_chars=max_chars,
            fallback_to_text=bool(fallback_to_text),
        ),
    )


def _guarded_search(label: str, fn) -> str:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {label} search failed: {type(e).__name__}: {e}"


def _proxy_post(path: str, payload: dict) -> dict:
    import httpx

    base_url = _search_proxy_url()
    if not base_url:
        raise RuntimeError("SEARCH_PROXY_URL is not set")
    response = httpx.post(
        f"{base_url}{path}",
        json=payload,
        headers=_search_proxy_headers(json_body=True),
        timeout=_search_proxy_timeout(),
        verify=_search_proxy_verify_ssl(),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"search-proxy returned non-object JSON: {data!r}")
    return data


def _proxy_upload_image_bytes(data: bytes, filename: str, mime: str) -> str:
    import httpx

    base_url = _search_proxy_url()
    if not base_url:
        raise RuntimeError("SEARCH_PROXY_URL is not set")
    files = {"file": (filename, data, mime)}
    attempts = _search_proxy_upload_retries() + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            response = httpx.post(
                f"{base_url}/upload_image",
                files=files,
                headers=_search_proxy_headers(json_body=False),
                timeout=_search_proxy_upload_timeout(),
                verify=_search_proxy_verify_ssl(),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not payload.get("ok"):
                raise RuntimeError(f"upload_image failed: {payload}")
            url = str(payload.get("url") or "")
            if not url:
                raise RuntimeError(f"upload_image returned empty url: {payload}")
            return url
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(1.0 + attempt, 3.0))
    raise RuntimeError(f"upload_image failed after {attempts} attempt(s): {last_error}") from last_error


def _upload_tmpfiles(data: bytes, filename: str, mime: str) -> str:
    import httpx

    files = {"file": (filename, data, mime)}
    response = httpx.post(
        "https://tmpfiles.org/api/v1/upload",
        files=files,
        timeout=_search_proxy_upload_timeout(),
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    url = str(((payload.get("data") or {}) if isinstance(payload, dict) else {}).get("url") or "")
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(f"tmpfiles upload returned invalid url: {payload}")
    parsed = urlparse(url)
    if parsed.netloc == "tmpfiles.org" and not parsed.path.startswith("/dl/"):
        url = f"{parsed.scheme}://{parsed.netloc}/dl{parsed.path}"
    return url


def _upload_catbox(data: bytes, filename: str, mime: str) -> str:
    import httpx

    files = {
        "reqtype": (None, "fileupload"),
        "fileToUpload": (filename, data, mime),
    }
    response = httpx.post(
        "https://catbox.moe/user/api.php",
        files=files,
        timeout=_search_proxy_upload_timeout(),
        follow_redirects=True,
    )
    response.raise_for_status()
    url = response.text.strip()
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(f"catbox upload returned invalid url: {url[:200]!r}")
    return url


def _upload_0x0(data: bytes, filename: str, mime: str) -> str:
    import httpx

    files = {"file": (filename, data, mime)}
    response = httpx.post(
        "https://0x0.st",
        files=files,
        headers={"User-Agent": "sii-agent/1.0"},
        timeout=_search_proxy_upload_timeout(),
        follow_redirects=True,
    )
    response.raise_for_status()
    url = response.text.strip()
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(f"0x0 upload returned invalid url: {url[:200]!r}")
    return url


def _public_upload_image_bytes(data: bytes, filename: str, mime: str) -> str:
    uploaders = {
        "tmpfiles": _upload_tmpfiles,
        "catbox": _upload_catbox,
        "0x0": _upload_0x0,
        "proxy": _proxy_upload_image_bytes,
    }
    errors: list[str] = []
    for backend in _image_upload_backends():
        uploader = uploaders.get(backend)
        if uploader is None:
            errors.append(f"{backend}: unknown uploader")
            continue
        try:
            return uploader(data, filename, mime)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all image upload backends failed: " + " | ".join(errors))


def _proxy_image_url(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return source
    if source.startswith("data:image/"):
        header, _, encoded = source.partition(",")
        if not encoded:
            raise ValueError("data image URL is missing base64 payload")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        suffix = (mimetypes.guess_extension(mime) or ".png").lstrip(".")
        return _public_upload_image_bytes(base64.b64decode(encoded), f"image.{suffix}", mime)
    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"image file not found: {source}")
    mime, _ = mimetypes.guess_type(str(path))
    return _public_upload_image_bytes(path.read_bytes(), path.name, mime or "application/octet-stream")


def _format_proxy_results(payload: dict, query: str | None = None) -> str:
    if not payload.get("ok", False):
        return f"ERROR: search-proxy failed: {payload.get('error', 'unknown error')}"
    rows: list[str] = []
    for i, item in enumerate(payload.get("results", []) or [], 1):
        title = str(item.get("title") or "")
        url = str(item.get("url") or "")
        snippet = str(item.get("snippet") or "")
        content = str(item.get("content") or "")
        body = snippet[:500]
        if content:
            body = (body + "\n" if body else "") + content[:3000]
        rows.append(f"[{item.get('rank') or i}] {title}\n    {url}\n    {body}")
    if rows:
        return "\n".join(rows)
    return "(no results)" if query is None else f"(no results for {query!r})"


def _proxy_text_search(query: str, k: int) -> str:
    fetch = _env_bool("SEARCH_PROXY_FETCH", False)
    max_chars = _clamp_int(os.getenv("SEARCH_PROXY_MAX_CHARS", "0"), 0, 0, 10000)
    payload = {"query": query, "top_k": k, "fetch": fetch, "max_chars": max_chars}
    return _format_proxy_results(_proxy_post("/search/text", payload), query=query)


def _proxy_image_search(
    source: str,
    query: str,
    k: int,
    fetch: bool,
    max_chars: int,
    fallback_to_text: bool,
) -> str:
    image_url = ""
    try:
        image_url = _proxy_image_url(source)
        payload = {"image_url": image_url, "top_k": k, "fetch": fetch, "max_chars": max_chars}
        result = _proxy_post("/search/image", payload)
        if result.get("ok", False):
            return json.dumps(
                {
                    "mode": "image_lens",
                    "source": source,
                    "uploaded_or_resolved_image_url": image_url,
                    "ok": True,
                    "error": None,
                    "results": result.get("results", []),
                },
                ensure_ascii=False,
                indent=2,
            )
        image_error = f"search-proxy image search failed: {result.get('error', 'unknown error')}"
    except Exception as exc:  # noqa: BLE001
        image_error = f"{type(exc).__name__}: {exc}"

    fallback_query = " ".join(str(query or "").split())
    if fallback_to_text and fallback_query:
        fallback_results = _proxy_text_search(fallback_query, k)
        return json.dumps(
            {
                "mode": "text_fallback",
                "source": source,
                "uploaded_or_resolved_image_url": image_url,
                "ok": False,
                "image_error": image_error,
                "fallback_query": fallback_query,
                "fallback_results": fallback_results,
                "usage_note": (
                    "Image lens/upload failed, so this returned text-search fallback results. "
                    "Use browser_open on promising URLs or refine the query if needed."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(
        {
            "mode": "image_lens",
            "source": source,
            "uploaded_or_resolved_image_url": image_url,
            "ok": False,
            "error": image_error,
            "results": [],
            "usage_note": "Image lens/upload failed and no fallback query was provided.",
        },
        ensure_ascii=False,
        indent=2,
    )
