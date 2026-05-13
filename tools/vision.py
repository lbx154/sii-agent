"""Vision/OCR tools backed by the configured GPT-5.4 OpenAI-compatible endpoint."""
from __future__ import annotations

import base64
import ipaddress
import mimetypes
import os
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv

from .registry import register

_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_ALLOWED_MIME_PREFIX = "image/"
_MAX_REDIRECTS = 5
_DEFAULT_PROMPT = (
    "Extract all readable text from the image, then briefly describe visual evidence "
    "that may help answer a factual question. Be concise and do not invent text."
)


def _env_model() -> str:
    load_dotenv()
    return (
        os.getenv("OPD_EXPERT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-5.4"
    )


def _env_base_url() -> str:
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")
    if base_url:
        return base_url
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/") + "/openai/v1/"
    raise ValueError("OPENAI_BASE_URL or AZURE_OPENAI_BASE_URL is required for image_to_text")


def _env_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for image_to_text")
    return api_key


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _assert_public_http_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source must be an http(s) image URL")
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve image URL host: {host}") from exc
    for address in {item[4][0] for item in addresses}:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"refusing to fetch non-public image URL host: {host}")


def _guess_mime(source: str, content_type: str | None = None) -> str:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith(_ALLOWED_MIME_PREFIX):
        return mime
    guessed, _ = mimetypes.guess_type(source)
    if guessed and guessed.startswith(_ALLOWED_MIME_PREFIX):
        return guessed
    return "image/png"


def _load_image(source: str) -> tuple[str, bytes]:
    if _is_http_url(source):
        current_url = source
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_public_http_url(current_url)
            response = httpx.get(current_url, timeout=30, follow_redirects=False)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    break
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise ValueError(f"too many redirects while fetching image URL; max {_MAX_REDIRECTS}")
        response.raise_for_status()
        data = response.content
        mime = _guess_mime(str(response.url), response.headers.get("content-type"))
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        data = path.read_bytes()
        mime = _guess_mime(str(path))

    if not mime.startswith(_ALLOWED_MIME_PREFIX):
        raise ValueError(f"source is not an image MIME type: {mime}")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image is too large: {len(data)} bytes; max {_MAX_IMAGE_BYTES}")
    return mime, data


def _call_vision(model: str, image_data_url: str, prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_env_api_key(), base_url=_env_base_url(), timeout=90)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
    return (response.choices[0].message.content or "").strip()


@register(
    "image_to_text",
    "Extract text/OCR and describe useful visual evidence from an image URL or local image path using GPT-5.4. "
    "Use for image-to-text search, screenshots, document images, charts, or visual clue questions.",
    {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "http(s) image URL or local image path",
            },
            "prompt": {
                "type": "string",
                "default": _DEFAULT_PROMPT,
                "description": "Specific OCR/vision instruction",
            },
            "model": {
                "type": "string",
                "default": "gpt-5.4",
                "description": "Vision-capable OpenAI-compatible model; defaults to OPD_EXPERT_MODEL/AZURE_OPENAI_DEPLOYMENT.",
            },
            "max_tokens": {"type": "integer", "default": 512, "minimum": 64, "maximum": 2048},
        },
        "required": ["source"],
    },
)
def image_to_text(
    source: str,
    prompt: str = _DEFAULT_PROMPT,
    model: str | None = None,
    max_tokens: int = 512,
) -> str:
    try:
        max_tokens = max(64, min(2048, int(max_tokens)))
        mime, data = _load_image(source)
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        text = _call_vision(model or _env_model(), data_url, prompt or _DEFAULT_PROMPT, max_tokens)
        return text or "(no text returned)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: image_to_text failed: {type(exc).__name__}: {exc}"
