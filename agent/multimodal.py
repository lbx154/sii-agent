"""Helpers for attaching image inputs to OpenAI-compatible chat messages."""
from __future__ import annotations

import base64
import binascii
import mimetypes
from pathlib import Path
from typing import Any


def guess_image_mime(data: bytes, path: str | None = None, fallback: str | None = None) -> str:
    if fallback and fallback.startswith("image/"):
        return fallback
    guessed = mimetypes.guess_type(path or "")[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def image_url_from_source(source: str, image_meta: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    ref = str(source or "").strip()
    if not ref:
        return None, None
    if ref.startswith(("http://", "https://", "data:image/")):
        return ref, None

    try:
        path = Path(ref).expanduser()
        if path.exists() and path.is_file():
            data = path.read_bytes()
            if not data:
                return None, "image file is empty"
            mime = guess_image_mime(data, str(path), str((image_meta or {}).get("mime") or ""))
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        data = base64.b64decode("".join(ref.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not data:
        return None, "image payload is empty"
    mime = guess_image_mime(data, fallback=str((image_meta or {}).get("mime") or ""))
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", None


def multimodal_user_content(text: str, image_url: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def build_user_content_with_image(
    text: str,
    image_source: str | None,
    image_meta: dict[str, Any] | None = None,
    enabled: bool = True,
) -> tuple[str | list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"enabled": bool(enabled), "attached": False}
    if not image_source or not enabled:
        return text, meta
    image_url, error = image_url_from_source(image_source, image_meta)
    if error:
        meta["error"] = error
        return text, meta
    if not image_url:
        return text, meta
    meta.update(
        {
            "attached": True,
            "source": "url" if image_url.startswith(("http://", "https://")) else "data_url",
        }
    )
    return multimodal_user_content(text, image_url), meta
