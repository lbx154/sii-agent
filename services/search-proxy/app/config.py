"""Runtime configuration for search-proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _raw_float_env(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float) -> float:
    return _raw_float_env(name, default, minimum)


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = _int_env("PORT", 8090, 1, 65535)

    # Upstream API keys (must be set on the *CPU host*, not on the GPU host).
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    jina_api_key: str = os.getenv("JINA_API_KEY", "")

    # Optional shared secret. If set, every request must carry
    # `Authorization: Bearer <token>`.
    api_token: str = os.getenv("PROXY_API_TOKEN", "")

    # Image upload backend used when the GPU side sends a local file
    # via /upload_image. Currently only `0x0` is supported.
    image_uploader: str = os.getenv("IMAGE_UPLOADER", "0x0")

    # Default HTTP timeouts (seconds) for outbound calls.
    serper_timeout: float = _float_env("SERPER_TIMEOUT", 30, 1)
    jina_timeout: float = _float_env("JINA_TIMEOUT", 45, 1)
    _default_jina_fetch_timeout = min(_raw_float_env("JINA_TIMEOUT", 45, 1), 30)
    jina_fetch_timeout: float = _float_env(
        "JINA_FETCH_TIMEOUT",
        _default_jina_fetch_timeout,
        1,
    )
    _default_jina_fetch_budget = min(
        _raw_float_env("JINA_FETCH_TIMEOUT", _default_jina_fetch_timeout, 1),
        60,
    )
    jina_fetch_budget: float = _float_env(
        "JINA_FETCH_BUDGET",
        _default_jina_fetch_budget,
        1,
    )
    upload_timeout: float = _float_env("UPLOAD_TIMEOUT", 60, 1)
    jina_fetch_workers: int = _int_env("JINA_FETCH_WORKERS", 20, 1, 128)
    proxy_workers: int = _int_env("PROXY_WORKERS", 64, 1, 256)
    http_pool_connections: int = _int_env("HTTP_POOL_CONNECTIONS", 64, 1, 512)
    http_pool_maxsize: int = _int_env("HTTP_POOL_MAXSIZE", 128, 1, 1024)
    query_repair: bool = _bool_env("SEARCH_PROXY_QUERY_REPAIR", False)


settings = Settings()
