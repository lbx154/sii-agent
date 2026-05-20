"""服务全局配置，通过环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _str_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


@dataclass
class Settings:
    host: str = _str_env("HOST", "0.0.0.0")
    port: int = _int_env("PORT", 8080, 1, 65535)

    headless: bool = _bool(os.getenv("HEADLESS"), True)
    browser_cdp_host: str = _str_env("BROWSER_CDP_HOST", "127.0.0.1")
    browser_cdp_port: int = _int_env("BROWSER_CDP_PORT", 9222, 1, 65535)
    viewport_width: int = _int_env("DEFAULT_VIEWPORT_WIDTH", 1280, 320, 7680)
    viewport_height: int = _int_env("DEFAULT_VIEWPORT_HEIGHT", 800, 240, 4320)

    max_sessions: int = _int_env("MAX_SESSIONS", 10, 1, 1000)
    max_tabs_per_session: int = _int_env("MAX_TABS_PER_SESSION", 20, 1, 1000)
    session_idle_timeout: int = _int_env("SESSION_IDLE_TIMEOUT", 1800, 60, 86400)
    block_resource_types: str = os.getenv("BROWSER_BLOCK_RESOURCE_TYPES", "").strip()

    api_token: str = os.getenv("API_TOKEN", "").strip()


settings = Settings()
