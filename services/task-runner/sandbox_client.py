"""
Browser-service client singleton — manages the remote browser HTTP service.

The old AIO Sandbox (`agent_sandbox`) has been replaced by the lightweight
`browser-service` (FastAPI) running on the host machine. This module wraps
the HTTP API as a thin singleton client and also persists ONE default
session id so all tool calls share the same browser tab across turns
(matching the previous "stateful current page" semantics).

Public API (kept stable for tool modules):
    get_sandbox()        -> BrowserSandboxClient   # singleton
    reconnect_sandbox()  -> BrowserSandboxClient   # force a fresh session
    reset_sandbox()                                # test helper
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# NOTE: same env var name as before so existing deployments keep working.
SANDBOX_BASE_URL = os.getenv("SANDBOX_BASE_URL", "http://0.0.0.0:8080")
SANDBOX_API_TOKEN = os.getenv("SANDBOX_API_TOKEN", "") or os.getenv(
    "BROWSER_API_TOKEN", ""
)
SANDBOX_HTTP_TIMEOUT = float(os.getenv("SANDBOX_HTTP_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
class BrowserSandboxClient:
    """Thin synchronous wrapper around the browser-service HTTP API.

    Owns one default ``session_id`` (lazily created) so successive tool
    calls operate on the same browser tab — this preserves the previous
    "the page state is shared across tool calls within one task" behaviour.
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        timeout: float = SANDBOX_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._session_id: Optional[str] = None
        self._lock = threading.Lock()

    # ---------- low level ----------
    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers,
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"{method} {url} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {url} -> {resp.status_code}: {resp.text[:500]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(f"{method} {url} returned non-JSON: {exc}") from exc

    # ---------- session ----------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def ensure_session(self) -> str:
        """Return the current default session_id, creating one if needed."""
        with self._lock:
            if self._session_id:
                return self._session_id
            r = self._request("POST", "/session/create")
            sid = r.get("session_id") or ""
            if not sid:
                raise RuntimeError(f"unexpected create_session response: {r}")
            self._session_id = sid
            logger.info("Browser session created: %s", sid)
            return sid

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def reset_session(self) -> None:
        """Drop the cached session_id (next call will create a new one)."""
        with self._lock:
            old = self._session_id
            self._session_id = None
        if old:
            try:
                self._request("DELETE", f"/session/{old}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_session(%s) failed: %s", old, exc)

    # ---------- browser actions ----------
    def navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout_ms: int = 30000,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/navigate",
            json={
                "url": url,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "wait_until": wait_until,
                "timeout_ms": timeout_ms,
            },
            timeout=max(self.timeout, timeout_ms / 1000 + 15),
        )

    def get_text(
        self,
        selector: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> str:
        r = self._request(
            "POST",
            "/browser/get_text",
            json={
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "selector": selector,
            },
        )
        return r.get("text", "")

    def get_html(
        self,
        selector: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> str:
        r = self._request(
            "POST",
            "/browser/get_html",
            json={
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "selector": selector,
            },
        )
        return r.get("html", "")

    def click(
        self,
        selector: str,
        timeout_ms: int = 10000,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/click",
            json={
                "selector": selector,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "timeout_ms": timeout_ms,
            },
            timeout=max(self.timeout, timeout_ms / 1000 + 10),
        )

    def type_text(
        self,
        selector: str,
        text: str,
        clear: bool = True,
        press_enter: bool = False,
        delay_ms: int = 0,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/type",
            json={
                "selector": selector,
                "text": text,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "clear": clear,
                "press_enter": press_enter,
                "delay_ms": delay_ms,
            },
        )

    def scroll(
        self,
        direction: str = "down",
        pixels: int = 600,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/browser/scroll",
            json={
                "direction": direction,
                "pixels": pixels,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
            },
        )

    def eval_js(
        self,
        script: str,
        tab_id: Optional[str] = None,
    ) -> Any:
        r = self._request(
            "POST",
            "/browser/eval",
            json={
                "script": script,
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
            },
        )
        return r.get("result")

    def title(self, tab_id: Optional[str] = None) -> dict:
        params = {"session_id": self.ensure_session()}
        if tab_id:
            params["tab_id"] = tab_id
        return self._request("GET", "/browser/title", params=params)

    def screenshot(
        self,
        full_page: bool = False,
        selector: Optional[str] = None,
        image_format: str = "png",
        save_to: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> bytes:
        r = self._request(
            "POST",
            "/browser/screenshot",
            json={
                "session_id": self.ensure_session(),
                "tab_id": tab_id,
                "full_page": full_page,
                "selector": selector,
                "image_format": image_format,
            },
        )
        data = base64.b64decode(r.get("image_base64", ""))
        if save_to and data:
            with open(save_to, "wb") as f:
                f.write(data)
        return data

    # ---------- tab management ----------
    def new_tab(self, url: Optional[str] = None) -> dict:
        return self._request(
            "POST",
            "/tab/new",
            json={"session_id": self.ensure_session(), "url": url},
        )

    def close_tab(self, tab_id: str) -> dict:
        return self._request(
            "POST",
            "/tab/close",
            json={"session_id": self.ensure_session(), "tab_id": tab_id},
        )

    def list_tabs(self) -> dict:
        return self._request("GET", f"/tab/list/{self.ensure_session()}")


# ---------------------------------------------------------------------------
# Singleton plumbing (API kept compatible with the old sandbox_client)
# ---------------------------------------------------------------------------
_client: Optional[BrowserSandboxClient] = None
_client_url: Optional[str] = None
_singleton_lock = threading.Lock()


def _create_client(url: str) -> BrowserSandboxClient:
    logger.info("Connecting to browser-service at %s", url)
    cli = BrowserSandboxClient(
        base_url=url, token=SANDBOX_API_TOKEN, timeout=SANDBOX_HTTP_TIMEOUT
    )
    try:
        h = cli.health()
        logger.info(
            "browser-service connected: status=%s, browser_running=%s, sessions=%s",
            h.get("status"),
            h.get("browser_running"),
            h.get("sessions"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("browser-service health check failed: %s", exc)
    return cli


def get_sandbox(base_url: Optional[str] = None) -> BrowserSandboxClient:
    """Return a lazily-initialised browser-service client (singleton).

    The function name is kept for backward compatibility — the returned
    object is a ``BrowserSandboxClient``, not the old ``agent_sandbox.Sandbox``.
    """
    global _client, _client_url
    url = (base_url or _client_url or SANDBOX_BASE_URL).rstrip("/")
    with _singleton_lock:
        _client_url = url
        if _client is None:
            _client = _create_client(url)
        return _client


# Friendlier alias — new code is encouraged to use this name.
def get_browser_client(base_url: Optional[str] = None) -> BrowserSandboxClient:
    return get_sandbox(base_url)


def reconnect_sandbox() -> BrowserSandboxClient:
    """Force a fresh client + session. Use after a 401 / dead-session error."""
    global _client
    logger.warning("Browser session expired — reconnecting...")
    with _singleton_lock:
        if _client is not None:
            try:
                _client.reset_session()
            except Exception:  # noqa: BLE001
                pass
        _client = None
    return get_sandbox()


def reset_sandbox() -> None:
    """Reset the singleton (useful for testing)."""
    global _client
    with _singleton_lock:
        _client = None
