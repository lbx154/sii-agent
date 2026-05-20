"""轻量级 Python 客户端，给内网另一台服务器使用。

用法
----
    from browser_client import BrowserClient

    bc = BrowserClient("http://192.168.1.10:8080")
    sid = bc.create_session()["session_id"]
    bc.navigate("https://example.com", session_id=sid)
    text = bc.get_text(session_id=sid)
    print(text[:500])
    bc.close_session(sid)
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import requests


class BrowserClient:
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # ---------- 内部 ----------
    def _req(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        r = requests.request(
            method,
            url,
            headers=self._headers,
            timeout=self.timeout,
            **kwargs,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {url} -> {r.status_code}: {r.text}")
        return r.json()

    # ---------- Session ----------
    def create_session(self) -> dict:
        return self._req("POST", "/session/create")

    def list_sessions(self) -> dict:
        return self._req("GET", "/session/list")

    def close_session(self, session_id: str) -> dict:
        return self._req("DELETE", f"/session/{session_id}")

    # ---------- Tab ----------
    def new_tab(
        self, session_id: Optional[str] = None, url: Optional[str] = None
    ) -> dict:
        return self._req(
            "POST",
            "/tab/new",
            json={"session_id": session_id, "url": url},
        )

    def close_tab(self, session_id: str, tab_id: str) -> dict:
        return self._req(
            "POST",
            "/tab/close",
            json={"session_id": session_id, "tab_id": tab_id},
        )

    def list_tabs(self, session_id: str) -> dict:
        return self._req("GET", f"/tab/list/{session_id}")

    # ---------- 浏览器操作 ----------
    def navigate(
        self,
        url: str,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        wait_until: str = "load",
        timeout_ms: int = 30000,
    ) -> dict:
        return self._req(
            "POST",
            "/browser/navigate",
            json={
                "url": url,
                "session_id": session_id,
                "tab_id": tab_id,
                "wait_until": wait_until,
                "timeout_ms": timeout_ms,
            },
        )

    def get_text(
        self,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> str:
        r = self._req(
            "POST",
            "/browser/get_text",
            json={
                "session_id": session_id,
                "tab_id": tab_id,
                "selector": selector,
            },
        )
        return r["text"]

    def get_html(
        self,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> str:
        r = self._req(
            "POST",
            "/browser/get_html",
            json={
                "session_id": session_id,
                "tab_id": tab_id,
                "selector": selector,
            },
        )
        return r["html"]

    def screenshot(
        self,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
        save_to: Optional[str] = None,
        image_format: str = "png",
    ) -> bytes:
        r = self._req(
            "POST",
            "/browser/screenshot",
            json={
                "session_id": session_id,
                "tab_id": tab_id,
                "full_page": full_page,
                "selector": selector,
                "image_format": image_format,
            },
        )
        data = base64.b64decode(r["image_base64"])
        if save_to:
            with open(save_to, "wb") as f:
                f.write(data)
        return data

    def click(
        self,
        selector: str,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        timeout_ms: int = 10000,
    ) -> dict:
        return self._req(
            "POST",
            "/browser/click",
            json={
                "selector": selector,
                "session_id": session_id,
                "tab_id": tab_id,
                "timeout_ms": timeout_ms,
            },
        )

    def type_text(
        self,
        selector: str,
        text: str,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
        clear: bool = True,
        press_enter: bool = False,
        delay_ms: int = 0,
    ) -> dict:
        return self._req(
            "POST",
            "/browser/type",
            json={
                "selector": selector,
                "text": text,
                "session_id": session_id,
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
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> dict:
        return self._req(
            "POST",
            "/browser/scroll",
            json={
                "direction": direction,
                "pixels": pixels,
                "session_id": session_id,
                "tab_id": tab_id,
            },
        )

    def eval_js(
        self,
        script: str,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> Any:
        r = self._req(
            "POST",
            "/browser/eval",
            json={
                "script": script,
                "session_id": session_id,
                "tab_id": tab_id,
            },
        )
        return r["result"]

    def title(
        self,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> dict:
        params = {}
        if session_id:
            params["session_id"] = session_id
        if tab_id:
            params["tab_id"] = tab_id
        return self._req("GET", "/browser/title", params=params)

    def cdp_url(self) -> dict:
        return self._req("GET", "/browser/cdp_url")

    def health(self) -> dict:
        return self._req("GET", "/health")
