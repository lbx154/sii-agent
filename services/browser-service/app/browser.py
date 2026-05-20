"""浏览器与会话管理。

设计要点
--------
- 全局只起 **一个** Chromium 实例（headless），暴露统一的 CDP 端口。
- 通过 `BrowserContext` 实现多 session 隔离（cookie / storage 互不影响）。
- 每个 session 内可有多个 tab（`Page`）。
- 提供线程安全的异步获取接口。
- 支持空闲会话自动回收。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class TabHandle:
    tab_id: str
    page: Page
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionHandle:
    session_id: str
    context: BrowserContext
    tabs: dict[str, TabHandle] = field(default_factory=dict)
    active_tab_id: Optional[str] = None
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()


class BrowserManager:
    """单例浏览器管理器。"""

    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._sessions: dict[str, SessionHandle] = {}
        self._lock = asyncio.Lock()
        self._gc_task: Optional[asyncio.Task] = None
        self._cdp_endpoint: str = ""

    def _attach_page_handlers(self, page: Page) -> None:
        page.on(
            "download",
            lambda download: asyncio.create_task(download.cancel()),
        )

    async def _new_page(self, sess: SessionHandle) -> Page:
        if len(sess.tabs) >= settings.max_tabs_per_session:
            victim_id = min(sess.tabs.values(), key=lambda tab: tab.created_at).tab_id
            logger.warning(
                "Session %s reached max tabs, evicting tab %s",
                sess.session_id,
                victim_id,
            )
            await self.close_tab(sess.session_id, victim_id)
        page = await sess.context.new_page()
        self._attach_page_handlers(page)
        return page

    # ---------- 生命周期 ----------
    async def start(self) -> None:
        async with self._lock:
            if self._browser:
                return
            logger.info("Launching Chromium (headless=%s) ...", settings.headless)
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=settings.headless,
                args=[
                    f"--remote-debugging-port={settings.browser_cdp_port}",
                    f"--remote-debugging-address={settings.browser_cdp_host}",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            # Playwright 启动后，CDP 端口是 chromium 自身的 DevTools 端口
            advertised_host = (
                "127.0.0.1"
                if settings.browser_cdp_host in {"0.0.0.0", "::"}
                else settings.browser_cdp_host
            )
            self._cdp_endpoint = f"http://{advertised_host}:{settings.browser_cdp_port}"
            logger.info("Chromium launched. CDP endpoint: %s", self._cdp_endpoint)

            # 启动空闲回收任务
            self._gc_task = asyncio.create_task(self._gc_loop())

    async def stop(self) -> None:
        async with self._lock:
            if self._gc_task:
                self._gc_task.cancel()
                self._gc_task = None
            for sid in list(self._sessions.keys()):
                await self._close_session_locked(sid)
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._pw:
                await self._pw.stop()
                self._pw = None
            logger.info("Browser stopped.")

    # ---------- Session ----------
    async def create_session(self) -> SessionHandle:
        await self.start()
        async with self._lock:
            if len(self._sessions) >= settings.max_sessions:
                # 回收最久未活动的
                victim = min(self._sessions.values(), key=lambda s: s.last_active)
                logger.warning(
                    "Reached max sessions, evicting %s", victim.session_id
                )
                await self._close_session_locked(victim.session_id)

            sid = uuid.uuid4().hex[:12]
            assert self._browser is not None
            ctx = await self._browser.new_context(
                viewport={
                    "width": settings.viewport_width,
                    "height": settings.viewport_height,
                },
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            blocked_types = {
                item.strip()
                for item in settings.block_resource_types.split(",")
                if item.strip()
            }
            if blocked_types:
                async def _route_resource(route):
                    if route.request.resource_type in blocked_types:
                        await route.abort()
                    else:
                        await route.continue_()

                await ctx.route("**/*", _route_resource)
            sess = SessionHandle(session_id=sid, context=ctx)
            self._sessions[sid] = sess

            # 默认开一个 tab
            page = await self._new_page(sess)
            tab_id = uuid.uuid4().hex[:8]
            sess.tabs[tab_id] = TabHandle(tab_id=tab_id, page=page)
            sess.active_tab_id = tab_id
            logger.info("Session %s created with tab %s", sid, tab_id)
            return sess

    async def get_or_create_session(
        self, session_id: Optional[str]
    ) -> SessionHandle:
        if session_id and session_id in self._sessions:
            sess = self._sessions[session_id]
            sess.touch()
            return sess
        if session_id and session_id not in self._sessions:
            raise KeyError(f"session_id '{session_id}' not found")
        # 没传 session_id：创建新 session
        return await self.create_session()

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            await self._close_session_locked(session_id)

    async def _close_session_locked(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if not sess:
            return
        try:
            await sess.context.close()
        except Exception as e:
            logger.warning("close session %s err: %s", session_id, e)
        logger.info("Session %s closed", session_id)

    def list_sessions(self) -> list[SessionHandle]:
        return list(self._sessions.values())

    # ---------- Tab ----------
    async def new_tab(
        self, session_id: Optional[str], url: Optional[str] = None
    ) -> tuple[SessionHandle, TabHandle]:
        sess = await self.get_or_create_session(session_id)
        page = await self._new_page(sess)
        try:
            if url:
                await page.goto(url, wait_until="load")
        except Exception:
            await page.close()
            raise
        tab_id = uuid.uuid4().hex[:8]
        sess.tabs[tab_id] = TabHandle(tab_id=tab_id, page=page)
        sess.active_tab_id = tab_id
        sess.touch()
        return sess, sess.tabs[tab_id]

    async def close_tab(self, session_id: str, tab_id: str) -> None:
        sess = self._sessions.get(session_id)
        if not sess:
            raise KeyError(f"session_id '{session_id}' not found")
        tab = sess.tabs.pop(tab_id, None)
        if not tab:
            raise KeyError(f"tab_id '{tab_id}' not found")
        try:
            await tab.page.close()
        except Exception as e:
            logger.warning("close tab %s err: %s", tab_id, e)
        if sess.active_tab_id == tab_id:
            sess.active_tab_id = next(iter(sess.tabs), None)
        sess.touch()

    def get_page(
        self,
        session_id: Optional[str],
        tab_id: Optional[str],
    ) -> tuple[SessionHandle, TabHandle]:
        """获取 (session, tab)。session_id 必须已存在；tab_id 不传则用 active。"""
        if not session_id:
            # 找第一个 session（兼容懒人调用）
            if not self._sessions:
                raise KeyError(
                    "no active session; call /session/create or pass session_id"
                )
            sess = next(iter(self._sessions.values()))
        else:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError(f"session_id '{session_id}' not found")
        sess.touch()

        if tab_id:
            tab = sess.tabs.get(tab_id)
            if not tab:
                raise KeyError(f"tab_id '{tab_id}' not found")
        else:
            if not sess.active_tab_id:
                raise KeyError("session has no active tab")
            tab = sess.tabs[sess.active_tab_id]
        return sess, tab

    # ---------- 浏览器元信息 ----------
    @property
    def cdp_http_endpoint(self) -> str:
        return self._cdp_endpoint

    async def browser_version(self) -> str:
        if not self._browser:
            return ""
        return self._browser.version

    # ---------- 空闲回收 ----------
    async def _gc_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(60)
                now = time.time()
                async with self._lock:
                    idle = [
                        s.session_id
                        for s in self._sessions.values()
                        if now - s.last_active > settings.session_idle_timeout
                    ]
                for sid in idle:
                    logger.info("GC idle session %s", sid)
                    await self.close_session(sid)
        except asyncio.CancelledError:
            return


# 单例
manager = BrowserManager()
