"""HTTP 路由。"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from playwright.async_api import Error as PlaywrightError

from .browser import manager
from .config import settings
from .schemas import (
    BaseResp,
    CDPInfoResp,
    ClickReq,
    CloseTabReq,
    CreateSessionResp,
    EvalReq,
    EvalResp,
    GetHtmlReq,
    GetHtmlResp,
    GetTextReq,
    GetTextResp,
    ListSessionsResp,
    ListTabsResp,
    NavigateReq,
    NavigateResp,
    NewTabReq,
    NewTabResp,
    ScreenshotReq,
    ScreenshotResp,
    ScrollReq,
    SessionInfo,
    TitleResp,
    TypeReq,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _playwright_message(exc: PlaywrightError) -> str:
    return str(exc).splitlines()[0].strip()


def _download_navigation_message(url: str) -> str:
    return f"navigation started a download; browser cannot render this URL: {url}"


def _playwright_http_error(action: str, exc: PlaywrightError) -> HTTPException:
    return HTTPException(502, f"{action} failed: {_playwright_message(exc)}")


# ---------- 鉴权 ----------
async def auth_dep(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(401, "invalid token")


# ---------- 健康检查 ----------
@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "browser_running": manager.cdp_http_endpoint != "",
        "sessions": len(manager.list_sessions()),
    }


# ---------- Session ----------
@router.post(
    "/session/create",
    response_model=CreateSessionResp,
    dependencies=[Depends(auth_dep)],
)
async def create_session() -> CreateSessionResp:
    sess = await manager.create_session()
    return CreateSessionResp(
        session_id=sess.session_id,
        tab_id=sess.active_tab_id or "",
        cdp_url=manager.cdp_http_endpoint,
    )


@router.get(
    "/session/list",
    response_model=ListSessionsResp,
    dependencies=[Depends(auth_dep)],
)
async def list_sessions() -> ListSessionsResp:
    out = []
    for s in manager.list_sessions():
        out.append(
            SessionInfo(
                session_id=s.session_id,
                cdp_url=manager.cdp_http_endpoint,
                tabs=list(s.tabs.keys()),
                active_tab=s.active_tab_id or "",
            )
        )
    return ListSessionsResp(sessions=out)


@router.delete(
    "/session/{session_id}",
    response_model=BaseResp,
    dependencies=[Depends(auth_dep)],
)
async def delete_session(session_id: str) -> BaseResp:
    await manager.close_session(session_id)
    return BaseResp()


# ---------- Tab ----------
@router.post(
    "/tab/new", response_model=NewTabResp, dependencies=[Depends(auth_dep)]
)
async def new_tab(req: NewTabReq) -> NewTabResp:
    try:
        sess, tab = await manager.new_tab(req.session_id, req.url)
    except PlaywrightError as exc:
        msg = _playwright_message(exc)
        if "Download is starting" in str(exc):
            raise HTTPException(422, _download_navigation_message(req.url or "")) from exc
        raise HTTPException(502, f"new_tab failed: {msg}") from exc
    return NewTabResp(session_id=sess.session_id, tab_id=tab.tab_id)


@router.post(
    "/tab/close", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def close_tab(req: CloseTabReq) -> BaseResp:
    if not req.session_id or not req.tab_id:
        raise HTTPException(400, "session_id and tab_id required")
    await manager.close_tab(req.session_id, req.tab_id)
    return BaseResp()


@router.get(
    "/tab/list/{session_id}",
    response_model=ListTabsResp,
    dependencies=[Depends(auth_dep)],
)
async def list_tabs(session_id: str) -> ListTabsResp:
    try:
        sess, _ = manager.get_page(session_id, None)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    async def _tab_info(tid: str, tab) -> dict:
        try:
            return {"tab_id": tid, "url": tab.page.url, "title": await tab.page.title()}
        except Exception:
            return {"tab_id": tid, "url": "", "title": ""}

    tabs = await asyncio.gather(*[_tab_info(tid, t) for tid, t in sess.tabs.items()])
    return ListTabsResp(
        session_id=sess.session_id,
        active_tab=sess.active_tab_id or "",
        tabs=tabs,
    )


# ---------- 浏览器操作 ----------
def _resolve(req: Any):
    try:
        return manager.get_page(
            getattr(req, "session_id", None), getattr(req, "tab_id", None)
        )
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post(
    "/browser/navigate",
    response_model=NavigateResp,
    dependencies=[Depends(auth_dep)],
)
async def navigate(req: NavigateReq) -> NavigateResp:
    created_session_id: str | None = None
    if not req.session_id:
        sess = await manager.create_session()
        tab = sess.tabs[sess.active_tab_id]  # type: ignore[index]
        created_session_id = sess.session_id
    else:
        sess, tab = _resolve(req)
    try:
        await tab.page.goto(req.url, wait_until=req.wait_until, timeout=req.timeout_ms)
    except PlaywrightError as exc:
        if created_session_id:
            await manager.close_session(created_session_id)
        if "Download is starting" in str(exc):
            logger.info("Download navigation skipped for %s", req.url)
            return NavigateResp(
                success=False,
                message=_download_navigation_message(req.url),
                url=req.url,
                title="",
            )
        raise _playwright_http_error("navigate", exc) from exc
    return NavigateResp(
        url=tab.page.url,
        title=await tab.page.title(),
        session_id=sess.session_id,
        tab_id=tab.tab_id,
    )


@router.post(
    "/browser/get_text",
    response_model=GetTextResp,
    dependencies=[Depends(auth_dep)],
)
async def get_text(req: GetTextReq) -> GetTextResp:
    _, tab = _resolve(req)
    try:
        if req.selector:
            loc = tab.page.locator(req.selector).first
            try:
                text = await loc.inner_text(timeout=3000)
            except PlaywrightError:
                text = await loc.text_content(timeout=3000) or ""
        else:
            try:
                text = await tab.page.evaluate("() => document.body ? document.body.innerText : ''")
            except PlaywrightError:
                loc = tab.page.locator("body").first
                try:
                    text = await loc.inner_text(timeout=3000)
                except PlaywrightError:
                    text = await loc.text_content(timeout=3000) or ""
    except PlaywrightError as exc:
        raise _playwright_http_error("get_text", exc) from exc
    return GetTextResp(text=text)


@router.post(
    "/browser/get_html",
    response_model=GetHtmlResp,
    dependencies=[Depends(auth_dep)],
)
async def get_html(req: GetHtmlReq) -> GetHtmlResp:
    _, tab = _resolve(req)
    try:
        if req.selector:
            loc = tab.page.locator(req.selector).first
            html = await loc.evaluate("el => el.outerHTML")
        else:
            html = await tab.page.content()
    except PlaywrightError as exc:
        raise _playwright_http_error("get_html", exc) from exc
    return GetHtmlResp(html=html)


@router.post(
    "/browser/screenshot",
    response_model=ScreenshotResp,
    dependencies=[Depends(auth_dep)],
)
async def screenshot(req: ScreenshotReq) -> ScreenshotResp:
    _, tab = _resolve(req)
    fmt = req.image_format.lower()
    if fmt not in {"png", "jpeg"}:
        raise HTTPException(400, "image_format must be png|jpeg")

    try:
        if req.selector:
            loc = tab.page.locator(req.selector).first
            img_bytes = await loc.screenshot(type=fmt)  # type: ignore[arg-type]
        else:
            img_bytes = await tab.page.screenshot(
                type=fmt,  # type: ignore[arg-type]
                full_page=req.full_page,
            )
    except PlaywrightError as exc:
        raise _playwright_http_error("screenshot", exc) from exc
    return ScreenshotResp(
        image_base64=base64.b64encode(img_bytes).decode("utf-8"),
        image_format=fmt,
    )


@router.post(
    "/browser/click", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def click(req: ClickReq) -> BaseResp:
    _, tab = _resolve(req)
    try:
        await tab.page.locator(req.selector).first.click(timeout=req.timeout_ms)
    except PlaywrightError as exc:
        raise _playwright_http_error("click", exc) from exc
    return BaseResp()


@router.post(
    "/browser/type", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def type_text(req: TypeReq) -> BaseResp:
    _, tab = _resolve(req)
    loc = tab.page.locator(req.selector).first
    try:
        if req.clear:
            await loc.fill("")
        if req.delay_ms > 0:
            await loc.type(req.text, delay=req.delay_ms)
        else:
            await loc.fill(req.text)
        if req.press_enter:
            await loc.press("Enter")
    except PlaywrightError as exc:
        raise _playwright_http_error("type", exc) from exc
    return BaseResp()


@router.post(
    "/browser/scroll", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def scroll(req: ScrollReq) -> BaseResp:
    _, tab = _resolve(req)
    direction = req.direction.lower()
    try:
        if direction == "down":
            await tab.page.evaluate(f"window.scrollBy(0, {req.pixels})")
        elif direction == "up":
            await tab.page.evaluate(f"window.scrollBy(0, -{req.pixels})")
        elif direction == "top":
            await tab.page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await tab.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            raise HTTPException(400, "direction must be up|down|top|bottom")
    except PlaywrightError as exc:
        raise _playwright_http_error("scroll", exc) from exc
    return BaseResp()


@router.post(
    "/browser/eval", response_model=EvalResp, dependencies=[Depends(auth_dep)]
)
async def evaluate_js(req: EvalReq) -> EvalResp:
    _, tab = _resolve(req)
    try:
        result = await tab.page.evaluate(req.script)
    except Exception as e:
        raise HTTPException(400, f"eval error: {e}") from e
    # 保证 JSON 可序列化
    try:
        import json

        json.dumps(result)
    except (TypeError, ValueError):
        result = repr(result)
    return EvalResp(result=result)


@router.get(
    "/browser/title", response_model=TitleResp, dependencies=[Depends(auth_dep)]
)
async def get_title(
    session_id: str | None = None, tab_id: str | None = None
) -> TitleResp:
    try:
        _, tab = manager.get_page(session_id, tab_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return TitleResp(title=await tab.page.title(), url=tab.page.url)


@router.get(
    "/browser/cdp_url",
    response_model=CDPInfoResp,
    dependencies=[Depends(auth_dep)],
)
async def cdp_url() -> CDPInfoResp:
    if not manager.cdp_http_endpoint:
        await manager.start()
    return CDPInfoResp(
        cdp_url=manager.cdp_http_endpoint,
        browser_version=await manager.browser_version(),
    )
