"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- 通用 ----------
class BaseResp(BaseModel):
    success: bool = True
    message: str = "ok"


class SessionScoped(BaseModel):
    """带 session/tab 上下文的请求基类。"""

    session_id: Optional[str] = Field(
        default=None, description="会话 ID；不传则使用/创建默认会话"
    )
    tab_id: Optional[str] = Field(
        default=None, description="tab ID；不传则使用当前活动 tab"
    )


# ---------- Session ----------
class CreateSessionResp(BaseResp):
    session_id: str
    tab_id: str
    cdp_url: str


class SessionInfo(BaseModel):
    session_id: str
    cdp_url: str
    tabs: list[str]
    active_tab: str


class ListSessionsResp(BaseResp):
    sessions: list[SessionInfo]


# ---------- Tab ----------
class NewTabReq(BaseModel):
    session_id: Optional[str] = None
    url: Optional[str] = None


class NewTabResp(BaseResp):
    session_id: str
    tab_id: str


class CloseTabReq(SessionScoped):
    pass


class ListTabsResp(BaseResp):
    session_id: str
    active_tab: str
    tabs: list[dict]


# ---------- 浏览器操作 ----------
class NavigateReq(SessionScoped):
    url: str
    wait_until: str = Field(
        default="load",
        pattern="^(load|domcontentloaded|networkidle|commit)$",
        description="load|domcontentloaded|networkidle|commit",
    )
    timeout_ms: int = Field(default=30000, ge=1000, le=1800000)


class NavigateResp(BaseResp):
    url: str
    title: str
    session_id: Optional[str] = None
    tab_id: Optional[str] = None


class GetTextReq(SessionScoped):
    selector: Optional[str] = Field(
        default=None, description="CSS 选择器；不传则取整页 body innerText"
    )


class GetTextResp(BaseResp):
    text: str


class GetHtmlReq(SessionScoped):
    selector: Optional[str] = None


class GetHtmlResp(BaseResp):
    html: str


class ScreenshotReq(SessionScoped):
    full_page: bool = False
    selector: Optional[str] = None
    image_format: str = Field(
        default="png",
        pattern="^(png|jpeg)$",
        description="png|jpeg",
    )


class ScreenshotResp(BaseResp):
    image_base64: str
    image_format: str


class ClickReq(SessionScoped):
    selector: str
    timeout_ms: int = Field(default=10000, ge=1000, le=1800000)


class TypeReq(SessionScoped):
    selector: str
    text: str
    clear: bool = True
    press_enter: bool = False
    delay_ms: int = Field(default=0, ge=0, le=1000)


class ScrollReq(SessionScoped):
    direction: str = Field(
        default="down",
        pattern="^(up|down|top|bottom)$",
        description="up|down|top|bottom",
    )
    pixels: int = Field(default=600, ge=0, le=200000)


class EvalReq(SessionScoped):
    script: str = Field(description="要执行的 JS 表达式或 () => {...} 函数体")


class EvalResp(BaseResp):
    result: Any


class CDPInfoResp(BaseResp):
    cdp_url: str
    browser_version: str


class TitleResp(BaseResp):
    title: str
    url: str
