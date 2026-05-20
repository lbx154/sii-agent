"""Pydantic schemas for the search-proxy HTTP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SearchTextReq(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    fetch: bool = True
    max_chars: int = Field(default=5000, ge=0, le=20000)
    fetch_top_k: Optional[int] = Field(default=None, ge=0, le=10)


class SearchImageReq(BaseModel):
    image_url: str = Field(description="Public http(s) URL of the image")
    top_k: int = Field(default=5, ge=1, le=10)
    fetch: bool = True
    max_chars: int = Field(default=5000, ge=0, le=20000)
    fetch_top_k: Optional[int] = Field(default=None, ge=0, le=10)


class FetchReq(BaseModel):
    url: str
    max_chars: int = Field(default=5000, ge=0, le=20000)


class SearchHit(BaseModel):
    rank: int
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: Optional[str] = None


class SearchResp(BaseModel):
    ok: bool = True
    results: list[SearchHit] = Field(default_factory=list)
    error: Optional[str] = None


class FetchResp(BaseModel):
    ok: bool = True
    url: str = ""
    content: str = ""
    truncated: bool = False
    error: Optional[str] = None


class UploadResp(BaseModel):
    ok: bool = True
    url: str = ""
    error: Optional[str] = None


class HealthResp(BaseModel):
    status: str = "ok"
    serper_configured: bool = False
    jina_configured: bool = False
