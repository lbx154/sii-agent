"""HTTP routes for the search-proxy service."""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from . import upstream
from .config import settings
from .schemas import (
    FetchReq,
    FetchResp,
    HealthResp,
    SearchHit,
    SearchImageReq,
    SearchResp,
    SearchTextReq,
    UploadResp,
)

logger = logging.getLogger("search-proxy.routes")
router = APIRouter()
_JINA_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.jina_fetch_workers))
_PROXY_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.proxy_workers))


# ---------- auth ----------
async def auth_dep(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(401, "invalid token")


# ---------- health ----------
@router.get("/health", response_model=HealthResp)
async def health() -> HealthResp:
    return HealthResp(
        status="ok",
        serper_configured=bool(settings.serper_api_key),
        jina_configured=bool(settings.jina_api_key),
    )


# ---------- search ----------
def _relaxed_search_query(query: str) -> str:
    value = " ".join(str(query or "").split())
    if not value:
        return ""
    if re.search(r"\b(?:site|filetype|intitle|inurl):", value, flags=re.IGNORECASE):
        return value
    value = re.sub(r"\b(?:JsonObject|json|answer|official answer|reference answer)\b", " ", value, flags=re.IGNORECASE)
    value = value.replace('"', " ")
    value = re.sub(r"\b(?:OR|AND)\b", " ", value)
    value = re.sub(r"[(){}\[\]]", " ", value)
    tokens = [token for token in value.split() if token]
    if len(tokens) <= 3:
        return " ".join(tokens)
    return " ".join(tokens[:12])[:220].strip()


def _attach_jina_content(
    hits: list[SearchHit],
    fetch: bool,
    max_chars: int,
    fetch_top_k: int | None = None,
) -> None:
    if not fetch:
        return
    limit = len(hits) if fetch_top_k is None else max(0, min(fetch_top_k, len(hits)))
    candidates = [hit for hit in hits[:limit] if hit.url]
    if not candidates:
        return
    future_to_hit = {
        _JINA_EXECUTOR.submit(upstream.jina_fetch, hit.url, max_chars, settings.jina_fetch_timeout): hit
        for hit in candidates
    }
    completed: set = set()
    try:
        futures = as_completed(future_to_hit, timeout=settings.jina_fetch_budget)
        for future in futures:
            completed.add(future)
            hit = future_to_hit[future]
            try:
                content, _ = future.result()
                hit.content = content
            except Exception as exc:  # noqa: BLE001
                hit.content = f"[jina-error] {type(exc).__name__}: {exc}"
    except FuturesTimeoutError:
        pass
    for future, hit in future_to_hit.items():
        if future not in completed:
            future.cancel()
            hit.content = f"[jina-timeout] exceeded {settings.jina_fetch_budget:.1f}s fetch budget"


def _search_text_sync(req: SearchTextReq) -> SearchResp:
    try:
        organic = upstream.serper_search(req.query, req.top_k)
        repaired_query = ""
        if not organic and settings.query_repair:
            relaxed = _relaxed_search_query(req.query)
            if relaxed and relaxed != " ".join(req.query.split()):
                repaired = upstream.serper_search(relaxed, req.top_k)
                if repaired:
                    organic = repaired
                    repaired_query = relaxed
    except Exception as exc:  # noqa: BLE001
        return SearchResp(ok=False, error=f"{type(exc).__name__}: {exc}")

    hits: list[SearchHit] = []
    for rank, item in enumerate(organic[: req.top_k], start=1):
        url = item.get("link") or ""
        snippet = item.get("snippet", "")
        if repaired_query:
            snippet = f"[query-repair fallback: {req.query!r} -> {repaired_query!r}] {snippet}".strip()
        hit = SearchHit(
            rank=rank,
            title=item.get("title", ""),
            url=url,
            snippet=snippet,
        )
        hits.append(hit)
    _attach_jina_content(hits, req.fetch, req.max_chars, req.fetch_top_k)
    return SearchResp(ok=True, results=hits)


def _search_image_sync(req: SearchImageReq) -> SearchResp:
    try:
        items = upstream.serper_lens(req.image_url, req.top_k)
    except Exception as exc:  # noqa: BLE001
        return SearchResp(ok=False, error=f"{type(exc).__name__}: {exc}")

    hits: list[SearchHit] = []
    for rank, item in enumerate(items[: req.top_k], start=1):
        url = item.get("link") or item.get("url") or ""
        hit = SearchHit(
            rank=rank,
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", "") or item.get("source", ""),
        )
        hits.append(hit)
    _attach_jina_content(hits, req.fetch, req.max_chars, req.fetch_top_k)
    return SearchResp(ok=True, results=hits)


@router.post(
    "/search/text", response_model=SearchResp, dependencies=[Depends(auth_dep)]
)
async def search_text(req: SearchTextReq) -> SearchResp:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PROXY_EXECUTOR, _search_text_sync, req)


@router.post(
    "/search/image", response_model=SearchResp, dependencies=[Depends(auth_dep)]
)
async def search_image(req: SearchImageReq) -> SearchResp:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PROXY_EXECUTOR, _search_image_sync, req)


@router.post("/fetch", response_model=FetchResp, dependencies=[Depends(auth_dep)])
def fetch_url(req: FetchReq) -> FetchResp:
    """Standalone Jina fetch — handy for manual debugging from the GPU side."""
    try:
        content, truncated = upstream.jina_fetch(req.url, req.max_chars)
        return FetchResp(ok=True, url=req.url, content=content, truncated=truncated)
    except Exception as exc:  # noqa: BLE001
        return FetchResp(
            ok=False, url=req.url, error=f"{type(exc).__name__}: {exc}"
        )


# ---------- image upload ----------
@router.post(
    "/upload_image", response_model=UploadResp, dependencies=[Depends(auth_dep)]
)
async def upload_image(file: UploadFile = File(...), filename: str | None = Form(default=None)) -> UploadResp:
    """Upload a local image (sent from GPU side) to a public host (e.g. 0x0.st)
    and return the resulting public URL — used as input to /search/image."""
    try:
        data = await file.read()
        if not data:
            return UploadResp(ok=False, error="empty file")
        public_url = await run_in_threadpool(
            upstream.upload_image,
            data, filename or file.filename or "image.bin"
        )
        return UploadResp(ok=True, url=public_url)
    except Exception as exc:  # noqa: BLE001
        return UploadResp(ok=False, error=f"{type(exc).__name__}: {exc}")
