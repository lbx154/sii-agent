"""FastAPI entrypoint for the search-proxy."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("search-proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401, ARG001
    logger.info(
        "search-proxy listening on %s:%d (auth=%s, serper=%s, jina=%s)",
        settings.host,
        settings.port,
        "on" if settings.api_token else "off",
        "set" if settings.serper_api_key else "missing",
        "set" if settings.jina_api_key else "anon",
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="search-proxy",
        version="0.1.0",
        description=(
            "HTTP proxy that lets an air-gapped GPU host run web search via "
            "an internet-connected CPU host. Designed to be reached over an "
            "SSH (or VS Code) port-forward."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
