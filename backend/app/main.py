"""SatQuery ARC - FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_archive,
    routes_contract,
    routes_health,
    routes_readiness,
    routes_review,
    routes_runs,
    routes_samples,
    routes_search,
    routes_sessions,
    routes_tools,
)
from app.config import APP_NAME, APP_VERSION, get_settings
from app.core.credentials import registry

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Probe external dependencies in the background at startup.

    Probing is fire-and-forget so a slow or unreachable provider cannot delay
    the server from accepting requests. ``/api/health`` reports ``unknown``
    until the first round of probes lands.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)
    settings.ensure_dirs()

    logger.info("%s %s starting (%s)", APP_NAME, APP_VERSION, settings.satquery_env)
    # Names the provider that will actually be used, not every key that happens
    # to be present. ``selected_provider`` is "none" when the configured provider
    # has no key, which is the case worth seeing in a startup line.
    logger.info(
        "credentials present - llm:%s copernicus:%s aws:%s",
        settings.selected_provider,
        settings.has_cdse,
        settings.has_aws,
    )

    probe_task = asyncio.create_task(registry.refresh(settings))

    async def _warm_archive() -> None:
        """Build the searchable archive index in the background at startup so the
        first search does not pay for a cold scan. Never fatal."""
        try:
            from app.core.archive import get_index

            await asyncio.to_thread(get_index)
        except Exception:  # noqa: BLE001
            logger.warning("archive warm-up failed; it will build on first use", exc_info=True)

    archive_task = asyncio.create_task(_warm_archive())
    try:
        yield
    finally:
        for task in (probe_task, archive_task):
            if not task.done():
                task.cancel()
        logger.info("%s shutting down", APP_NAME)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "Archive Search & Verified Change. Searches a growing multi-sensor "
            "satellite archive by meaning (free text, an example tile, or "
            "find-more-like-this), then verifies every candidate change against "
            "season, cloud and misregistration before it reaches an analyst - "
            "fully offline, with full provenance."
        ),
        lifespan=lifespan,
        # Served under /api so the Vite dev proxy, which only forwards /api,
        # can reach the documentation on the frontend's own origin.
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_sessions.router, prefix="/api")
    app.include_router(routes_readiness.router, prefix="/api")
    app.include_router(routes_samples.router, prefix="/api")
    app.include_router(routes_tools.router, prefix="/api")
    app.include_router(routes_contract.router, prefix="/api")
    app.include_router(routes_runs.router, prefix="/api")
    # SatQuery ARC: archive search, verified change, analyst review.
    app.include_router(routes_archive.router, prefix="/api")
    app.include_router(routes_search.router, prefix="/api")
    app.include_router(routes_review.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "app": APP_NAME,
            "version": APP_VERSION,
            "docs": "/api/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
