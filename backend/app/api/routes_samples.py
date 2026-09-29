"""Sample scene library routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path as PathParam
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.samples import (
    SampleNotAvailable,
    load_manifest,
    load_sample_into_session,
)
from app.core.sessions import get_store
from app.models.schemas import SampleManifest, SessionRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/samples", tags=["samples"])

SAMPLE_KEY = PathParam(description="Sample scene key", pattern="^[a-z0-9_]{3,40}$")


@router.get("", response_model=SampleManifest)
async def list_samples() -> SampleManifest:
    """The curated scene library.

    Scenes whose cached files are absent are still listed, with an empty asset
    map, so the UI can explain that the cache needs populating rather than
    silently hiding a demo.
    """
    return load_manifest(get_settings().samples_dir)


@router.post("/{key}/load", response_model=SessionRecord, status_code=201)
async def load_sample(key: str = SAMPLE_KEY) -> SessionRecord:
    """Create a session pre-loaded with a sample scene."""
    settings = get_settings()
    try:
        # Copies and reads the scene's rasters, so off the loop like the rest of
        # the ingest path. This is the first request of every demo.
        return await run_in_threadpool(
            load_sample_into_session, key, get_store(), settings.samples_dir
        )
    except SampleNotAvailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to load sample %s", key)
        raise HTTPException(
            status_code=500, detail=f"Could not load sample '{key}': {exc}"
        ) from exc
