"""Archive routes: browse the searchable tile catalogue and serve thumbnails."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.core.archive import build_archive, get_index, thumbnail_path
from app.models.archive import ArchiveAOI, ArchiveStats, ArchiveTile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/archive", tags=["archive"])

_TILE_ID = re.compile(r"^[a-z0-9_]{3,80}$")


@router.get("")
async def archive_overview() -> dict:
    """The archive as browsable AOIs plus summary stats."""
    index = await run_in_threadpool(get_index)
    return {
        "stats": index.stats.model_dump(),
        "aois": [a.model_dump() for a in index.aois.values()],
    }


@router.get("/stats", response_model=ArchiveStats)
async def archive_stats() -> ArchiveStats:
    index = await run_in_threadpool(get_index)
    return index.stats


@router.get("/tiles", response_model=list[ArchiveTile])
async def archive_tiles() -> list[ArchiveTile]:
    index = await run_in_threadpool(get_index)
    return list(index.tiles.values())


@router.get("/aois/{aoi_key}", response_model=ArchiveAOI)
async def archive_aoi(aoi_key: str) -> ArchiveAOI:
    index = await run_in_threadpool(get_index)
    aoi = index.aois.get(aoi_key)
    if aoi is None:
        raise HTTPException(status_code=404, detail=f"Unknown AOI '{aoi_key}'.")
    return aoi


@router.post("/rebuild", response_model=ArchiveStats)
async def archive_rebuild() -> ArchiveStats:
    """Rescan the source imagery and rebuild the index + thumbnails."""
    return await run_in_threadpool(build_archive, True)


@router.get("/thumb/{tile_file}")
async def archive_thumbnail(tile_file: str) -> FileResponse:
    """Serve a rendered tile thumbnail (PNG)."""
    tile_id = tile_file[:-4] if tile_file.endswith(".png") else tile_file
    if not _TILE_ID.match(tile_id):
        raise HTTPException(status_code=400, detail="Malformed tile id.")
    path = thumbnail_path(tile_id)
    if path is None:
        # Build the index (which renders thumbnails) then retry once.
        await run_in_threadpool(get_index)
        path = thumbnail_path(tile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No such thumbnail.")
    return FileResponse(path, media_type="image/png")
