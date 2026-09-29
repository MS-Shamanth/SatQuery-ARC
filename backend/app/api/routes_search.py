"""Semantic archive search."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.core.review import get_review_store
from app.core.search import search as run_search
from app.models.archive import SearchQuery, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(query: SearchQuery) -> SearchResponse:
    response = await run_in_threadpool(run_search, query)
    # Record the search in the audit log so the analyst workflow is traceable.
    if query.mode == "text" and query.text.strip():
        detail = f'"{query.text.strip()}" -> {response.matched} matches'
    elif query.tile_id:
        detail = f"find more like {query.tile_id} -> {response.matched} matches"
    else:
        detail = f"{response.matched} matches"
    try:
        get_review_store().log("searched", query.text.strip() or "archive", detail)
    except Exception:  # noqa: BLE001 - the search result matters more than the log
        logger.debug("audit log for search failed", exc_info=True)
    return response
