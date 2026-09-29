"""Verified change + the analyst review queue and audit log."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.core.review import get_review_store
from app.core.verify import verify_aoi
from app.models.archive import (
    AuditEntry,
    ChangeVerification,
    ReviewDecision,
    ReviewItem,
    ReviewState,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])

_AOI_KEY = re.compile(r"^[a-z0-9_]{3,80}$")


@router.post("/verify/{aoi_key}", response_model=ChangeVerification)
async def verify(aoi_key: str, refresh: bool = Query(default=False)) -> ChangeVerification:
    """Run the change→confounder→verdict pipeline for an AOI and queue it."""
    if not _AOI_KEY.match(aoi_key):
        raise HTTPException(status_code=400, detail="Malformed AOI key.")
    try:
        verification = await run_in_threadpool(verify_aoi, aoi_key, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("verification failed for %s", aoi_key)
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    # Seed / refresh the review queue candidate.
    try:
        await run_in_threadpool(get_review_store().upsert_from_verification, verification)
    except Exception:  # noqa: BLE001
        logger.debug("review upsert failed", exc_info=True)
    return verification


@router.get("/review", response_model=ReviewState)
async def review_state() -> ReviewState:
    return await run_in_threadpool(get_review_store().state)


@router.post("/review/{item_id}/decision", response_model=ReviewItem)
async def review_decision(item_id: str, decision: ReviewDecision) -> ReviewItem:
    if not _AOI_KEY.match(item_id):
        raise HTTPException(status_code=400, detail="Malformed item id.")
    try:
        return await run_in_threadpool(
            get_review_store().decide, item_id, decision.decision,
            actor=decision.actor, note=decision.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No review item '{item_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/audit", response_model=list[AuditEntry])
async def audit_log() -> list[AuditEntry]:
    state = await run_in_threadpool(get_review_store().state)
    return state.audit


@router.post("/review/reset")
async def review_reset() -> dict:
    await run_in_threadpool(get_review_store().reset)
    return {"ok": True}
