"""Readiness gate route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path as PathParam
from starlette.concurrency import run_in_threadpool

from app.core.readiness import evaluate_readiness
from app.core.sessions import SessionNotFound, get_store
from app.models.schemas import ReadinessReport

router = APIRouter(prefix="/sessions", tags=["readiness"])

SESSION_ID = PathParam(
    description="32-character hexadecimal session id", pattern="^[0-9a-f]{32}$"
)


@router.get("/{session_id}/readiness", response_model=ReadinessReport)
async def get_readiness(session_id: str = SESSION_ID) -> ReadinessReport:
    """Run the data readiness gate over the session's current inputs.

    Always returns 200 with a report. A refusal is a result, not an HTTP error:
    the client needs the measured values and the list of requirements in order to
    explain to the user what to do next.
    """
    store = get_store()
    try:
        record = store.load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # The gate opens every raster in the session and reads bands from it. On the
    # event loop that stalls every other request behind it, and the client fires
    # this one as soon as the first file lands, while uploads are still arriving.
    return await run_in_threadpool(evaluate_readiness, record, store)
