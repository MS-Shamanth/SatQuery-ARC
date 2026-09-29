"""Analysis Contract route.

The contract is returned, not executed. Showing what will be done before doing it
is the point, so acceptance is a separate step that arrives with the orchestrator.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path as PathParam
from starlette.concurrency import run_in_threadpool

from app.core.contract import generate_contract
from app.core.readiness import evaluate_readiness
from app.core.registry import get_registry
from app.core.sessions import SessionNotFound, get_store
from app.models.contract import AnalysisContract, ContractRequest
from app.models.schemas import InputConfiguration
from app.tools.base import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["contract"])

SESSION_ID = PathParam(
    description="32-character hexadecimal session id", pattern="^[0-9a-f]{32}$"
)


@router.post("/{session_id}/contract", response_model=AnalysisContract)
async def create_contract(
    request: ContractRequest, session_id: str = SESSION_ID
) -> AnalysisContract:
    """Draft the analysis contract for a query, without running anything."""
    store = get_store()
    try:
        record = store.load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record.configuration is InputConfiguration.INCOMPLETE:
        raise HTTPException(
            status_code=409,
            detail=(
                "The input configuration is not complete, so there is nothing to "
                "plan against. Fill every slot of the selected configuration first."
            ),
        )

    def draft() -> AnalysisContract:
        """The blocking part: the readiness gate, then the model."""
        readiness = evaluate_readiness(record, store)
        context = ToolContext(session=record, store=store, readiness=readiness)
        return generate_contract(
            request.query,
            context,
            get_registry(),
            force_offline=request.force_offline,
            refresh=request.refresh,
        )

    try:
        # Off the event loop, deliberately.
        #
        # Both halves of this block: the readiness gate reads rasters, and the
        # provider call is synchronous httpx that waits on a remote model. Run
        # inline in an async endpoint they froze the entire server for as long as
        # the model took, so /api/health timed out and the UI looked dead while it
        # was only drafting a plan. Nothing here touches the event loop, so a
        # thread is all it needs.
        return await run_in_threadpool(draft)
    except Exception as exc:  # noqa: BLE001
        logger.exception("contract generation failed for session %s", session_id)
        raise HTTPException(
            status_code=500, detail=f"Could not draft a contract: {exc}"
        ) from exc
