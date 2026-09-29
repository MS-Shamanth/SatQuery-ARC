"""Tool registry routes.

Exposing the registry is what makes the execution trace verifiable: a reader can
check that every tool named in a run is a tool the system actually has, with the
version and the parameter whitelist it declares.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path as PathParam

from app.core.readiness import evaluate_readiness
from app.core.registry import get_registry
from app.core.sessions import SessionNotFound, get_store
from app.models.schemas import ToolDescriptor
from app.tools.base import ToolContext

router = APIRouter(tags=["tools"])

SESSION_ID = PathParam(
    description="32-character hexadecimal session id", pattern="^[0-9a-f]{32}$"
)


@router.get("/tools", response_model=list[ToolDescriptor])
async def list_tools() -> list[ToolDescriptor]:
    """Every registered tool, with its requirements and permitted parameters."""
    return get_registry().describe()


@router.get("/tools/declined")
async def list_declined_tools() -> dict[str, str]:
    """Tools that did not register, and why.

    Kept visible so an absent capability is explained rather than merely missing.
    """
    return get_registry().declined()


@router.get("/sessions/{session_id}/tools")
async def tools_for_session(session_id: str = SESSION_ID) -> dict[str, Any]:
    """Which tools can run on this session's inputs, with reasons for those that cannot."""
    store = get_store()
    try:
        record = store.load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    registry = get_registry()
    context = ToolContext(
        session=record,
        store=store,
        readiness=evaluate_readiness(record, store),
    )

    availability = registry.availability(context)
    return {
        "session_id": session_id,
        "configuration": record.configuration.value,
        "available": [item.tool_name for item in availability if item.available],
        "unavailable": [
            {"tool": item.tool_name, "reason": item.reason}
            for item in availability
            if not item.available
        ],
        "declined_registration": registry.declined(),
    }
