"""Execution routes: approve a contract, watch it run, read the trace.

Approval is a separate call from drafting on purpose. The contract is shown
first, and nothing touches a pixel until the user accepts it. The body of that
approval is only a hash, so the plan that executes is the plan that was
displayed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Path as PathParam
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from app.core.contract import (
    ContractStale,
    ContractUnavailable,
    contract_for_execution,
)
from app.core.readiness import evaluate_readiness
from app.core.registry import get_registry
from app.core.runs import RunConflict, RunNotFound, get_run_manager
from app.core.packet import MANIFEST_FILENAME
from app.core.sessions import SessionNotFound, get_store
from app.models.packet import EvidencePacket
from app.models.schemas import InputConfiguration
from app.models.trace import MapLayer, RunEvent, RunRequest, RunTrace
from app.tools.base import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["runs"])

SESSION_ID = PathParam(
    description="32-character hexadecimal session id", pattern="^[0-9a-f]{32}$"
)
RUN_ID = PathParam(
    description="32-character hexadecimal run id", pattern="^[0-9a-f]{32}$"
)

# Comment lines every few seconds keep intermediaries from closing an idle
# stream while a slow tool is working.
STREAM_PING_SECONDS = 10


def encode_event(event: RunEvent) -> str:
    """One streamed event as JSON, with nulls kept.

    A named function so a test can assert the real wire format rather than a
    reimplementation of it.

    Nulls are sent, not dropped. This once serialised with ``exclude_none=True``,
    which is recursive: a trace with no verdict, no packet and no disagreement yet
    arrived with those keys absent rather than null. The client's types declare
    them nullable, so every panel guarding on ``=== null`` sailed past an
    ``undefined`` and threw on the next line. That is how a run in progress showed
    two red panel errors over a result that was perfectly fine.

    The cost is a few null fields per event. The alternative is a wire format that
    contradicts the types on both sides of it.
    """
    return event.model_dump_json()


def _context(session_id: str) -> ToolContext:
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
                "run against."
            ),
        )

    readiness = evaluate_readiness(record, store)
    return ToolContext(session=record, store=store, readiness=readiness)


@router.post("/{session_id}/runs", response_model=RunTrace, status_code=202)
async def start_run(request: RunRequest, session_id: str = SESSION_ID) -> RunTrace:
    """Execute an approved contract. Returns at once with the pending trace.

    The response carries the full stage list before anything has happened, so the
    client can draw the pipeline and then light it up from the event stream.
    """
    # _context runs the readiness gate, which reads every raster in the session,
    # and contract_for_execution reads the cached plan off disk. Both are blocking
    # and neither touches the event loop, so they belong in a thread: this is the
    # request the user makes by pressing approve, and stalling the loop here also
    # stalls the event stream that is about to report the run.
    registry = get_registry()
    context = await run_in_threadpool(_context, session_id)

    try:
        contract = await run_in_threadpool(
            contract_for_execution, request.contract_hash, context, registry
        )
    except ContractUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContractStale as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    manager = get_run_manager()
    try:
        run = await manager.start(contract, context, registry)
    except RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return run.trace


@router.get("/{session_id}/runs/{run_id}", response_model=RunTrace)
async def read_trace(session_id: str = SESSION_ID, run_id: str = RUN_ID) -> RunTrace:
    """The current trace, live from memory or replayed from disk."""
    manager = get_run_manager()
    try:
        run = manager.get(run_id)
    except RunNotFound as exc:
        stored = manager.load_trace(session_id, run_id)
        if stored is None:
            raise HTTPException(
                status_code=404, detail=f"No such run: {run_id}"
            ) from exc
        return stored

    if run.session_id != session_id:
        raise HTTPException(
            status_code=404, detail="That run belongs to a different session."
        )
    return run.trace


@router.get("/{session_id}/runs/{run_id}/stream")
async def stream_run(session_id: str = SESSION_ID, run_id: str = RUN_ID):
    """Server-sent events for a run, from the beginning.

    History is replayed before live events, so opening this late, or reopening it
    after a dropped connection, still shows the whole run.
    """
    manager = get_run_manager()
    try:
        run = manager.get(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc

    if run.session_id != session_id:
        raise HTTPException(
            status_code=404, detail="That run belongs to a different session."
        )

    async def events() -> AsyncIterator[dict[str, str]]:
        async for event in manager.subscribe(run):
            yield {"id": str(event.seq), "data": encode_event(event)}

    return EventSourceResponse(events(), ping=STREAM_PING_SECONDS)


@router.post("/{session_id}/runs/{run_id}/cancel", response_model=RunTrace)
async def cancel_run(session_id: str = SESSION_ID, run_id: str = RUN_ID) -> RunTrace:
    """Ask a run to stop. It stops between steps, never mid-measurement."""
    manager = get_run_manager()
    try:
        run = manager.get(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}") from exc

    if run.session_id != session_id:
        raise HTTPException(
            status_code=404, detail="That run belongs to a different session."
        )
    return manager.request_cancel(run_id).trace


@router.get("/{session_id}/runs/{run_id}/layers", response_model=list[MapLayer])
async def read_layers(
    session_id: str = SESSION_ID, run_id: str = RUN_ID
) -> list[MapLayer]:
    """Map layers for a run, with the bounds that place them."""
    manager = get_run_manager()
    records = manager.load_layers(session_id, run_id)
    if not records:
        try:
            run = manager.get(run_id)
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail=f"No such run: {run_id}"
            ) from exc
        if run.is_running:
            # Layers are rendered once the run settles, so an empty list here
            # means "not yet" rather than "none".
            return []
        records = [layer.__dict__ for layer in run.layers]

    base = f"/api/sessions/{session_id}/runs/{run_id}/layers"
    layers: list[MapLayer] = []
    for record in records:
        geojson = record.get("geojson_filename")
        layers.append(
            MapLayer(
                key=record["key"],
                label=record["label"],
                description=record.get("description") or "",
                kind=record.get("kind", "mask"),
                png_url=f"{base}/{record['png_filename']}",
                geojson_url=f"{base}/{geojson}" if geojson else None,
                bounds_wgs84=record["bounds_wgs84"],
                colour=record.get("colour", "#22d3ee"),
                area_km2=record.get("area_km2"),
                pixel_count=record.get("pixel_count"),
                applies_to=record.get("applies_to") or [],
            )
        )
    # Base imagery first so the map stacks masks above it.
    layers.sort(key=lambda layer: 0 if layer.kind == "base" else 1)
    return layers


@router.get("/{session_id}/runs/{run_id}/layers/{filename}")
async def read_layer_file(
    filename: str = PathParam(pattern=r"^[A-Za-z0-9._-]{1,140}$"),
    session_id: str = SESSION_ID,
    run_id: str = RUN_ID,
):
    """Serve a rendered overlay or its vector footprint.

    The filename pattern excludes separators outright, so no client-supplied text
    can walk out of the run's directory.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".geojson"}:
        raise HTTPException(status_code=404, detail="Unknown layer file.")

    path = get_run_manager().layers_dir(session_id, run_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="That layer has not been rendered.")

    media = "image/png" if suffix == ".png" else "application/geo+json"
    return FileResponse(
        path,
        media_type=media,
        # A run's layers are immutable once written.
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{session_id}/runs/{run_id}/packet", response_model=EvidencePacket)
async def read_packet(
    session_id: str = SESSION_ID, run_id: str = RUN_ID
) -> EvidencePacket:
    """What was exported for this run, with a checksum for every file.

    Read from the packet's own manifest on disk rather than from the trace in
    memory, so the answer still describes the files that are actually there after
    the process that made them has gone.
    """
    manager = get_run_manager()
    path = manager.packet_dir(session_id, run_id) / MANIFEST_FILENAME
    if path.exists():
        try:
            return EvidencePacket.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:  # noqa: BLE001
            logger.warning("unreadable packet manifest for %s: %s", run_id, exc)

    trace = manager.load_trace(session_id, run_id)
    if trace is None:
        try:
            trace = manager.get(run_id).trace
        except RunNotFound as exc:
            raise HTTPException(
                status_code=404, detail=f"No such run: {run_id}"
            ) from exc
    if trace.packet is None:
        raise HTTPException(
            status_code=404,
            detail="No evidence packet was written for this run.",
        )
    return trace.packet


# The overlays and vectors already have a route; these are the packet's own
# additions. Kept to an allowlist of suffixes so no client-supplied name can ask
# for something the packet does not contain.
PACKET_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".csv": "text/csv",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".png": "image/png",
}


@router.get("/{session_id}/runs/{run_id}/packet/{filename}")
async def download_packet_file(
    filename: str = PathParam(pattern=r"^[A-Za-z0-9._-]{1,140}$"),
    session_id: str = SESSION_ID,
    run_id: str = RUN_ID,
):
    """Serve one file from the packet as a download."""
    media = PACKET_MEDIA_TYPES.get(Path(filename).suffix.lower())
    if media is None:
        raise HTTPException(status_code=404, detail="Unknown packet file.")

    path = get_run_manager().packet_dir(session_id, run_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="That file was not written.")

    return FileResponse(
        path,
        media_type=media,
        # Named after the run so several downloads do not collide in one folder.
        filename=f"satquery-{run_id[:8]}-{filename}",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{session_id}/runs", response_model=list[RunTrace])
async def list_runs(session_id: str = SESSION_ID) -> list[RunTrace]:
    """Traces for this session, most recent last."""
    manager = get_run_manager()
    directory = manager.runs_dir(session_id)
    traces: list[RunTrace] = []

    for path in sorted(directory.glob("*.json")):
        try:
            traces.append(RunTrace.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:  # noqa: PERF203
            logger.warning("skipping unreadable trace %s: %s", path.name, exc)

    # An in-flight run has no file yet, so it is added from memory.
    live = manager.active_for_session(session_id)
    if live is not None and all(t.run_id != live.run_id for t in traces):
        traces.append(live.trace)

    traces.sort(key=lambda trace: trace.started_at)
    return traces


__all__ = ["router"]
