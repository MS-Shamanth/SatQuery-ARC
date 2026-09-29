"""Run lifecycle: threading, streaming, and persistence.

The orchestrator is synchronous numpy work. Running it on the event loop would
stall every other request including the stream's own keepalive, so a run executes
in a worker thread and its events are handed back to the loop.

Two properties are worth stating because they are what make the live trace
trustworthy rather than decorative:

* Every event is kept in order on the run, so a client that opens the stream
  after the run started, or reopens it after a dropped connection, receives the
  full history and then continues live. Nothing is lost by watching late.
* A run completes whether or not anyone is watching, and its trace is written to
  disk. Closing the tab does not abandon the work, and a backend restart does not
  erase the record.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from app.core.orchestrator import execute_run
from app.core.registry import ToolRegistry
from app.core.sessions import SessionStore
from app.models.contract import AnalysisContract
from app.models.trace import RunEvent, RunEventType, RunStatus, RunTrace
from app.tools.base import ToolContext, ToolOutcome

logger = logging.getLogger(__name__)

# Completed runs stay in memory so the map workspace can reach the arrays their
# tools produced. Each one holds rasters, so the number is bounded and the oldest
# is dropped rather than letting a long session grow without limit.
MAX_RETAINED_RUNS = 6

# A subscriber that cannot keep up is disconnected rather than allowed to hold
# events in memory indefinitely.
SUBSCRIBER_QUEUE_LIMIT = 512


class RunNotFound(KeyError):
    pass


class RunConflict(RuntimeError):
    """Raised when a session already has a run in flight."""


@dataclass
class Run:
    """One execution, with everything needed to watch or revisit it."""

    run_id: str
    session_id: str
    contract: AnalysisContract
    trace: RunTrace
    outcomes: dict[str, ToolOutcome] = field(default_factory=dict)
    history: list[RunEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    cancel_requested: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    # Reprojected rasters and vectors written for the map workspace.
    layers: list = field(default_factory=list)

    @property
    def is_running(self) -> bool:
        """Whether work is still outstanding.

        The trace status is checked as well as the thread flag. The orchestrator
        settles the status before it emits its closing event, so a client that
        starts a new run the instant it sees that event is not told, wrongly, that
        the previous one is still in flight.
        """
        return not self.finished.is_set() and self.trace.status is RunStatus.RUNNING


class RunManager:
    """Owns the in-flight and recently completed runs for this process."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._lock = threading.Lock()

    # -- lookup -----------------------------------------------------------
    def get(self, run_id: str) -> Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise RunNotFound(f"No such run: {run_id}")
        return run

    def active_for_session(self, session_id: str) -> Run | None:
        with self._lock:
            for run in reversed(self._runs.values()):
                if run.session_id == session_id and run.is_running:
                    return run
        return None

    def latest_for_session(self, session_id: str) -> Run | None:
        with self._lock:
            for run in reversed(self._runs.values()):
                if run.session_id == session_id:
                    return run
        return None

    def outcomes_for(self, run_id: str) -> dict[str, ToolOutcome]:
        """The in-memory tool outputs, for stages that need the arrays."""
        return self.get(run_id).outcomes

    # -- persistence ------------------------------------------------------
    def runs_dir(self, session_id: str) -> Path:
        directory = self.store.session_dir(session_id) / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def trace_path(self, session_id: str, run_id: str) -> Path:
        return self.runs_dir(session_id) / f"{run_id}.json"

    def layers_dir(self, session_id: str, run_id: str) -> Path:
        directory = self.runs_dir(session_id) / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def layers_path(self, session_id: str, run_id: str) -> Path:
        return self.layers_dir(session_id, run_id) / "layers.json"

    def _write_layers(self, run: Run, context: ToolContext) -> None:
        """Render the run's masks and imagery as map layers.

        Done here rather than inside each tool: a tool produces a mask and knows
        nothing about maps, and anything that produces a mask becomes mappable
        without being changed.
        """
        from app.core.overlays import persist_base_layers, persist_outcome_layers

        directory = self.layers_dir(run.session_id, run.run_id)
        layers = []
        try:
            layers.extend(persist_base_layers(directory, context.session, self.store))
            for tool, outcome in run.outcomes.items():
                if outcome.ok:
                    layers.extend(persist_outcome_layers(directory, tool, outcome))
        except Exception as exc:  # noqa: BLE001 - the map is not load-bearing
            logger.warning("layer rendering failed for run %s: %s", run.run_id, exc)

        run.layers = layers
        payload = [layer.__dict__ for layer in layers]
        try:
            self.layers_path(run.session_id, run.run_id).write_text(
                json.dumps(payload, default=str, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # noqa: BLE001
            logger.warning("could not write layers for %s: %s", run.run_id, exc)

    def packet_dir(self, session_id: str, run_id: str) -> Path:
        """Where a run's exports live.

        The same directory as the overlays, because the packet includes them and
        two directories would mean either copying the files or a zip that points
        outside itself.
        """
        return self.layers_dir(session_id, run_id)

    def _write_packet(self, run: Run, context: ToolContext):
        """Assemble the exportable record of this run.

        Returns the packet, or None when it could not be written. The orchestrator
        reports either outcome in the trace; nothing here can fail the run.
        """
        from app.core.packet import build_packet

        try:
            return build_packet(
                run.trace,
                run.outcomes,
                self.packet_dir(run.session_id, run.run_id),
                session=context.session,
                readiness=context.readiness,
                layers=run.layers,
            )
        except Exception as exc:  # noqa: BLE001 - an export is not load-bearing
            logger.warning("packet failed for run %s: %s", run.run_id, exc)
            raise

    def load_layers(self, session_id: str, run_id: str) -> list[dict]:
        path = self.layers_path(session_id, run_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _persist(self, run: Run) -> None:
        try:
            self.trace_path(run.session_id, run.run_id).write_text(
                run.trace.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:  # noqa: BLE001 - losing the copy is not fatal
            logger.warning("could not write trace for run %s: %s", run.run_id, exc)

    def load_trace(self, session_id: str, run_id: str) -> RunTrace | None:
        """A trace from disk, so a record outlives the process that made it."""
        path = self.trace_path(session_id, run_id)
        if not path.exists():
            return None
        try:
            return RunTrace.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("unreadable trace %s: %s", path.name, exc)
            return None

    # -- starting a run ---------------------------------------------------
    async def start(
        self,
        contract: AnalysisContract,
        context: ToolContext,
        registry: ToolRegistry,
    ) -> Run:
        """Begin executing a contract and return immediately.

        The returned run already carries the full stage list with everything
        pending, so the client can draw the pipeline before the first event
        arrives.
        """
        session_id = context.session.session_id
        existing = self.active_for_session(session_id)
        if existing is not None:
            raise RunConflict(
                f"Run {existing.run_id} is still in flight for this session. "
                "Wait for it to finish or cancel it."
            )

        run_id = uuid.uuid4().hex
        run = Run(
            run_id=run_id,
            session_id=session_id,
            contract=contract,
            trace=_pending_trace(run_id, contract),
        )
        with self._lock:
            self._runs[run_id] = run
            self._evict_locked()

        loop = asyncio.get_running_loop()

        def emit(event: RunEvent) -> None:
            """Called from the worker thread for every event."""
            run.history.append(event)
            for queue in list(run.subscribers):
                loop.call_soon_threadsafe(_offer, queue, event)

        def compose(trace, outcomes):
            """Render the layers, then assemble the packet from them.

            Both happen here, inside the run's own closing stage, because the
            packet embeds the overlays and the trace has to be able to name what
            was written. Rendering the layers at the end of the run instead would
            leave the report describing files that did not exist yet.
            """
            run.trace = trace
            run.outcomes = outcomes
            self._write_layers(run, context)
            return self._write_packet(run, context)

        def finalise(trace, outcomes) -> None:
            run.trace = trace
            run.outcomes = outcomes
            # A refused, cancelled or failed run never reaches the composing
            # stage, and the imagery it was given is still worth putting on the
            # map. Already-rendered layers are not rendered again.
            if not run.layers:
                self._write_layers(run, context)
            self._persist(run)

        def work() -> None:
            try:
                trace, outcomes = execute_run(
                    contract,
                    context,
                    registry,
                    run_id=run_id,
                    emit=emit,
                    cancelled=lambda: run.cancel_requested,
                    finalise=finalise,
                    compose=compose,
                )
                run.trace = trace
                run.outcomes = outcomes
            except BaseException as exc:  # noqa: BLE001 - the thread must not die silently
                logger.exception("run %s crashed outside the orchestrator", run_id)
                run.trace.status = RunStatus.FAILED
                run.trace.error = str(exc)
            finally:
                run.finished.set()
                # Written again so the stored trace carries the closing status,
                # including the case where the orchestrator never reached its
                # own finalising step.
                self._persist(run)
                # Release anyone still waiting, including in the case where the
                # orchestrator never reached its closing event.
                for queue in list(run.subscribers):
                    loop.call_soon_threadsafe(_offer, queue, None)

        threading.Thread(
            target=work, name=f"satquery-run-{run_id[:8]}", daemon=True
        ).start()
        logger.info("run %s started for session %s", run_id[:8], session_id)
        return run

    def request_cancel(self, run_id: str) -> Run:
        """Ask a run to stop. It stops between steps, never mid-measurement."""
        run = self.get(run_id)
        run.cancel_requested = True
        logger.info("cancellation requested for run %s", run_id[:8])
        return run

    def _evict_locked(self) -> None:
        while len(self._runs) > MAX_RETAINED_RUNS:
            for run_id, candidate in self._runs.items():
                if candidate.is_running:
                    continue
                self._runs.pop(run_id)
                logger.debug("evicted completed run %s from memory", run_id[:8])
                break
            else:
                return

    # -- watching a run ---------------------------------------------------
    async def subscribe(self, run: Run) -> AsyncIterator[RunEvent]:
        """Every event so far, then every event as it happens.

        The queue is registered before the history is snapshotted, so an event
        emitted during handover is queued rather than lost; the sequence number
        then discards the duplicate.
        """
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_LIMIT
        )
        run.subscribers.append(queue)
        try:
            replayed = list(run.history)
            for event in replayed:
                yield event
                if event.type is RunEventType.RUN_FINISHED:
                    return

            if run.finished.is_set():
                return

            seen = replayed[-1].seq if replayed else 0
            while True:
                event = await queue.get()
                if event is None:
                    return
                if event.seq <= seen:
                    continue
                seen = event.seq
                yield event
                if event.type is RunEventType.RUN_FINISHED:
                    return
        finally:
            with contextlib.suppress(ValueError):
                run.subscribers.remove(queue)


def _offer(queue: asyncio.Queue, event: RunEvent | None) -> None:
    """Hand an event to a subscriber, dropping it rather than blocking the loop."""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:  # pragma: no cover - requires a stalled client
        logger.warning("dropping event for a subscriber that is not keeping up")


def _pending_trace(run_id: str, contract: AnalysisContract) -> RunTrace:
    """The trace as it looks before anything has run.

    Built from the same stage list the orchestrator walks, so the pipeline the
    client draws up front is the pipeline that will actually execute.
    """
    from app.core.orchestrator import STAGE_SPECS
    from app.models.trace import TraceStage

    return RunTrace(
        run_id=run_id,
        session_id=contract.session_id,
        contract_hash=contract.contract_hash,
        query=contract.query,
        claim=contract.claim,
        task_type=contract.task_type,
        configuration=contract.configuration,
        stages=[
            TraceStage(id=spec.id, label=spec.label, purpose=spec.purpose)
            for spec in STAGE_SPECS
        ],
    )


_manager: RunManager | None = None


def get_run_manager() -> RunManager:
    global _manager
    if _manager is None:
        from app.core.sessions import get_store

        _manager = RunManager(get_store())
    return _manager


def reset_run_manager_for_tests(store: SessionStore | None = None) -> RunManager:
    """Replace the process-wide manager. Test-only seam."""
    global _manager
    from app.core.sessions import get_store

    _manager = RunManager(store if store is not None else get_store())
    return _manager


__all__ = [
    "MAX_RETAINED_RUNS",
    "Run",
    "RunConflict",
    "RunManager",
    "RunNotFound",
    "get_run_manager",
    "reset_run_manager_for_tests",
]
