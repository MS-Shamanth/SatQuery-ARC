"""The execution trace.

A run is not a black box that returns an answer. It is a sequence of named
stages, each containing named steps, each recording which tool ran with which
parameters and what it produced. The trace is written as the run proceeds and
streamed to the client, so what the user watches happening is the same record
that ends up in the evidence packet.

Three rules shape these models:

* A stage that cannot run says so and says why. There is no silent skip.
* A step names its tool and version. A number in the UI can always be traced
  back through ``measurement_keys`` to the step that produced it.
* Capabilities that are not part of this build are declared as stages with
  ``NOT_BUILT`` status rather than omitted, so the pipeline the user sees is the
  whole pipeline and the gaps are visible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.core.remedy import RemedySet
from app.models.confounders import ConfounderTest
from app.models.disagreement import DisagreementReport
from app.models.packet import EvidencePacket
from app.models.contract import ContractTaskType
from app.models.verdict import EvidenceLedger, Verdict
from app.models.schemas import (
    ImageRole,
    InputConfiguration,
    MaskSummary,
    Measurement,
    ToolImplementation,
    ToolRun,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StageId(str, Enum):
    """The stages of an investigation, in the order they run.

    This is the pipeline the system claims to implement, written down once. The
    orchestrator walks it in order and the UI draws it as a graph.
    """

    VERIFY_INPUTS = "verify_inputs"
    ACCEPT_CONTRACT = "accept_contract"
    RUN_SPECIALISTS = "run_specialists"
    TEST_CONFOUNDERS = "test_confounders"
    COMPARE_EVIDENCE = "compare_evidence"
    RESOLVE_VERDICT = "resolve_verdict"
    COMPOSE_PACKET = "compose_packet"


class StageStatus(str, Enum):
    """Where a stage or step stands.

    ``PARTIAL`` matters: it is the state of a stage where some tools ran and
    others declined, which is a different thing from either success or failure
    and should not be rounded to one of them.
    """

    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"
    NOT_BUILT = "not_built"


TERMINAL_STATUSES = frozenset(
    {
        StageStatus.OK,
        StageStatus.PARTIAL,
        StageStatus.SKIPPED,
        StageStatus.FAILED,
        StageStatus.NOT_BUILT,
    }
)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TraceStep(BaseModel):
    """One unit of work inside a stage."""

    id: str = Field(description="Stable within a run, so the UI can update in place")
    stage: StageId
    label: str
    detail: str = ""
    status: StageStatus = StageStatus.PENDING

    tool: str | None = None
    version: str | None = None
    implementation: ToolImplementation | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    applies_to: list[ImageRole] = Field(default_factory=list)

    # What this step put into the record. The UI links a displayed number back to
    # the step through these keys.
    measurement_keys: list[str] = Field(default_factory=list)
    mask_keys: list[str] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)
    # Present when the step could not do its work, in the user's language.
    reason: str | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0


class TraceStage(BaseModel):
    id: StageId
    label: str
    purpose: str = Field(description="One line on why this stage exists")
    status: StageStatus = StageStatus.PENDING
    steps: list[TraceStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Set when status is NOT_BUILT: names the capability that is absent, in the
    # user's language rather than as an internal task number.
    unavailable_note: str = ""

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0

    @property
    def ran(self) -> bool:
        return self.status in {StageStatus.OK, StageStatus.PARTIAL}


class RunTrace(BaseModel):
    """The complete record of one execution."""

    run_id: str
    session_id: str
    contract_hash: str
    query: str
    claim: str
    task_type: ContractTaskType
    configuration: InputConfiguration
    status: RunStatus = RunStatus.RUNNING

    stages: list[TraceStage] = Field(default_factory=list)

    # Flattened results, gathered as the run proceeds. The evidence ledger and
    # the packet read from these rather than re-deriving them.
    tool_runs: list[ToolRun] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    masks: list[MaskSummary] = Field(default_factory=list)
    # The alternative explanations, tested. Lifted out of the tool's own output
    # because the verdict depends on them and so does the reader.
    confounders: list[ConfounderTest] = Field(default_factory=list)
    # The conclusion, and the ledger it rests on. Absent when nothing ruled.
    verdict: Verdict | None = None
    ledger: EvidenceLedger | None = None
    # Imagery in the library that would settle what is still open, so a refusal
    # can be acted on rather than only read.
    remedies: RemedySet | None = None
    # Where the methods conflict, which is the most informative thing a run can
    # show and so is carried on the trace rather than buried in a tool's output.
    disagreement: DisagreementReport | None = None
    # The run written out as files, once the packet has been assembled. Recorded on
    # the trace so the record names its own exports.
    packet: EvidencePacket | None = None

    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    error: str | None = None
    # Set when the readiness gate stopped the run, which is a refusal and not a
    # failure: the system worked correctly and declined to answer.
    refusal_reasons: list[str] = Field(default_factory=list)

    def stage(self, stage_id: StageId) -> TraceStage | None:
        for item in self.stages:
            if item.id is stage_id:
                return item
        return None

    def measurement(self, key: str) -> Measurement | None:
        for item in self.measurements:
            if item.key == key:
                return item
        return None

    @property
    def tools_run(self) -> int:
        return sum(1 for run in self.tool_runs if run.ok)

    @property
    def tools_skipped(self) -> int:
        return sum(1 for run in self.tool_runs if not run.ok)

    def summary_line(self) -> str:
        return (
            f"{self.tools_run} tool(s) ran, {self.tools_skipped} skipped, "
            f"{len(self.measurements)} measurement(s) recorded"
        )


# ---------------------------------------------------------------------------
# Streamed events
# ---------------------------------------------------------------------------


class RunEventType(str, Enum):
    RUN_STARTED = "run.started"
    STAGE_STARTED = "stage.started"
    STAGE_FINISHED = "stage.finished"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    LOG = "log"
    RUN_FINISHED = "run.finished"


class RunEvent(BaseModel):
    """One thing that happened, as it happened.

    Stages and steps are sent whole rather than as deltas. The payloads are small
    and it makes the client a replace-by-id reducer, which cannot drift out of
    step with the server even if an event is missed on reconnect.
    """

    type: RunEventType
    run_id: str
    seq: int = 0
    at: datetime = Field(default_factory=_now)
    message: str = ""

    stage: TraceStage | None = None
    step: TraceStep | None = None
    # Attached to step.finished for a tool step. Carries the measurements with
    # their formulas, so the client never has to ask for them separately.
    tool_run: ToolRun | None = None
    # Carried on run.started and run.finished so a client that joins late or
    # misses an event still ends up holding the authoritative record.
    trace: RunTrace | None = None


class MapLayer(BaseModel):
    """A layer the map can draw, already reprojected to EPSG:4326.

    ``bounds_wgs84`` is what places the image, and it comes from a real
    reprojection of the source grid rather than from reading the corner
    coordinates of a UTM raster as if they were degrees. A highlighted region
    therefore sits on the pixels whose area was measured.
    """

    key: str
    label: str
    description: str = ""
    kind: str = Field(description="base or mask")
    png_url: str
    geojson_url: str | None = None
    bounds_wgs84: list[float]
    colour: str
    area_km2: float | None = None
    pixel_count: int | None = None
    applies_to: list[ImageRole] = Field(default_factory=list)


class RunRequest(BaseModel):
    """Ask to execute a contract that was already drafted and shown.

    Only the hash is accepted. The contract itself is fetched server-side, so
    what executes is necessarily the plan the user approved and not a version of
    it edited in transit.
    """

    contract_hash: str = Field(min_length=8, max_length=64, pattern="^[0-9a-f]+$")


__all__ = [
    "MapLayer",
    "RunEvent",
    "RunEventType",
    "RunRequest",
    "RunStatus",
    "RunTrace",
    "StageId",
    "StageStatus",
    "TERMINAL_STATUSES",
    "TraceStage",
    "TraceStep",
]
