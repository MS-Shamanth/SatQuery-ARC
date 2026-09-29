"""The orchestrator.

Executes an approved Analysis Contract stage by stage, writing the trace as it
goes and emitting an event for everything that happens. The whole pipeline is
declared up front, including the stages this build does not implement yet: a
capability that is absent appears in the trace saying so, because a pipeline
diagram with the gaps quietly removed is a lie about what the system does.

This module is deliberately synchronous and side-effect free apart from the
``emit`` callback it is handed. Threading, streaming, and persistence live in
``app.core.runs``, so the execution logic can be tested by calling it and
collecting the events in a list.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.registry import ToolRegistry
from app.models.confounders import (
    VERDICT_LABEL,
    ConfounderReport,
    ConfounderVerdict,
)
from app.models.contract import (
    AnalysisContract,
    ConfounderKind,
    ConfounderPlan,
)
from app.models.disagreement import DisagreementReport
from app.models.packet import EvidencePacket
from app.models.schemas import CheckStatus, ReadinessCheck, ReadinessVerdict
from app.models.trace import (
    RunEvent,
    RunEventType,
    RunStatus,
    RunTrace,
    StageId,
    StageStatus,
    TraceStage,
    TraceStep,
)
from app.models.verdict import VERDICT_LABEL_TEXT, EvidenceLedger, Verdict
from app.tools.base import ToolContext, ToolOutcome

logger = logging.getLogger(__name__)

Emit = Callable[[RunEvent], None]
# Writes the run out as files and returns what it wrote. Supplied by the caller
# because the orchestrator knows what the record contains but not where a record
# belongs on disk.
Compose = Callable[[RunTrace, dict[str, ToolOutcome]], "EvidencePacket | None"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# The pipeline, declared once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    id: StageId
    label: str
    purpose: str


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        StageId.VERIFY_INPUTS,
        "Verify inputs",
        "Confirm the imagery can carry the question before spending effort on it.",
    ),
    StageSpec(
        StageId.ACCEPT_CONTRACT,
        "Accept contract",
        "Record exactly what was approved, so the run can be audited against it.",
    ),
    StageSpec(
        StageId.RUN_SPECIALISTS,
        "Run specialists",
        "Execute each planned tool and collect the measurements it produced.",
    ),
    StageSpec(
        StageId.TEST_CONFOUNDERS,
        "Test confounders",
        "Try to explain the finding away before believing it.",
    ),
    StageSpec(
        StageId.COMPARE_EVIDENCE,
        "Compare evidence",
        "Check whether measurements taken by different means agree.",
    ),
    StageSpec(
        StageId.RESOLVE_VERDICT,
        "Resolve verdict",
        "Commit to supported, refuted, or inconclusive, with a confidence breakdown.",
    ),
    StageSpec(
        StageId.COMPOSE_PACKET,
        "Compose packet",
        "Assemble the reproducible record of how the answer was reached.",
    ),
)

# Shown when a stage could not be reached in this configuration. Worded as what is
# missing rather than as an internal task number, because the user reads it.
NOT_BUILT_NOTES: dict[StageId, str] = {
    StageId.COMPOSE_PACKET: (
        "No exportable packet was assembled for this run, because nothing was "
        "wired up to write one. The trace on this page is the record."
    ),
}

# Which readiness check already measures each confounder. The gate runs before a
# contract is drafted, so several of the alternative explanations the contract
# names already have a measured value attached; reusing it means the trace and the
# readiness panel can never show two different numbers for the same quantity.
CONFOUNDER_EVIDENCE: dict[ConfounderKind, tuple[str, ...]] = {
    ConfounderKind.SEASONALITY: ("pair.temporal",),
    ConfounderKind.MISREGISTRATION: ("pair.coregistration",),
    ConfounderKind.CLOUD_SHADOW: ("cloud.",),
    ConfounderKind.SENSOR_MISMATCH: ("pair.resolution_match", "pair.crs_match"),
    ConfounderKind.RADIOMETRY: (),
    ConfounderKind.SAR_SPECIFIC: (),
    ConfounderKind.ILLUMINATION_TERRAIN: (),
}

CONFOUNDER_ENGINE_NOTE = (
    "These are the alternative explanations this contract committed to testing. "
    "Where the readiness gate already measured one, its value is carried forward "
    "rather than measured again. Scoring each against the finding is the "
    "confounder engine's job and is not part of this build yet."
)


class RunCancelled(RuntimeError):
    """Raised internally when a cancellation was requested between steps."""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class Orchestrator:
    """Walks the stage list for one contract."""

    def __init__(
        self,
        *,
        contract: AnalysisContract,
        context: ToolContext,
        registry: ToolRegistry,
        run_id: str | None = None,
        emit: Emit | None = None,
        cancelled: Callable[[], bool] | None = None,
        finalise: Callable[[RunTrace, dict[str, ToolOutcome]], None] | None = None,
        compose: Compose | None = None,
    ) -> None:
        self.contract = contract
        self.context = context
        self.registry = registry
        self.run_id = run_id or uuid.uuid4().hex
        self._emit_to = emit
        self._cancelled = cancelled or (lambda: False)
        self._finalise = finalise
        self._compose = compose
        self._seq = 0
        self._started = time.perf_counter()

        # Kept so later stages, and Task 8's map overlays, can reach the arrays
        # the tools produced. Only the summaries go into the trace.
        self.outcomes: dict[str, ToolOutcome] = {}

        self.trace = RunTrace(
            run_id=self.run_id,
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

    # -- event plumbing ---------------------------------------------------
    def _emit(self, event_type: RunEventType, **fields: object) -> None:
        if self._emit_to is None:
            return
        self._seq += 1
        self._emit_to(
            RunEvent(
                type=event_type,
                run_id=self.run_id,
                seq=self._seq,
                **fields,  # type: ignore[arg-type]
            )
        )

    def _log(self, message: str) -> None:
        logger.info("run %s | %s", self.run_id[:8], message)
        self._emit(RunEventType.LOG, message=message)

    def _checkpoint(self) -> None:
        if self._cancelled():
            raise RunCancelled()

    # -- stage plumbing ---------------------------------------------------
    def _stage(self, stage_id: StageId) -> TraceStage:
        stage = self.trace.stage(stage_id)
        if stage is None:  # pragma: no cover - the stage list is a constant
            raise KeyError(f"Unknown stage {stage_id}")
        return stage

    @contextmanager
    def _running(self, stage_id: StageId) -> Iterator[TraceStage]:
        """Mark a stage running, then settle its status from its steps."""
        stage = self._stage(stage_id)
        stage.status = StageStatus.RUNNING
        stage.started_at = _now()
        started = time.perf_counter()
        self._emit(RunEventType.STAGE_STARTED, stage=stage, message=stage.label)
        try:
            yield stage
        finally:
            stage.finished_at = _now()
            stage.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            if stage.status is StageStatus.RUNNING:
                stage.status = derive_status(stage.steps)
            self._emit(RunEventType.STAGE_FINISHED, stage=stage, message=stage.label)

    def _step(
        self,
        stage: TraceStage,
        step_id: str,
        label: str,
        *,
        detail: str = "",
        tool: str | None = None,
        version: str | None = None,
        parameters: dict[str, object] | None = None,
    ) -> TraceStep:
        step = TraceStep(
            id=f"{stage.id.value}:{step_id}",
            stage=stage.id,
            label=label,
            detail=detail,
            status=StageStatus.RUNNING,
            tool=tool,
            version=version,
            parameters=dict(parameters or {}),
            started_at=_now(),
        )
        stage.steps.append(step)
        self._emit(RunEventType.STEP_STARTED, step=step, message=label)
        return step

    def _settle(
        self,
        step: TraceStep,
        status: StageStatus,
        *,
        detail: str | None = None,
        reason: str | None = None,
        notes: list[str] | None = None,
        tool_run: object | None = None,
    ) -> TraceStep:
        step.status = status
        if detail is not None:
            step.detail = detail
        step.reason = reason
        if notes:
            step.notes.extend(notes)
        step.finished_at = _now()
        if step.started_at is not None:
            step.duration_ms = round(
                (step.finished_at - step.started_at).total_seconds() * 1000, 2
            )
        self._emit(
            RunEventType.STEP_FINISHED,
            step=step,
            message=step.label,
            tool_run=tool_run,
        )
        return step

    # -- the run ----------------------------------------------------------
    def run(self) -> RunTrace:
        self._emit(
            RunEventType.RUN_STARTED,
            trace=self.trace,
            message=f"Investigating: {self.contract.claim}",
        )
        self._log(
            f"Contract {self.contract.contract_hash[:12]} approved: "
            f"{self.contract.summary_line()}"
        )

        try:
            gate_passed = self._verify_inputs()
            if not gate_passed:
                self.trace.status = RunStatus.REFUSED
                self._mark_remaining_unreached(
                    "The readiness gate refused this input, so nothing downstream ran."
                )
                return self._finish()

            self._accept_contract()
            self._run_specialists()
            self._test_confounders()
            self._compare_evidence()
            self._resolve_verdict()
            self._compose_packet()
            self.trace.status = RunStatus.COMPLETED
        except RunCancelled:
            self.trace.status = RunStatus.CANCELLED
            self._mark_remaining_unreached("Cancelled before this stage ran.")
            self._log("Run cancelled.")
        except Exception as exc:  # noqa: BLE001 - a run failure is reported, not raised
            self.trace.status = RunStatus.FAILED
            self.trace.error = str(exc)
            self._mark_remaining_unreached("The run failed before this stage ran.")
            logger.exception("run %s failed", self.run_id)

        return self._finish()

    def _finish(self) -> RunTrace:
        # Artifacts are written before the closing event, so a client that reacts
        # to run.finished by fetching the map layers always finds them there.
        if self._finalise is not None:
            try:
                self._finalise(self.trace, self.outcomes)
            except Exception:  # noqa: BLE001 - presentation must not fail the run
                logger.exception("finalising run %s failed", self.run_id)

        self.trace.finished_at = _now()
        self.trace.duration_ms = round((time.perf_counter() - self._started) * 1000, 2)
        self._log(self.trace.summary_line())
        self._emit(
            RunEventType.RUN_FINISHED,
            trace=self.trace,
            message=self.trace.status.value,
        )
        return self.trace

    def _mark_remaining_unreached(self, reason: str) -> None:
        for stage in self.trace.stages:
            if stage.status is StageStatus.PENDING:
                stage.status = StageStatus.SKIPPED
                stage.unavailable_note = reason

    # -- stage: verify inputs ---------------------------------------------
    def _verify_inputs(self) -> bool:
        """Report the gate's finding. Returns False when the run must stop.

        The gate is not re-run here. It measured these values when the imagery
        was read, and re-deriving them would let the trace and the readiness
        panel disagree about the same quantity.
        """
        report = self.context.readiness
        with self._running(StageId.VERIFY_INPUTS) as stage:
            if report is None:
                step = self._step(stage, "gate", "Readiness gate")
                self._settle(
                    step,
                    StageStatus.SKIPPED,
                    detail="not evaluated",
                    reason=(
                        "No readiness report was available for this run, so the "
                        "input was not verified."
                    ),
                )
                stage.notes.append(
                    "Running without a readiness report means nothing checked "
                    "whether these images can answer the question."
                )
                return True

            failures = report.by_status(CheckStatus.FAIL)
            warnings = report.by_status(CheckStatus.WARN)
            passes = report.by_status(CheckStatus.PASS)

            step = self._step(
                stage,
                "gate",
                "Readiness gate",
                detail=report.verdict.value.replace("_", " "),
            )
            self._settle(
                step,
                StageStatus.FAILED
                if report.verdict is ReadinessVerdict.REFUSED
                else (StageStatus.PARTIAL if warnings else StageStatus.OK),
                detail=(
                    f"{len(passes)} passed, {len(warnings)} warned, "
                    f"{len(failures)} failed"
                ),
                notes=[
                    f"{check.label}: {check.measured} - {check.message}"
                    for check in (failures + warnings)
                ],
            )

            parts: list[str] = []
            if report.common_epsg is not None:
                parts.append(f"EPSG:{report.common_epsg}")
            if report.overlap_fraction is not None:
                parts.append(f"overlap {report.overlap_fraction:.1%}")
            if report.day_delta is not None:
                parts.append(f"{report.day_delta} days apart")
            if report.month_of_year_delta is not None:
                parts.append(
                    f"{report.month_of_year_delta} month(s) apart in the annual cycle"
                )
            if report.seasonal_risk:
                parts.append("seasonal risk flagged")

            # A single image has no pair geometry, which is an absence rather than
            # a skipped test, so the step only appears when there is something to
            # carry.
            if parts:
                carried = self._step(stage, "carried", "Values carried forward")
                self._settle(
                    carried,
                    StageStatus.OK,
                    detail="; ".join(parts),
                    notes=[f"Gate computed in {report.computed_ms:.0f} ms."],
                )
            else:
                stage.notes.append(
                    f"Gate computed in {report.computed_ms:.0f} ms. A single image "
                    "carries no pair geometry or date separation forward."
                )

            if report.verdict is ReadinessVerdict.REFUSED:
                stage.status = StageStatus.FAILED
                self.trace.refusal_reasons = list(report.refusal_reasons)
                for reason in report.refusal_reasons:
                    self._log(f"Refused: {reason}")
                return False

        return True

    # -- stage: accept contract -------------------------------------------
    def _accept_contract(self) -> None:
        self._checkpoint()
        contract = self.contract
        with self._running(StageId.ACCEPT_CONTRACT) as stage:
            plan_step = self._step(
                stage,
                "authorised",
                "Plan authorised",
                detail=f"{len(contract.tools)} tool(s) in order",
            )
            self._settle(
                plan_step,
                StageStatus.OK if contract.tools else StageStatus.SKIPPED,
                reason=(
                    None
                    if contract.tools
                    else "The validated contract planned no runnable tool."
                ),
                notes=[
                    f"{index + 1}. {plan.tool} v{plan.version}"
                    + (f" with {_render_params(plan.parameters)}" if plan.parameters else "")
                    for index, plan in enumerate(_ordered(contract))
                ],
            )

            audit = self._step(stage, "audit", "Validation audit")
            changes = len(contract.repairs) + len(contract.rejections)
            self._settle(
                audit,
                StageStatus.OK,
                detail=(
                    f"{len(contract.rejections)} rejected, "
                    f"{len(contract.repairs)} adjusted"
                    if changes
                    else "the draft validated without changes"
                ),
                notes=[
                    f"Rejected '{item.value}' in {item.field}: {item.reason}"
                    for item in contract.rejections
                ]
                + [
                    f"Adjusted {item.field}: {item.detail}"
                    for item in contract.repairs
                ],
            )

    # -- stage: run specialists -------------------------------------------
    def _run_specialists(self) -> None:
        self._checkpoint()
        with self._running(StageId.RUN_SPECIALISTS) as stage:
            plans = _ordered(self.contract)
            if not plans:
                stage.notes.append(
                    "The contract planned no tool that is registered and usable "
                    "on this input."
                )
                return

            for plan in plans:
                self._checkpoint()
                step = self._step(
                    stage,
                    plan.tool,
                    plan.tool,
                    detail=plan.rationale,
                    tool=plan.tool,
                    version=plan.version,
                    parameters=plan.parameters,
                )

                if plan.tool not in self.registry:
                    self._settle(
                        step,
                        StageStatus.FAILED,
                        detail="not registered",
                        reason=(
                            f"'{plan.tool}' was registered when this contract was "
                            "drafted but is not registered now, so it was not run."
                        ),
                    )
                    continue

                tool = self.registry.get(plan.tool)
                self._log(
                    f"{plan.tool} v{plan.version}"
                    + (f" {_render_params(plan.parameters)}" if plan.parameters else "")
                )

                outcome = tool.run(
                    ToolContext(
                        session=self.context.session,
                        store=self.context.store,
                        readiness=self.context.readiness,
                        parameters=dict(plan.parameters),
                        target_classes=list(self.contract.target_classes),
                        upstream=self.outcomes,
                        contract=self.contract,
                    )
                )
                self.outcomes[plan.tool] = outcome
                record = outcome.to_record()
                self.trace.tool_runs.append(record)

                step.implementation = outcome.implementation
                step.measurement_keys = [m.key for m in outcome.measurements]
                step.mask_keys = [layer.key for layer in outcome.masks]
                step.applies_to = sorted(
                    {role for m in outcome.measurements for role in m.applies_to},
                    key=lambda role: role.value,
                )

                if outcome.ok:
                    self.trace.measurements.extend(outcome.measurements)
                    self.trace.masks.extend(record.masks)
                    self._settle(
                        step,
                        StageStatus.OK,
                        detail=(
                            f"{len(outcome.measurements)} measurement(s), "
                            f"{len(outcome.masks)} mask(s)"
                        ),
                        notes=list(outcome.notes),
                        tool_run=record,
                    )
                    self._log(
                        f"{plan.tool} produced {len(outcome.measurements)} "
                        f"measurement(s) in {outcome.duration_ms:.0f} ms"
                    )
                else:
                    self._settle(
                        step,
                        StageStatus.SKIPPED,
                        detail="did not run",
                        reason=outcome.skipped_reason,
                        notes=list(outcome.notes),
                        tool_run=record,
                    )
                    self._log(f"{plan.tool} skipped: {outcome.skipped_reason}")

    # -- stage: test confounders ------------------------------------------
    def _test_confounders(self) -> None:
        """Report what the confounder engine found, per alternative explanation.

        Stage and step status describe whether the test *ran*, not what it
        concluded. A test that ran and found seasonality to be the likely
        explanation did its job perfectly; collapsing that into a failed step
        would confuse an inconvenient answer with a broken one. The verdict lives
        in the detail and in the confounder report.
        """
        self._checkpoint()
        readiness = self.context.readiness
        report = self._confounder_report()

        with self._running(StageId.TEST_CONFOUNDERS) as stage:
            if report is None:
                stage.notes.append(CONFOUNDER_ENGINE_NOTE)
            else:
                self.trace.confounders = list(report.tests)
            if not self.contract.confounders:
                stage.notes.append(
                    "This contract named no alternative explanation, which should "
                    "not happen: every contract is built to challenge something."
                )
                return

            tested_kinds: set[ConfounderKind] = set()

            for plan in self.contract.confounders:
                self._checkpoint()
                step = self._step(
                    stage, plan.kind.value, plan.label, detail=plan.question
                )

                test = report.by_kind(plan.kind) if report is not None else None
                if test is not None:
                    tested_kinds.add(plan.kind)
                    ran = test.verdict is not ConfounderVerdict.NOT_TESTED
                    notes = [test.explanation]
                    if test.method:
                        notes.append(f"Method: {test.method}.")
                    notes.append(f"Threshold: {test.threshold}.")
                    if test.requirement:
                        notes.append(f"What would settle it: {test.requirement}")
                    self._settle(
                        step,
                        StageStatus.OK if ran else StageStatus.SKIPPED,
                        detail=(
                            f"{VERDICT_LABEL[test.verdict]} \u2014 {test.measured}"
                        ),
                        reason=None if ran else test.explanation,
                        notes=notes,
                    )
                    continue

                # No dedicated test. The readiness gate may still have measured the
                # quantity, which is weaker than a test but better than silence.
                check = _evidence_for(plan, readiness.checks if readiness else [])
                if check is None:
                    self._settle(
                        step,
                        StageStatus.SKIPPED,
                        detail=plan.question,
                        reason=(
                            "Nothing in this build tests this yet, so it stays on "
                            "the table rather than being assumed away."
                        ),
                        notes=[plan.reason] if plan.reason else None,
                    )
                    continue

                self._settle(
                    step,
                    StageStatus.OK,
                    detail=f"measured but not tested \u2014 {check.measured}",
                    notes=[
                        f"{check.message}"
                        + (f" Method: {check.method}." if check.method else ""),
                        "The readiness gate measured this. Scoring it against the "
                        "finding is a separate test, which was not run here.",
                    ],
                )

            # Anything the engine tested that the contract did not name is still
            # worth reporting: it is evidence either way.
            if report is not None:
                for test in report.tests:
                    if test.kind in tested_kinds:
                        continue
                    self._checkpoint()
                    step = self._step(
                        stage,
                        f"extra_{test.kind.value}",
                        test.label,
                        detail=test.question,
                    )
                    self._settle(
                        step,
                        StageStatus.OK
                        if test.verdict is not ConfounderVerdict.NOT_TESTED
                        else StageStatus.SKIPPED,
                        detail=f"{VERDICT_LABEL[test.verdict]} \u2014 {test.measured}",
                        notes=[
                            test.explanation,
                            "Tested although the contract did not name it.",
                        ],
                    )

                if report.likely:
                    stage.notes.append(
                        "A likely alternative explanation was found: "
                        + "; ".join(test.label for test in report.likely)
                        + ". The finding cannot be read as real change until it is "
                        "addressed."
                    )
                for requirement in report.requirements:
                    stage.notes.append(f"Would settle an open question: {requirement}")

                self._offer_remedies(stage, report)

    def _offer_remedies(self, stage: TraceStage, report: ConfounderReport) -> None:
        """Look for imagery that would settle whatever is still standing.

        A refusal that names the acquisition it needs is useful. One that can hand
        you that acquisition is actionable, which is the difference between the
        system explaining why it stopped and the user being able to continue.
        """
        try:
            from app.config import get_settings
            from app.core.remedy import offers_for
            from app.core.samples import load_manifest

            manifest = load_manifest(get_settings().samples_dir)
            remedies = offers_for(
                report, manifest, current_sample=self.context.session.sample_key
            )
        except Exception:  # noqa: BLE001 - an offer is a convenience, never load-bearing
            logger.exception("could not resolve remedies")
            return

        if not remedies.offers and not remedies.unmet:
            return

        self.trace.remedies = remedies
        step = self._step(
            stage,
            "remedies",
            "Imagery that would settle this",
            detail=f"{len(remedies.offers)} scene(s) available",
        )
        self._settle(
            step,
            StageStatus.OK if remedies.offers else StageStatus.SKIPPED,
            reason=(
                None
                if remedies.offers
                else "Nothing in the sample library removes what is still open."
            ),
            notes=[
                f"{offer.title} ({offer.place}) settles {offer.confounder_label.lower()}"
                f": {offer.why}"
                for offer in remedies.offers
            ]
            or [f"Still open, with nothing on hand: {item}" for item in remedies.unmet],
        )

    def _confounder_report(self) -> ConfounderReport | None:
        outcome = self.outcomes.get("confounder-engine")
        if outcome is None or not outcome.ok:
            return None
        report = outcome.artifacts.get("confounders.report")
        return report if isinstance(report, ConfounderReport) else None

    def _disagreement(self) -> DisagreementReport | None:
        outcome = self.outcomes.get("evidence-disagreement-engine")
        if outcome is None or not outcome.ok:
            return None
        report = outcome.artifacts.get("disagreement.report")
        return report if isinstance(report, DisagreementReport) else None

    # -- stage: compare evidence ------------------------------------------
    def _compare_evidence(self) -> None:
        """Report the ledger: what was admitted, what was excluded, what agrees."""
        self._checkpoint()
        ledger = self._ledger()
        disagreement = self._disagreement()
        with self._running(StageId.COMPARE_EVIDENCE) as stage:
            # Reported before the ledger on purpose. Where the methods conflict is
            # the more informative finding, and a reader who sees the conclusion
            # first has already stopped asking how solid it is.
            if disagreement is not None:
                self.trace.disagreement = disagreement
                conflict = self._step(
                    stage, "disagreement", "Where the methods disagree"
                )
                self._settle(
                    conflict,
                    StageStatus.OK,
                    detail=disagreement.summary_line(),
                    notes=[
                        f"Compared: {', '.join(disagreement.methods)}",
                        *(
                            [
                                f"Region {region.region_id}: "
                                f"{region.area_km2:.4f} km2, {region.reason}"
                                for region in disagreement.regions[:5]
                            ]
                        ),
                        *disagreement.notes,
                    ],
                )

            tools = sorted({run.tool for run in self.trace.tool_runs if run.ok})
            gathered = self._step(stage, "gathered", "Measurements gathered")
            self._settle(
                gathered,
                StageStatus.OK if self.trace.measurements else StageStatus.SKIPPED,
                detail=(
                    f"{len(self.trace.measurements)} measurement(s) from "
                    f"{len(tools)} tool(s)"
                ),
                reason=(
                    None
                    if self.trace.measurements
                    else "No tool produced a measurement to compare."
                ),
                notes=[f"Contributing tools: {', '.join(tools)}"] if tools else None,
            )

            if ledger is None:
                cross = self._step(stage, "cross_check", "Cross-check")
                self._settle(
                    cross,
                    StageStatus.SKIPPED,
                    detail="not scored",
                    reason=(
                        "Nothing ruled on a claim in this run, so no ledger was "
                        "built and no cross-check was made."
                    ),
                )
                return

            self.trace.ledger = ledger
            admitted = self._step(stage, "admitted", "Evidence admitted")
            self._settle(
                admitted,
                StageStatus.OK if ledger.items else StageStatus.SKIPPED,
                detail=(
                    f"{len(ledger.items)} admitted, {len(ledger.excluded)} set aside"
                ),
                reason=(
                    None
                    if ledger.items
                    else "No measurement bore on the claim under test."
                ),
                notes=[f"{item.label}: {item.statement}" for item in ledger.items],
            )

            if not ledger.consistency:
                cross = self._step(stage, "cross_check", "Cross-check")
                self._settle(
                    cross,
                    StageStatus.SKIPPED,
                    detail="nothing to compare",
                    reason=(
                        "No quantity was measured twice by different routes, so the "
                        "finding rests on a single method."
                    ),
                )
                return

            for check in ledger.consistency:
                self._checkpoint()
                step = self._step(
                    stage,
                    f"cross_{check.first_key}",
                    check.label,
                    detail=check.method,
                )
                self._settle(
                    step,
                    StageStatus.OK,
                    detail=(
                        f"{check.first_value:.3f} against {check.second_value:.3f} "
                        f"{check.unit}, {check.relative_difference:.0%} apart"
                        + ("" if check.agrees else " \u2014 beyond tolerance")
                    ),
                    notes=[check.explanation],
                )

    # -- stage: resolve verdict -------------------------------------------
    def _resolve_verdict(self) -> None:
        """State the conclusion, or say why none was reached."""
        self._checkpoint()
        verdict = self._verdict()
        with self._running(StageId.RESOLVE_VERDICT) as stage:
            if verdict is None:
                stage.status = StageStatus.NOT_BUILT
                stage.unavailable_note = (
                    "No verdict was issued: nothing in this run weighed the "
                    "measurements against a claim. The system does not state a "
                    "conclusion it cannot defend with a confidence breakdown."
                )
                return

            self.trace.verdict = verdict

            decision = self._step(
                stage,
                "decision",
                VERDICT_LABEL_TEXT[verdict.label],
                detail=verdict.claim,
            )
            self._settle(
                decision,
                StageStatus.OK,
                detail=(
                    f"{VERDICT_LABEL_TEXT[verdict.label]} at "
                    f"{verdict.confidence:.0%} confidence"
                ),
                notes=[verdict.reasoning],
            )

            breakdown = self._step(stage, "confidence", "Confidence breakdown")
            self._settle(
                breakdown,
                StageStatus.OK,
                detail=" ".join(
                    f"{component.name} {component.contribution:+.2f}"
                    for component in verdict.confidence_components
                ),
                notes=[
                    f"{component.label}: measured {component.measured:.3f}, "
                    f"contributes {component.contribution:+.3f} of "
                    f"{component.weight:.2f}. {component.rationale}"
                    for component in verdict.confidence_components
                ],
            )

            audit = self._step(stage, "numeric_audit", "Numeric audit")
            if verdict.narrative_audit is None:
                self._settle(
                    audit,
                    StageStatus.SKIPPED,
                    detail="no phrased explanation to audit",
                    reason=(
                        "Only the measured reasoning was produced, and it is built "
                        "from the ledger by construction."
                    ),
                )
            else:
                report = verdict.narrative_audit
                self._settle(
                    audit,
                    StageStatus.OK if report.passed else StageStatus.PARTIAL,
                    detail=(
                        f"{report.traced} of {report.checked} figure(s) traced to a "
                        "measurement"
                    ),
                    reason=None if report.passed else report.note,
                    notes=[report.note] if report.note else None,
                )

            if verdict.what_would_change_it:
                for lever in verdict.what_would_change_it:
                    stage.notes.append(f"Would change the answer: {lever}")

    def _ledger(self) -> EvidenceLedger | None:
        outcome = self.outcomes.get("verdict-engine")
        if outcome is None or not outcome.ok:
            return None
        ledger = outcome.artifacts.get("verdict.ledger")
        return ledger if isinstance(ledger, EvidenceLedger) else None

    def _verdict(self) -> Verdict | None:
        outcome = self.outcomes.get("verdict-engine")
        if outcome is None or not outcome.ok:
            return None
        verdict = outcome.artifacts.get("verdict.result")
        return verdict if isinstance(verdict, Verdict) else None

    # -- stage: compose packet --------------------------------------------
    def _compose_packet(self) -> None:
        """Write the run out as files, and report what was written.

        The packet is assembled here rather than after the run so that the trace
        records it: a set of exported files that the record does not mention is a
        set of files nobody can vouch for. Writing it cannot fail the run, which is
        why the hook's exceptions are caught and reported as a failed step rather
        than raised.
        """
        self._checkpoint()
        with self._running(StageId.COMPOSE_PACKET) as stage:
            if self._compose is None:
                stage.status = StageStatus.NOT_BUILT
                stage.unavailable_note = NOT_BUILT_NOTES[StageId.COMPOSE_PACKET]
                return

            step = self._step(stage, "assemble", "Assemble the packet")
            try:
                packet = self._compose(self.trace, self.outcomes)
            except Exception as exc:  # noqa: BLE001 - an export never fails a run
                logger.exception("composing the packet for run %s failed", self.run_id)
                self._settle(
                    step,
                    StageStatus.FAILED,
                    detail="not written",
                    reason=(
                        "The packet could not be written, so this run has no export. "
                        f"The measurements above are unaffected. ({exc})"
                    ),
                )
                return

            if packet is None:
                self._settle(
                    step,
                    StageStatus.SKIPPED,
                    detail="nothing to export",
                    reason=(
                        "This run produced nothing that could be assembled into a "
                        "packet."
                    ),
                )
                return

            self.trace.packet = packet
            self._settle(
                step,
                StageStatus.OK,
                detail=packet.summary_line(),
                notes=[
                    f"{item.filename} \u2014 {item.description}"
                    for item in packet.files
                ],
            )

            audit = self._step(stage, "attribution", "Figure attribution")
            self._settle(
                audit,
                StageStatus.OK if packet.audit.passed else StageStatus.PARTIAL,
                detail=(
                    f"{packet.audit.traced} of {packet.audit.checked} figure(s) in "
                    "the report attributed to a source"
                ),
                reason=None if packet.audit.passed else packet.audit.note,
                notes=[packet.audit.note],
            )

            for omission in packet.omissions:
                stage.notes.append(f"Not in the packet: {omission}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def derive_status(steps: list[TraceStep]) -> StageStatus:
    """A stage's status is the honest summary of its steps.

    ``PARTIAL`` exists so that "two tools ran and one declined" is not rounded up
    to success or down to failure.
    """
    if not steps:
        return StageStatus.SKIPPED
    statuses = {step.status for step in steps}
    if statuses == {StageStatus.OK}:
        return StageStatus.OK
    if StageStatus.FAILED in statuses and not (
        statuses & {StageStatus.OK, StageStatus.PARTIAL}
    ):
        return StageStatus.FAILED
    if statuses <= {StageStatus.SKIPPED, StageStatus.NOT_BUILT}:
        return StageStatus.SKIPPED
    return StageStatus.PARTIAL


def _ordered(contract: AnalysisContract) -> list:
    return sorted(contract.tools, key=lambda plan: plan.order)


def _render_params(parameters: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(parameters.items()))


def _evidence_for(
    plan: ConfounderPlan, checks: list[ReadinessCheck]
) -> ReadinessCheck | None:
    """The readiness check that already measured this confounder, if any.

    Per-image checks carry the role in their id, so a prefix entry matches the
    worst-status one rather than an arbitrary one.
    """
    patterns = CONFOUNDER_EVIDENCE.get(plan.kind, ())
    if not patterns:
        return None

    severity = {
        CheckStatus.FAIL: 0,
        CheckStatus.WARN: 1,
        CheckStatus.PASS: 2,
        CheckStatus.NOT_APPLICABLE: 3,
    }
    matches = [
        check
        for check in checks
        if any(
            check.id == pattern
            if not pattern.endswith(".")
            else check.id.startswith(pattern)
            for pattern in patterns
        )
        and check.status is not CheckStatus.NOT_APPLICABLE
    ]
    if not matches:
        return None
    return min(matches, key=lambda check: severity.get(check.status, 4))


def execute_run(
    contract: AnalysisContract,
    context: ToolContext,
    registry: ToolRegistry,
    *,
    run_id: str | None = None,
    emit: Emit | None = None,
    cancelled: Callable[[], bool] | None = None,
    finalise: Callable[[RunTrace, dict[str, ToolOutcome]], None] | None = None,
    compose: Compose | None = None,
) -> tuple[RunTrace, dict[str, ToolOutcome]]:
    """Execute a contract, returning the trace and the in-memory tool outcomes.

    The outcomes carry the raster arrays, which the trace deliberately does not,
    so the map workspace can build overlays from the same pixels the
    measurements came from.
    """
    orchestrator = Orchestrator(
        contract=contract,
        context=context,
        registry=registry,
        run_id=run_id,
        emit=emit,
        cancelled=cancelled,
        finalise=finalise,
        compose=compose,
    )
    trace = orchestrator.run()
    return trace, orchestrator.outcomes


__all__ = [
    "CONFOUNDER_EVIDENCE",
    "NOT_BUILT_NOTES",
    "Orchestrator",
    "RunCancelled",
    "STAGE_SPECS",
    "derive_status",
    "execute_run",
]
