"""Orchestrator tests.

The orchestrator is where the project's central promise is either kept or broken:
nothing appears in the trace that a tool did not produce, and nothing is quietly
omitted. These tests are written against that, not against the happy path alone.
"""

from __future__ import annotations

import pytest

from app.core.orchestrator import (
    STAGE_SPECS,
    Orchestrator,
    derive_status,
    execute_run,
)
from app.core.readiness import evaluate_readiness
from app.models.contract import (
    AnalysisContract,
    ChangeDirection,
    ConfounderKind,
    ConfounderPlan,
    ContractSource,
    ContractTaskType,
    ToolPlan,
)
from app.models.schemas import (
    CheckStatus,
    ImageRole,
    InputConfiguration,
    ReadinessVerdict,
)
from app.models.trace import RunEventType, RunStatus, StageId, StageStatus, TraceStep
from tests.raster_fixtures import make_labels, make_optical_scene, make_sar_scene


@pytest.fixture
def scene(tmp_path):
    return make_optical_scene(tmp_path / "single.tif")


@pytest.fixture
def context(make_context, scene):
    return make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)


def build_contract(
    context,
    *,
    tools: list[ToolPlan] | None = None,
    confounders: list[ConfounderPlan] | None = None,
    task_type: ContractTaskType = ContractTaskType.REGION_GROUNDING,
    **overrides,
) -> AnalysisContract:
    payload: dict = dict(
        session_id=context.session.session_id,
        query="Highlight the water body",
        claim="The water body is visible and can be highlighted.",
        task_type=task_type,
        configuration=context.configuration,
        acts_on=context.roles,
        target_classes=["water"],
        change_direction=ChangeDirection.UNSPECIFIED,
        confounders=confounders
        if confounders is not None
        else [
            ConfounderPlan(
                kind=ConfounderKind.CLOUD_SHADOW,
                label="Cloud and shadow",
                question="Could cloud be hiding the surface?",
            )
        ],
        tools=tools
        if tools is not None
        else [
            ToolPlan(
                tool="spectral-index-engine",
                version="1.0.0",
                rationale="compute NDWI",
                parameters={"indices": ["NDWI"], "threshold_method": "fixed",
                            "fixed_threshold": 0.0},
                order=0,
            ),
            ToolPlan(
                tool="gis-measure-engine",
                version="1.0.0",
                rationale="measure the scene",
                order=1,
            ),
        ],
        source=ContractSource.OFFLINE_RULE_ROUTER,
        contract_hash="a" * 32,
    )
    payload.update(overrides)
    return AnalysisContract(**payload)


def run(contract, context, tool_registry):
    events = []
    trace, outcomes = execute_run(
        contract, context, tool_registry, emit=events.append
    )
    return trace, outcomes, events


# -- the pipeline is declared in full -------------------------------------


def test_every_declared_stage_appears_in_the_trace(context, tool_registry):
    trace, _, _ = run(build_contract(context), context, tool_registry)
    assert [stage.id for stage in trace.stages] == [spec.id for spec in STAGE_SPECS]


def test_no_stage_is_left_pending_after_a_completed_run(context, tool_registry):
    trace, _, _ = run(build_contract(context), context, tool_registry)
    assert trace.status is RunStatus.COMPLETED
    assert [s.id for s in trace.stages if s.status is StageStatus.PENDING] == []


def test_stages_this_build_does_not_implement_say_so(context, tool_registry):
    """A pipeline diagram that hides its gaps misrepresents the system."""
    trace, _, _ = run(build_contract(context), context, tool_registry)
    verdict = trace.stage(StageId.RESOLVE_VERDICT)
    assert verdict is not None
    assert verdict.status is StageStatus.NOT_BUILT
    assert verdict.unavailable_note
    # The note must be written for a user, not as an internal task number.
    assert "task" not in verdict.unavailable_note.lower()


# -- specialists ----------------------------------------------------------


def test_planned_tools_run_in_contract_order(context, tool_registry):
    contract = build_contract(context)
    trace, _, _ = run(contract, context, tool_registry)
    stage = trace.stage(StageId.RUN_SPECIALISTS)
    assert stage is not None
    assert [step.tool for step in stage.steps] == [
        "spectral-index-engine",
        "gis-measure-engine",
    ]


def test_measurements_reach_the_trace_with_their_provenance(context, tool_registry):
    trace, _, _ = run(build_contract(context), context, tool_registry)
    assert trace.measurements
    for measurement in trace.measurements:
        assert measurement.formula
        assert measurement.source_tool
        assert measurement.source_version


def test_every_measurement_key_a_step_claims_actually_exists(context, tool_registry):
    """The link from a displayed number back to its step must not dangle."""
    trace, _, _ = run(build_contract(context), context, tool_registry)
    present = {m.key for m in trace.measurements}
    for stage in trace.stages:
        for step in stage.steps:
            if step.status is StageStatus.OK:
                assert set(step.measurement_keys) <= present


def test_a_tool_that_cannot_run_is_recorded_with_its_reason(
    tmp_path, make_context, tool_registry
):
    """An unusable input must show as a named skip, not a missing row."""
    sar = make_sar_scene(tmp_path / "s.tif")
    context = make_context({ImageRole.SINGLE: sar.path}, with_readiness=True)
    contract = build_contract(
        context,
        tools=[
            ToolPlan(tool="spectral-index-engine", version="1.0.0", order=0),
        ],
    )
    trace, _, _ = run(contract, context, tool_registry)

    stage = trace.stage(StageId.RUN_SPECIALISTS)
    assert stage is not None
    step = stage.steps[0]
    assert step.status is StageStatus.SKIPPED
    assert step.reason
    assert stage.status is StageStatus.SKIPPED
    assert trace.status is RunStatus.COMPLETED


def test_a_tool_that_left_the_registry_is_reported_not_silently_dropped(
    context, tool_registry
):
    contract = build_contract(
        context,
        tools=[ToolPlan(tool="spectral-index-engine", version="1.0.0", order=0)],
    )
    tool_registry.unregister("spectral-index-engine")
    trace, _, _ = run(contract, context, tool_registry)

    stage = trace.stage(StageId.RUN_SPECIALISTS)
    assert stage is not None
    assert stage.steps[0].status is StageStatus.FAILED
    assert "not registered" in (stage.steps[0].reason or "")


def test_a_contract_with_no_runnable_tool_says_so(context, tool_registry):
    trace, _, _ = run(build_contract(context, tools=[]), context, tool_registry)
    stage = trace.stage(StageId.RUN_SPECIALISTS)
    assert stage is not None
    assert stage.status is StageStatus.SKIPPED
    assert stage.notes


# -- the audit trail ------------------------------------------------------


def test_rejected_tool_names_appear_in_the_execution_trace(context, tool_registry):
    """A hallucinated tool rejected at draft time must remain visible at run time."""
    from app.models.contract import ContractRejection

    contract = build_contract(
        context,
        rejections=[
            ContractRejection(
                field="tools",
                value="changeformer-large",
                reason="not a registered tool",
            )
        ],
    )
    trace, _, _ = run(contract, context, tool_registry)
    stage = trace.stage(StageId.ACCEPT_CONTRACT)
    assert stage is not None
    audit = next(step for step in stage.steps if step.label == "Validation audit")
    assert any("changeformer-large" in note for note in audit.notes)


# -- confounders ----------------------------------------------------------


def test_each_named_confounder_gets_a_step(context, tool_registry):
    contract = build_contract(
        context,
        confounders=[
            ConfounderPlan(
                kind=ConfounderKind.CLOUD_SHADOW, label="Cloud and shadow", question="?"
            ),
            ConfounderPlan(
                kind=ConfounderKind.RADIOMETRY, label="Radiometry", question="?"
            ),
        ],
    )
    trace, _, _ = run(contract, context, tool_registry)
    stage = trace.stage(StageId.TEST_CONFOUNDERS)
    assert stage is not None
    assert {step.label for step in stage.steps} == {"Cloud and shadow", "Radiometry"}


def test_a_confounder_the_gate_measured_carries_that_value_forward(
    context, tool_registry
):
    """The trace must not re-derive a number the readiness panel already shows."""
    contract = build_contract(
        context,
        confounders=[
            ConfounderPlan(
                kind=ConfounderKind.CLOUD_SHADOW, label="Cloud and shadow", question="?"
            )
        ],
    )
    trace, _, _ = run(contract, context, tool_registry)
    stage = trace.stage(StageId.TEST_CONFOUNDERS)
    assert stage is not None
    step = stage.steps[0]
    assert step.status is StageStatus.OK
    cloud = next(c for c in context.readiness.checks if c.id.startswith("cloud."))
    assert cloud.measured in step.detail


def test_a_confounder_nothing_measures_yet_is_named_but_marked_untested(
    context, tool_registry
):
    contract = build_contract(
        context,
        confounders=[
            ConfounderPlan(
                kind=ConfounderKind.SAR_SPECIFIC, label="Radar geometry", question="?"
            )
        ],
    )
    trace, _, _ = run(contract, context, tool_registry)
    stage = trace.stage(StageId.TEST_CONFOUNDERS)
    assert stage is not None
    assert stage.steps[0].status is StageStatus.SKIPPED
    assert stage.steps[0].reason


# -- the gate stops the run ----------------------------------------------


def test_a_refused_input_stops_the_run_before_any_tool_runs(
    tmp_path, make_context, tool_registry
):
    """The gate is not advisory: a refusal at execution time halts execution."""
    scene = make_optical_scene(
        tmp_path / "cloudy.tif",
        labels=make_labels(cloud=(0, 0, 210, 256)),
        with_scl=True,
    )
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)
    assert context.readiness.verdict is ReadinessVerdict.REFUSED

    trace, _, _ = run(build_contract(context), context, tool_registry)

    assert trace.status is RunStatus.REFUSED
    assert trace.refusal_reasons
    assert trace.tool_runs == []
    specialists = trace.stage(StageId.RUN_SPECIALISTS)
    assert specialists is not None
    assert specialists.status is StageStatus.SKIPPED
    assert specialists.unavailable_note


def test_running_without_a_readiness_report_is_flagged(
    make_context, scene, tool_registry
):
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=False)
    trace, _, _ = run(build_contract(context), context, tool_registry)
    stage = trace.stage(StageId.VERIFY_INPUTS)
    assert stage is not None
    assert stage.steps[0].status is StageStatus.SKIPPED
    assert stage.notes


# -- events ---------------------------------------------------------------


def test_the_stream_opens_and_closes_with_the_whole_trace(context, tool_registry):
    _, _, events = run(build_contract(context), context, tool_registry)
    assert events[0].type is RunEventType.RUN_STARTED
    assert events[0].trace is not None
    assert events[-1].type is RunEventType.RUN_FINISHED
    assert events[-1].trace is not None


def test_event_sequence_numbers_are_contiguous(context, tool_registry):
    _, _, events = run(build_contract(context), context, tool_registry)
    assert [event.seq for event in events] == list(range(1, len(events) + 1))


def test_every_stage_emits_a_start_and_a_finish(context, tool_registry):
    _, _, events = run(build_contract(context), context, tool_registry)
    started = [e.stage.id for e in events if e.type is RunEventType.STAGE_STARTED]
    finished = [e.stage.id for e in events if e.type is RunEventType.STAGE_FINISHED]
    assert started == finished
    assert len(started) == len(STAGE_SPECS)


def test_a_finished_tool_step_carries_its_full_record(context, tool_registry):
    _, _, events = run(build_contract(context), context, tool_registry)
    tool_events = [e for e in events if e.tool_run is not None]
    assert tool_events
    for event in tool_events:
        assert event.tool_run.tool
        assert event.tool_run.implementation


# -- cancellation ---------------------------------------------------------


def test_cancellation_stops_between_steps_and_marks_the_rest_unreached(
    context, tool_registry
):
    contract = build_contract(context)
    events = []
    orchestrator = Orchestrator(
        contract=contract,
        context=context,
        registry=tool_registry,
        emit=events.append,
        # Cancel as soon as the contract has been accepted.
        cancelled=lambda: any(
            e.type is RunEventType.STAGE_FINISHED
            and e.stage is not None
            and e.stage.id is StageId.ACCEPT_CONTRACT
            for e in events
        ),
    )
    trace = orchestrator.run()

    assert trace.status is RunStatus.CANCELLED
    assert trace.tool_runs == []
    assert events[-1].type is RunEventType.RUN_FINISHED


# -- status derivation ----------------------------------------------------


def _step(status: StageStatus) -> TraceStep:
    return TraceStep(
        id="x", stage=StageId.RUN_SPECIALISTS, label="x", status=status
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], StageStatus.SKIPPED),
        ([StageStatus.OK], StageStatus.OK),
        ([StageStatus.OK, StageStatus.OK], StageStatus.OK),
        ([StageStatus.SKIPPED], StageStatus.SKIPPED),
        ([StageStatus.FAILED], StageStatus.FAILED),
        ([StageStatus.FAILED, StageStatus.SKIPPED], StageStatus.FAILED),
        ([StageStatus.OK, StageStatus.SKIPPED], StageStatus.PARTIAL),
        ([StageStatus.OK, StageStatus.FAILED], StageStatus.PARTIAL),
    ],
)
def test_a_stage_summarises_its_steps_without_rounding(statuses, expected):
    assert derive_status([_step(s) for s in statuses]) is expected


# -- outcomes are kept for later stages -----------------------------------


def test_the_arrays_tools_produced_are_returned_for_later_stages(
    context, tool_registry
):
    """The map workspace must overlay the same pixels the measurements came from."""
    _, outcomes, _ = run(build_contract(context), context, tool_registry)
    assert "spectral-index-engine" in outcomes
    masks = outcomes["spectral-index-engine"].masks
    assert masks
    assert masks[0].array.any()


def test_the_trace_carries_no_raster_arrays(context, tool_registry):
    """Nothing that reaches the client should carry a raster."""
    trace, _, _ = run(build_contract(context), context, tool_registry)
    payload = trace.model_dump_json()
    assert trace.masks
    assert "array" not in payload


# -- readiness is not recomputed -----------------------------------------


def test_the_trace_and_the_gate_report_the_same_numbers(context, tool_registry):
    """One measurement, one number, shown in two places."""
    before = evaluate_readiness(context.session, context.store)
    trace, _, _ = run(build_contract(context), context, tool_registry)
    stage = trace.stage(StageId.VERIFY_INPUTS)
    assert stage is not None
    passes = len(before.by_status(CheckStatus.PASS))
    assert f"{passes} passed" in stage.steps[0].detail


def test_a_single_image_run_has_a_clean_verify_stage(context, tool_registry):
    """A ready single image must not read as a partial verification."""
    assert context.configuration is InputConfiguration.SINGLE
    trace, _, _ = run(build_contract(context), context, tool_registry)
    stage = trace.stage(StageId.VERIFY_INPUTS)
    assert stage is not None
    assert stage.status is StageStatus.OK
