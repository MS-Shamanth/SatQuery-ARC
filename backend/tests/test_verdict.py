"""Verdict engine tests.

The engine has to be able to answer against the claim, against itself, and
against answering at all. These tests are organised around those three, plus the
numeric guard, which is the one mechanism in the system that can catch a
fabricated number before a user sees it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.models.confounders import (
    ConfounderReport,
    ConfounderTest,
    ConfounderVerdict,
)
from app.models.contract import (
    AnalysisContract,
    ChangeDirection,
    ConfounderKind,
    ContractSource,
    ContractTaskType,
)
from app.models.schemas import ImageRole, InputConfiguration, Measurement
from app.models.verdict import EvidenceDirection, Verdict, VerdictLabel
from app.tools.base import ToolOutcome
from app.tools.verdict import (
    DECISION_CONFIDENCE,
    VerdictEngine,
    audit_numbers,
)
from app.models.schemas import ToolImplementation


@pytest.fixture
def engine():
    return VerdictEngine()


def measurement(
    key: str,
    value: float,
    unit: str = "km2",
    *,
    precision: int = 3,
    tool: str = "change-cva-engine",
) -> Measurement:
    return Measurement(
        key=key,
        label=key.replace("_", " "),
        value=value,
        unit=unit,
        formula=f"{key} = measured",
        inputs={"pixel_count": 100},
        source_tool=tool,
        source_version="1.0.0",
        precision=precision,
    )


def contract_for(
    direction: ChangeDirection,
    *,
    session_id: str = "a" * 32,
    configuration: InputConfiguration = InputConfiguration.BI_TEMPORAL_PAIR,
    task: ContractTaskType = ContractTaskType.CLAIM_INVESTIGATION,
) -> AnalysisContract:
    return AnalysisContract(
        session_id=session_id,
        query="Has built-up area increased?",
        claim="Built-up area increased between the two acquisitions.",
        task_type=task,
        configuration=configuration,
        acts_on=[ImageRole.DATE_A, ImageRole.DATE_B],
        target_classes=["built-up"],
        change_direction=direction,
        source=ContractSource.OFFLINE_RULE_ROUTER,
        contract_hash="c" * 32,
    )


def confounder(
    kind: ConfounderKind, verdict: ConfounderVerdict, *, requirement: str | None = None
) -> ConfounderTest:
    return ConfounderTest(
        kind=kind,
        label=kind.value.replace("_", " ").title(),
        question="Could something else explain this?",
        verdict=verdict,
        measured="0.42 ratio",
        threshold="likely above 0.5",
        formula="a / b",
        method="a test",
        explanation="Because of what was measured.",
        requirement=requirement,
    )


def upstream(
    measurements: list[Measurement],
    *,
    confounders: list[ConfounderTest] | None = None,
) -> dict[str, ToolOutcome]:
    change = ToolOutcome(
        tool="change-cva-engine",
        version="1.0.0",
        implementation=ToolImplementation.DETERMINISTIC,
        ok=True,
        measurements=measurements,
    )
    outcomes = {"change-cva-engine": change}
    if confounders is not None:
        report = ConfounderReport(tests=confounders)
        outcomes["confounder-engine"] = ToolOutcome(
            tool="confounder-engine",
            version="1.0.0",
            implementation=ToolImplementation.DETERMINISTIC,
            ok=True,
            artifacts={"confounders.report": report},
        )
    return outcomes


def rule(
    make_context,
    scene_paths,
    *,
    direction: ChangeDirection,
    measurements: list[Measurement],
    confounders: list[ConfounderTest] | None = None,
    narrate: bool = False,
) -> tuple[ToolOutcome, Verdict]:
    context = make_context(scene_paths, target_classes=["built-up"])
    probe = replace(
        context,
        parameters={"narrate": narrate},
        upstream=upstream(measurements, confounders=confounders),
        contract=contract_for(direction, session_id=context.session.session_id),
    )
    outcome = VerdictEngine().run(probe)
    assert outcome.ok, outcome.skipped_reason
    return outcome, outcome.artifacts["verdict.result"]


@pytest.fixture
def pair(tmp_path):
    from tests.raster_fixtures import make_labels, make_optical_scene

    a = make_optical_scene(tmp_path / "a.tif", labels=make_labels(), seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=make_labels(), seed=4)
    return {ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path}


CLEAN = [
    confounder(ConfounderKind.SEASONALITY, ConfounderVerdict.RULED_OUT),
    confounder(ConfounderKind.MISREGISTRATION, ConfounderVerdict.RULED_OUT),
    confounder(ConfounderKind.RADIOMETRY, ConfounderVerdict.RULED_OUT),
]

STRONG = [
    measurement("area_km2_before", 10.0),
    measurement("area_km2_after", 14.0),
    measurement("area_change_km2", 4.0),
    measurement("net_change_km2", 3.9),
    measurement("gain_km2", 4.1),
    measurement("corroborated_gain_km2", 3.9),
    measurement("loss_km2", 0.2),
    measurement("class_separability", 0.85, "ratio"),
    measurement("observable_fraction", 0.98, "fraction", precision=4),
]


# -- committing -----------------------------------------------------------


def test_a_clean_matching_finding_is_supported(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    assert verdict.label is VerdictLabel.SUPPORTED
    assert verdict.confidence >= DECISION_CONFIDENCE
    assert verdict.measured_direction is ChangeDirection.INCREASED


def test_an_opposite_direction_with_good_evidence_is_refuted(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.DECREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    assert verdict.label is VerdictLabel.REFUTED
    assert verdict.asserted_direction is ChangeDirection.DECREASED
    assert verdict.measured_direction is ChangeDirection.INCREASED


def test_refuting_requires_the_same_confidence_as_supporting(make_context, pair):
    """Measuring the other direction badly is not grounds to call a claim false."""
    weak = [
        measurement("area_km2_before", 10.0),
        measurement("area_km2_after", 14.0),
        measurement("area_change_km2", 4.0),
        # The index does not separate the class, so the areas describe a threshold
        # rather than the class.
        measurement("class_separability", 0.0, "ratio"),
        measurement("observable_fraction", 0.4, "fraction"),
    ]
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.DECREASED,
        measurements=weak, confounders=CLEAN,
    )
    assert verdict.confidence < DECISION_CONFIDENCE
    assert verdict.label is VerdictLabel.INCONCLUSIVE
    assert "same standard as supporting" in verdict.reasoning


# -- declining ------------------------------------------------------------


def test_a_likely_confounder_overrides_the_measurements(make_context, pair):
    """A finding with a live alternative explanation is not a finding."""
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG,
        confounders=[
            confounder(
                ConfounderKind.SEASONALITY,
                ConfounderVerdict.LIKELY,
                requirement="An acquisition from the same season",
            ),
            confounder(ConfounderKind.MISREGISTRATION, ConfounderVerdict.RULED_OUT),
        ],
    )
    assert verdict.label is VerdictLabel.INCONCLUSIVE
    assert ConfounderKind.SEASONALITY in verdict.surviving_confounders
    assert "likely explanation" in verdict.reasoning
    assert "An acquisition from the same season" in verdict.requirements


def test_an_untested_claim_cannot_reach_full_confidence(make_context, pair):
    """Nothing tested means nothing excluded, and that has to cost something."""
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=[],
    )
    penalty = next(
        c for c in verdict.confidence_components
        if c.name == "alternative_explanations"
    )
    assert penalty.contribution < 0
    assert "not a confirmed one" in penalty.rationale


def test_a_claim_with_no_directional_measurement_is_unanswerable(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=[measurement("class_separability", 0.8, "ratio")],
        confounders=CLEAN,
    )
    assert verdict.label is VerdictLabel.UNANSWERABLE
    assert "bears on whether" in verdict.reasoning


def test_a_change_inside_the_deadband_reads_as_unchanged(make_context, pair):
    """A 0.4 per cent shift is not a direction a thresholded index can resolve."""
    flat = [
        measurement("area_km2_before", 10.0),
        measurement("area_km2_after", 10.04),
        measurement("area_change_km2", 0.04),
        measurement("class_separability", 0.85, "ratio"),
        measurement("observable_fraction", 0.99, "fraction"),
    ]
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=flat, confounders=CLEAN,
    )
    assert verdict.measured_direction is ChangeDirection.UNCHANGED
    assert verdict.label in {VerdictLabel.REFUTED, VerdictLabel.INCONCLUSIVE}


# -- the ledger -----------------------------------------------------------


def test_the_ledger_records_what_it_set_aside_and_why(make_context, pair):
    outcome, _ = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG + [measurement("ndvi_mean", 0.4, "index")],
        confounders=CLEAN,
    )
    ledger = outcome.artifacts["verdict.ledger"]
    assert "ndvi_mean" in ledger.excluded
    assert ledger.excluded["ndvi_mean"]
    assert all(item.measurement_key != "ndvi_mean" for item in ledger.items)


def test_every_admitted_item_carries_its_provenance(make_context, pair):
    outcome, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    assert verdict.evidence
    for item in verdict.evidence:
        assert item.source_tool
        assert item.formula
        assert item.relevance
        assert item.statement
        assert item.display


def test_loss_argues_against_a_claim_of_increase(make_context, pair):
    """A counter-directional measurement has to point the other way."""
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    loss = next(i for i in verdict.evidence if i.measurement_key == "loss_km2")
    gain = next(i for i in verdict.evidence if i.measurement_key == "gain_km2")
    assert loss.direction is EvidenceDirection.REFUTES
    assert gain.direction is EvidenceDirection.SUPPORTS


def test_a_quality_measurement_argues_neither_way(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    quality = next(
        i for i in verdict.evidence if i.measurement_key == "class_separability"
    )
    assert quality.direction is EvidenceDirection.NEUTRAL


# -- cross-checking -------------------------------------------------------


def test_two_estimates_of_the_same_quantity_are_compared_not_averaged(
    make_context, pair
):
    outcome, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    check = next(c for c in verdict.consistency if c.first_key == "gain_km2")
    assert check.second_key == "corroborated_gain_km2"
    assert check.agrees
    assert check.relative_difference == pytest.approx(abs(4.1 - 3.9) / 4.1, rel=1e-6)
    # Neither the mean nor any blend of the two appears anywhere.
    assert all(
        item.value != pytest.approx(4.0, abs=1e-9)
        or item.measurement_key == "area_change_km2"
        for item in verdict.evidence
    )


def test_disagreeing_estimates_are_reported_as_disagreeing(make_context, pair):
    conflicting = [
        measurement("area_km2_before", 10.0),
        measurement("area_km2_after", 14.0),
        measurement("area_change_km2", 4.0),
        measurement("gain_km2", 4.0),
        # The second index sees almost none of the gain the first one found.
        measurement("corroborated_gain_km2", 0.5),
        measurement("class_separability", 0.85, "ratio"),
        measurement("observable_fraction", 0.99, "fraction"),
    ]
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=conflicting, confounders=CLEAN,
    )
    check = next(c for c in verdict.consistency if c.first_key == "gain_km2")
    assert not check.agrees
    assert "rather than averaged" in check.explanation
    agreement = next(
        c for c in verdict.confidence_components if c.name == "independent_agreement"
    )
    assert agreement.contribution < agreement.weight


def test_two_estimates_of_almost_nothing_are_not_in_conflict(make_context, pair):
    """A relative difference between two near-zero figures says nothing.

    Measured on real imagery of the same fields a year apart in the same season,
    two methods returned -0.1486 and -0.1992 km2 on a scene of about 26 km2. That
    is 25% apart and was scored as the methods disagreeing, costing the finding a
    quarter of its confidence. Both were saying the same thing: nothing measurable
    happened. Two estimates that both sit inside the deadband the verdict already
    treats as no-change cannot be in conflict about a change neither is claiming.
    """
    near_zero = [
        measurement("area_km2_before", 23.8),
        measurement("area_km2_after", 23.6),
        measurement("area_change_km2", -0.1486),
        measurement("net_change_km2", -0.1992),
        measurement("scene_valid_area_km2.date_a", 26.21),
        measurement("class_separability", 0.26, "ratio"),
        measurement("threshold_sensitivity", 0.009, "fraction"),
        measurement("observable_fraction", 1.0, "fraction"),
    ]
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.DECREASED,
        measurements=near_zero, confounders=CLEAN,
    )
    check = next(
        c for c in verdict.consistency if c.first_key == "area_change_km2"
    )
    assert check.agrees
    assert "neither is claiming a change" in check.explanation
    # The relative difference is still reported honestly; it is the reading of it
    # that changes.
    assert check.relative_difference > 0.2
    agreement = next(
        c for c in verdict.confidence_components if c.name == "independent_agreement"
    )
    assert agreement.contribution == pytest.approx(agreement.weight)


def test_a_real_disagreement_is_not_excused_by_the_scene_being_large(
    make_context, pair
):
    """The exemption is for negligible quantities, not for large scenes.

    Without this the rule would be a loophole: any disagreement could be waved
    through by pointing at a big enough footprint.
    """
    real_conflict = [
        measurement("area_km2_before", 10.0),
        measurement("area_km2_after", 14.0),
        measurement("area_change_km2", 4.0),
        measurement("net_change_km2", 1.0),
        measurement("scene_valid_area_km2.date_a", 26.21),
        measurement("class_separability", 0.85, "ratio"),
        measurement("observable_fraction", 0.99, "fraction"),
    ]
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=real_conflict, confounders=CLEAN,
    )
    check = next(
        c for c in verdict.consistency if c.first_key == "area_change_km2"
    )
    assert not check.agrees


# -- confidence -----------------------------------------------------------


def test_confidence_is_the_sum_of_its_named_components(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    total = sum(c.contribution for c in verdict.confidence_components)
    assert verdict.confidence == pytest.approx(max(0.0, min(1.0, total)), abs=1e-9)


def test_every_component_states_what_it_measured_and_why(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    assert len(verdict.confidence_components) >= 4
    for component in verdict.confidence_components:
        assert component.rationale
        assert component.label
        assert 0.0 <= component.weight <= 1.0


def test_a_component_at_its_best_offers_no_lever(make_context, pair):
    """Suggesting an improvement that does not exist is worse than suggesting none."""
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN,
    )
    observability = next(
        c for c in verdict.confidence_components if c.name == "observability"
    )
    assert observability.headroom < 0.01
    assert all(
        observability.lever not in lever for lever in verdict.what_would_change_it
    )


def test_confidence_is_reported_even_when_it_is_low(make_context, pair):
    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=[
            measurement("area_km2_before", 10.0),
            measurement("area_km2_after", 10.3),
            measurement("area_change_km2", 0.3),
            measurement("class_separability", 0.1, "ratio"),
            measurement("observable_fraction", 0.3, "fraction"),
        ],
        confounders=[
            confounder(
                ConfounderKind.RADIOMETRY,
                ConfounderVerdict.PLAUSIBLE,
                requirement="Consistently corrected reflectance products",
            )
        ],
    )
    assert 0.0 <= verdict.confidence < DECISION_CONFIDENCE
    assert verdict.label is VerdictLabel.INCONCLUSIVE
    assert verdict.what_would_change_it


# -- the numeric guard ----------------------------------------------------


def _verdict_with(values: list[Measurement]) -> tuple[dict, Verdict]:
    measurements = {m.key: m for m in values}
    verdict = Verdict(
        label=VerdictLabel.SUPPORTED,
        claim="Built-up area increased.",
        confidence=0.7234,
    )
    return measurements, verdict


def test_a_figure_that_matches_a_measurement_is_traced():
    measurements, verdict = _verdict_with([measurement("gain_km2", 4.123)])
    audit = audit_numbers("The area gained is 4.123 km2.", measurements, verdict)
    assert audit.passed
    assert audit.traced == 1


def test_a_figure_rounded_to_the_declared_precision_is_traced():
    """A measurement of 4.1234 shown as 4.123 is the system's own rendering."""
    measurements, verdict = _verdict_with([measurement("gain_km2", 4.1234)])
    audit = audit_numbers("The area gained is 4.123 km2.", measurements, verdict)
    assert audit.passed


def test_a_confidence_written_as_a_percentage_is_traced():
    measurements, verdict = _verdict_with([measurement("gain_km2", 4.0)])
    audit = audit_numbers("Confidence is 72%.", measurements, verdict)
    assert audit.passed


def test_an_invented_figure_is_caught():
    """The Never-Guess Rule as a check that can fail."""
    measurements, verdict = _verdict_with([measurement("gain_km2", 4.123)])
    audit = audit_numbers(
        "The area gained is 4.123 km2, about 17.6% of the district.",
        measurements,
        verdict,
    )
    assert not audit.passed
    assert "17.6" in audit.untraceable


def test_precision_asserted_is_precision_required():
    """Claiming four decimals means matching to four decimals."""
    measurements, verdict = _verdict_with([measurement("gain_km2", 4.1234)])
    loose = audit_numbers("Gained 4 km2.", measurements, verdict)
    tight = audit_numbers("Gained 4.1200 km2.", measurements, verdict)
    assert loose.passed
    assert not tight.passed


def test_a_narrative_that_fails_the_audit_is_not_shown(make_context, pair, monkeypatch):
    """A fabricated figure must cost the whole explanation, not earn a footnote."""
    import app.core.narrate as narrate_module

    monkeypatch.setattr(
        narrate_module,
        "narrate_verdict",
        lambda verdict, ledger: (
            "Built-up area grew by 4.100 km2, which is 61% of the floodplain.",
            "stub-model",
        ),
    )

    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN, narrate=True,
    )
    assert verdict.narrative is None
    assert verdict.narrative_source == "template"
    assert verdict.narrative_audit is not None
    assert not verdict.narrative_audit.passed
    assert "61" in verdict.narrative_audit.untraceable
    assert "discarded" in verdict.narrative_audit.note
    # The deterministic reasoning survives, so the finding is not lost with it.
    assert verdict.reasoning


def test_a_narrative_that_passes_the_audit_is_shown(make_context, pair, monkeypatch):
    import app.core.narrate as narrate_module

    monkeypatch.setattr(
        narrate_module,
        "narrate_verdict",
        lambda verdict, ledger: (
            "Built-up area grew by 4.100 km2, confirmed independently at 3.900 km2.",
            "stub-model",
        ),
    )

    _, verdict = rule(
        make_context, pair, direction=ChangeDirection.INCREASED,
        measurements=STRONG, confounders=CLEAN, narrate=True,
    )
    assert verdict.narrative is not None
    assert verdict.narrative_source == "stub-model"
    assert verdict.narrative_audit.passed


# -- requirements ---------------------------------------------------------


def test_the_engine_refuses_to_rule_without_a_claim(make_context, pair, engine):
    context = make_context(pair)
    probe = replace(context, upstream=upstream(STRONG), contract=None)
    ready, reason = engine.upstream_ready(probe)
    assert not ready
    assert "no claim" in reason
    # The input itself is fine; only the run is not ready.
    assert engine.can_run(probe)[0]


def test_the_engine_refuses_to_rule_without_measurements(make_context, pair, engine):
    context = make_context(pair)
    probe = replace(
        context,
        upstream={},
        contract=contract_for(ChangeDirection.INCREASED),
    )
    ready, reason = engine.upstream_ready(probe)
    assert not ready
    assert "measurement" in reason

    outcome = engine.run(probe)
    assert not outcome.ok
    assert "measurement" in (outcome.skipped_reason or "")


def test_the_engine_is_plannable_before_anything_has_run(make_context, pair):
    from app.core.contract import runnable_once_scheduled
    from app.core.registry import build_default_registry

    context = make_context(pair)
    assert runnable_once_scheduled(
        "verdict-engine", build_default_registry(), context, planned=set()
    )
