"""The evidence ledger and verdict engine.

Everything upstream measures. This decides, and it has to be able to decide
against the claim, against itself, and against answering at all.

The order is deliberate:

1. Build the ledger: which measurements bear on this claim, and which way each
   one points. A measurement that is not in the ledger did not contribute.
2. Cross-check: where two methods measured the same quantity, compare them. Do
   not average them. Two estimates that disagree by 40 per cent are a finding.
3. Score confidence as named components, each with the lever that would move it.
4. Choose a label. A surviving likely confounder overrides the measurements: a
   finding that has a live alternative explanation is not a finding.
5. Write the explanation, then audit every number in it against the ledger.

The last step is the one that makes the rest trustworthy. A figure in the
explanation that does not appear in the ledger is not a rounding slip to be
footnoted, it is a fabricated measurement, and the explanation carrying it is
discarded.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.models.confounders import (
    VERDICT_PENALTY,
    ConfounderReport,
    ConfounderTest,
    ConfounderVerdict,
)
from app.models.contract import ChangeDirection, ConfounderKind
from app.models.schemas import (
    InputConfiguration,
    Measurement,
    ToolImplementation,
    ToolRequirement,
)
from app.models.verdict import (
    ConfidenceComponent,
    ConsistencyCheck,
    EvidenceDirection,
    EvidenceItem,
    EvidenceLedger,
    NumericAudit,
    Verdict,
    VerdictLabel,
)
from app.tools.base import BaseTool, ToolContext, ToolError, ToolOutcome
from app.tools.indices import WEAK_SEPARABILITY

logger = logging.getLogger(__name__)

# Confidence must reach this for the system to commit either way. Below it the
# answer is inconclusive, which is a result rather than a failure.
DECISION_CONFIDENCE = 0.5

# A change smaller than this share of the starting quantity is not treated as a
# direction: measuring a 0.4 per cent shift and calling it an increase overstates
# what a thresholded index can resolve.
DIRECTION_DEADBAND = 0.02

# Two independent estimates of the same quantity agreeing within this relative
# difference are treated as agreeing.
CONSISTENCY_TOLERANCE = 0.25

# Weights for the confidence components. Declared here, in one place, so the
# arithmetic can be checked against the intent.
WEIGHT_MEASUREMENT_QUALITY = 0.30
WEIGHT_INDEPENDENT_AGREEMENT = 0.25
WEIGHT_OBSERVABILITY = 0.15
WEIGHT_EFFECT_SIZE = 0.30
# Confounders subtract rather than add: they are reasons to doubt.
WEIGHT_CONFOUNDERS = 0.60

# How stable an answer has to be before a poorly separating index is trusted
# anyway. Set high on purpose: this is the escape hatch for the legitimate case
# where a scene is almost entirely one class, and an escape hatch that is easy to
# reach is not a standard. Below it, a low separability drags the quality down in
# proportion, because then neither signal vouches for the areas.
UNSEPARATED_STABILITY = 0.90

# Which measurements bear on a directional claim about area, and how.
DIRECTIONAL_KEYS = ("area_change_km2", "net_change_km2", "percentage_change")

# Pairs of measurements that estimate the same quantity by different routes.
CONSISTENCY_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    (
        "gain_km2",
        "corroborated_gain_km2",
        "area gained",
        "the target's own index against an independent index of the same transition",
    ),
    (
        "area_change_km2",
        "net_change_km2",
        "net area change",
        "the difference between the class totals against the sum of confirmed "
        "transitions",
    ),
)

# Numbers this small are formatting, not measurements: years, counts of tools,
# and the like. Auditing them produces noise without catching anything.
AUDIT_IGNORE = {"0", "1", "2", "100"}


class VerdictEngine(BaseTool):
    """Weighs the evidence and commits to a verdict, or declines to."""

    name = "verdict-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Builds the evidence ledger, cross-checks independent measurements of the "
        "same quantity, scores confidence as named components, and returns "
        "supported, refuted, inconclusive, or unanswerable. Every number in its "
        "explanation is audited against the ledger."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            configurations=[
                InputConfiguration.SINGLE,
                InputConfiguration.CROSS_MODAL_PAIR,
                InputConfiguration.BI_TEMPORAL_PAIR,
            ],
            description="A claim to test and at least one measurement bearing on it.",
        )

    def depends_on(self) -> tuple[str, ...]:
        # The change engine is the usual source of the measurements a claim turns
        # on. Named so the planner schedules this last rather than declaring it
        # unavailable.
        return ("change-cva-engine",)

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "narrate": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Let the language model phrase the explanation. Its output is "
                    "discarded if any number in it fails the ledger audit."
                ),
            }
        }

    def produces(self) -> list[str]:
        return ["verdict_confidence", "evidence_items", "independent_agreement"]

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]:
        if context.contract is None:
            return False, "there is no claim to rule on"
        if not any(outcome.ok for outcome in context.upstream.values()):
            return False, "no tool produced a measurement to weigh"
        return True, ""

    def execute(self, context: ToolContext) -> ToolOutcome:
        contract = context.contract
        if contract is None:
            raise ToolError("A verdict needs a claim, and none was supplied.")

        measurements = _gather(context)
        if not measurements:
            raise ToolError(
                "No measurement reached the verdict engine, so there is nothing to "
                "weigh."
            )

        report = _confounder_report(context)
        narrate = bool(context.param("narrate", True))

        outcome = self.outcome(parameters={"narrate": narrate})

        ledger = self._build_ledger(contract, measurements, report)
        verdict = self._decide(contract, ledger, measurements, context)

        if narrate:
            self._narrate(verdict, ledger, measurements)

        outcome.artifacts["verdict.ledger"] = ledger
        outcome.artifacts["verdict.result"] = verdict

        outcome.measurements.append(
            self.measurement(
                key="verdict_confidence",
                label="Confidence in the verdict",
                value=verdict.confidence,
                unit="fraction",
                formula=" + ".join(
                    f"{component.contribution:+.3f} ({component.name})"
                    for component in verdict.confidence_components
                )
                or "no component contributed",
                inputs={
                    component.name: {
                        "measured": round(component.measured, 6),
                        "weight": component.weight,
                        "contribution": round(component.contribution, 6),
                    }
                    for component in verdict.confidence_components
                },
                method=(
                    "sum of named components, each clamped to its weight; "
                    f"a verdict is only committed above {DECISION_CONFIDENCE:.0%}"
                ),
                precision=3,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="evidence_items",
                label="Measurements admitted as evidence",
                value=float(len(ledger.items)),
                unit="count",
                formula=(
                    f"{len(ledger.items)} admitted of {len(measurements)} measured"
                ),
                inputs={
                    "admitted": [item.measurement_key for item in ledger.items],
                    "independent": ledger.independent_count,
                    "excluded": ledger.excluded,
                },
                precision=0,
            )
        )
        if ledger.consistency:
            agreeing = sum(1 for check in ledger.consistency if check.agrees)
            outcome.measurements.append(
                self.measurement(
                    key="independent_agreement",
                    label="Independent estimates that agree",
                    value=agreeing / len(ledger.consistency),
                    unit="fraction",
                    formula=(
                        f"{agreeing} agreeing of {len(ledger.consistency)} "
                        "cross-checked pair(s)"
                    ),
                    inputs={
                        check.quantity: {
                            "first": f"{check.first_key}={check.first_value:.4f}",
                            "second": f"{check.second_key}={check.second_value:.4f}",
                            "relative_difference": round(
                                check.relative_difference, 4
                            ),
                            "tolerance": check.tolerance,
                            "agrees": check.agrees,
                        }
                        for check in ledger.consistency
                    },
                    method=(
                        "two routes to the same quantity, compared and not averaged"
                    ),
                    precision=3,
                )
            )

        outcome.notes.append(verdict.summary_line().capitalize() + ".")
        outcome.notes.append(verdict.reasoning)
        for check in ledger.consistency:
            outcome.notes.append(check.explanation)
        if verdict.narrative_audit is not None and not verdict.narrative_audit.passed:
            outcome.notes.append(verdict.narrative_audit.note)
        return outcome

    # -- the ledger --------------------------------------------------------
    def _build_ledger(
        self,
        contract: Any,
        measurements: dict[str, Measurement],
        report: ConfounderReport | None,
    ) -> EvidenceLedger:
        ledger = EvidenceLedger(
            confounders=list(report.tests) if report is not None else []
        )
        asserted = contract.change_direction
        target = (
            contract.target_classes[0] if contract.target_classes else "the target"
        )

        for key, measurement in measurements.items():
            admission = _admit(key, measurement, asserted, target)
            if admission is None:
                ledger.excluded[key] = _exclusion_reason(key)
                continue
            ledger.items.append(admission)

        ledger.consistency = _cross_check(measurements)
        return ledger

    # -- the decision ------------------------------------------------------
    def _decide(
        self,
        contract: Any,
        ledger: EvidenceLedger,
        measurements: dict[str, Measurement],
        context: ToolContext,
    ) -> Verdict:
        asserted = contract.change_direction
        measured, direction_basis = _measured_direction(measurements)

        components = _confidence_components(ledger, measurements, context)
        confidence = max(
            0.0, min(1.0, sum(component.contribution for component in components))
        )

        surviving = [
            test for test in ledger.confounders if test.survived
        ]
        likely = [
            test
            for test in ledger.confounders
            if test.verdict is ConfounderVerdict.LIKELY
        ]
        requirements: list[str] = []
        for test in ledger.confounders:
            if test.requirement and test.requirement not in requirements:
                requirements.append(test.requirement)

        label, reasoning = _choose_label(
            asserted=asserted,
            measured=measured,
            direction_basis=direction_basis,
            confidence=confidence,
            likely=likely,
            ledger=ledger,
            configuration=contract.configuration,
        )

        verdict = Verdict(
            label=label,
            claim=contract.claim,
            asserted_direction=asserted,
            measured_direction=measured,
            confidence=confidence,
            confidence_components=components,
            evidence=ledger.items,
            consistency=ledger.consistency,
            surviving_confounders=[test.kind for test in surviving],
            requirements=requirements,
            reasoning=reasoning,
        )
        verdict.what_would_change_it = _levers(verdict, components, requirements)
        return verdict

    # -- the explanation ---------------------------------------------------
    def _narrate(
        self,
        verdict: Verdict,
        ledger: EvidenceLedger,
        measurements: dict[str, Measurement],
    ) -> None:
        """Ask the language model to phrase the finding, then audit its numbers."""
        from app.core.narrate import narrate_verdict

        try:
            text, model = narrate_verdict(verdict, ledger)
        except Exception as exc:  # noqa: BLE001 - phrasing is never load-bearing
            logger.info("narration unavailable: %s", exc)
            verdict.narrative_audit = NumericAudit(
                passed=False,
                note=f"No phrased explanation was produced ({exc}).",
            )
            return

        if not text:
            return

        audit = audit_numbers(text, measurements, verdict)
        verdict.narrative_audit = audit
        if audit.passed:
            verdict.narrative = text
            verdict.narrative_source = model or "language model"
        else:
            verdict.narrative = None
            verdict.narrative_source = "template"
            audit.note = (
                "The phrased explanation was discarded because "
                f"{', '.join(audit.untraceable[:4])} "
                + ("does" if len(audit.untraceable) == 1 else "do")
                + " not appear in the evidence ledger. The measured reasoning above "
                "is shown instead."
            )
            logger.warning(
                "narrative rejected, untraceable figures: %s", audit.untraceable
            )


# ---------------------------------------------------------------------------
# Ledger construction
# ---------------------------------------------------------------------------


def _gather(context: ToolContext) -> dict[str, Measurement]:
    """Every measurement produced so far, keyed. Later tools win on collision."""
    gathered: dict[str, Measurement] = {}
    for outcome in context.upstream.values():
        if not outcome.ok:
            continue
        for measurement in outcome.measurements:
            gathered[measurement.key] = measurement
    return gathered


def _confounder_report(context: ToolContext) -> ConfounderReport | None:
    outcome = context.upstream.get("confounder-engine")
    if outcome is None or not outcome.ok:
        return None
    report = outcome.artifacts.get("confounders.report")
    return report if isinstance(report, ConfounderReport) else None


# Measurements that bear on a claim, with how to read them. Anything not listed
# is excluded from the ledger with a reason, so the ledger is a decision and not
# a dump of everything that happened to be computed.
_RELEVANCE: dict[str, tuple[str, str]] = {
    "area_km2_before": (
        "the starting quantity the claim is measured against",
        "baseline",
    ),
    "area_km2_after": ("the ending quantity", "endpoint"),
    "area_change_km2": (
        "the difference between the two class totals",
        "directional",
    ),
    "percentage_change": (
        "the change relative to the starting quantity",
        "directional",
    ),
    "gain_km2": (
        "area that crossed into the class and whose index confirmed it",
        "directional",
    ),
    "loss_km2": (
        "area that crossed out of the class and whose index confirmed it",
        "counter-directional",
    ),
    "net_change_km2": ("confirmed gain minus confirmed loss", "directional"),
    "corroborated_gain_km2": (
        "gain that a second, independent index also shows",
        "directional",
    ),
    "class_separability": (
        "whether the index divides this scene into two populations at all",
        "quality",
    ),
    "threshold_sensitivity": (
        "how much the answer would move if the threshold moved",
        "quality",
    ),
    "observable_fraction": (
        "how much of the overlap could be compared on both dates",
        "quality",
    ),
    "unconfirmed_crossings_km2": (
        "area that crossed the boundary without the index moving",
        "quality",
    ),
    "grounded_area_km2": ("the measured extent of the located region", "directional"),
    "grounded_separability": (
        "whether the index cleanly separates the located region",
        "quality",
    ),
    "changed_extent_of_scene": (
        "how much of the scene the detection covers",
        "quality",
    ),
}


def _admit(
    key: str,
    measurement: Measurement,
    asserted: ChangeDirection,
    target: str,
) -> EvidenceItem | None:
    """Decide whether a measurement is evidence, and which way it points."""
    entry = _RELEVANCE.get(key)
    if entry is None:
        return None
    relevance, role = entry

    direction = EvidenceDirection.NEUTRAL
    if role == "directional":
        direction = _direction_for(measurement.value, asserted)
    elif role == "counter-directional":
        direction = _direction_for(-measurement.value, asserted)

    # A quality measurement never argues for or against the claim; it governs how
    # much the directional ones are worth.
    independent = key in {"corroborated_gain_km2", "class_separability"}
    weight = 1.0 if role in {"directional", "counter-directional"} else 0.5

    return EvidenceItem(
        measurement_key=key,
        label=measurement.label,
        value=measurement.value,
        unit=measurement.unit,
        display=(
            f"{measurement.value:.{measurement.precision}f} {measurement.unit}".strip()
        ),
        direction=direction,
        relevance=relevance,
        statement=_statement(key, measurement, target),
        source_tool=measurement.source_tool,
        source_version=measurement.source_version,
        formula=measurement.formula,
        independent=independent,
        weight=weight,
    )


def _exclusion_reason(key: str) -> str:
    if key.endswith(("_mean", "_threshold", "_separability")):
        return "an intermediate quantity, not a statement about the claim"
    if key.startswith(("pixel_area", "scene_area", "scene_valid_area")):
        return "describes the input geometry rather than the claim"
    if key.startswith(("cva_", "delta_")):
        return "supports the detection but does not itself bear on the direction"
    return "not a quantity this claim turns on"


def _direction_for(value: float, asserted: ChangeDirection) -> EvidenceDirection:
    """Which way a signed quantity points, relative to what was asserted."""
    if asserted is ChangeDirection.UNSPECIFIED:
        return EvidenceDirection.NEUTRAL
    if abs(value) < 1e-9:
        return (
            EvidenceDirection.SUPPORTS
            if asserted is ChangeDirection.UNCHANGED
            else EvidenceDirection.REFUTES
        )
    rising = value > 0
    if asserted is ChangeDirection.INCREASED:
        return EvidenceDirection.SUPPORTS if rising else EvidenceDirection.REFUTES
    if asserted is ChangeDirection.DECREASED:
        return EvidenceDirection.REFUTES if rising else EvidenceDirection.SUPPORTS
    if asserted is ChangeDirection.UNCHANGED:
        return EvidenceDirection.REFUTES
    return EvidenceDirection.NEUTRAL


def _statement(key: str, measurement: Measurement, target: str) -> str:
    value = f"{measurement.value:.{measurement.precision}f} {measurement.unit}".strip()
    templates = {
        "area_km2_before": f"{target} covered {value} on the first date",
        "area_km2_after": f"{target} covered {value} on the second date",
        "area_change_km2": f"the class total moved by {value}",
        "percentage_change": f"that is a change of {value}",
        "gain_km2": f"{value} was gained",
        "loss_km2": f"{value} was lost",
        "net_change_km2": f"the confirmed net change is {value}",
        "corroborated_gain_km2": (
            f"a second index independently confirms {value} of that gain"
        ),
        "class_separability": (
            f"the index separates this scene with a score of {value}"
        ),
        "observable_fraction": f"{value} of the overlap was comparable on both dates",
        "unconfirmed_crossings_km2": (
            f"{value} crossed the boundary without the index moving"
        ),
        "grounded_area_km2": f"the located region measures {value}",
        "grounded_separability": f"the region's boundary separates at {value}",
        "changed_extent_of_scene": f"the detection covers {value} of the scene",
        "threshold_sensitivity": (
            f"nudging the threshold moves the answer by {value}"
        ),
    }
    return templates.get(key, f"{measurement.label} is {value}")


def _cross_check(measurements: dict[str, Measurement]) -> list[ConsistencyCheck]:
    """Compare independent estimates of the same quantity."""
    # The scene's own area, so "is this difference worth arguing about" can be
    # judged against the ground rather than against the smaller of two estimates.
    # The gis engine emits one of these per image role, so the smallest is taken:
    # the comparison only covers ground both dates could see.
    footprints = [
        item.value
        for key, item in measurements.items()
        if key.startswith(("scene_valid_area_km2", "scene_area_km2"))
        and item.value > 0.0
    ]
    scene_area = min(footprints) if footprints else None

    checks: list[ConsistencyCheck] = []
    for first_key, second_key, quantity, method in CONSISTENCY_PAIRS:
        first = measurements.get(first_key)
        second = measurements.get(second_key)
        if first is None or second is None:
            continue

        scale = max(abs(first.value), abs(second.value))
        difference = (
            abs(first.value - second.value) / scale if scale > 1e-12 else 0.0
        )
        # Relative difference is meaningless when both estimates are near zero.
        #
        # On a pair where almost nothing changed, two methods returned -0.1486 and
        # -0.1992 km2 on a 26 km2 scene. That is a relative difference of 25% and it
        # was scored as the methods disagreeing, which cost the finding a quarter of
        # its confidence. Both methods were in fact saying the same thing: nothing
        # measurable happened. Two estimates that both fall inside the deadband the
        # verdict already treats as no-change cannot be in conflict about a change
        # neither of them is claiming.
        negligible = (
            first.unit == second.unit
            and scene_area is not None
            and scene_area > 0.0
            and scale / scene_area <= DIRECTION_DEADBAND
        )
        agrees = negligible or difference <= CONSISTENCY_TOLERANCE

        checks.append(
            ConsistencyCheck(
                quantity=quantity,
                label=f"Two estimates of {quantity}",
                first_key=first_key,
                first_value=first.value,
                second_key=second_key,
                second_value=second.value,
                unit=first.unit,
                relative_difference=difference,
                tolerance=CONSISTENCY_TOLERANCE,
                agrees=agrees,
                method=method,
                explanation=(
                    f"{quantity.capitalize()} measured two ways: "
                    f"{first.value:.3f} and {second.value:.3f} {first.unit}, "
                    f"a relative difference of {difference:.0%}. "
                    + (
                        "Both estimates are smaller than the change this analysis "
                        "can resolve, so they are not in conflict: neither is "
                        "claiming a change."
                        if negligible
                        else f"Within the {CONSISTENCY_TOLERANCE:.0%} tolerance, so "
                        "the two routes agree."
                        if agrees
                        else f"Beyond the {CONSISTENCY_TOLERANCE:.0%} tolerance. "
                        "They are reported separately rather than averaged, because "
                        "an average of two disagreeing estimates is a number "
                        "neither method supports."
                    )
                ),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _confidence_components(
    ledger: EvidenceLedger,
    measurements: dict[str, Measurement],
    context: ToolContext,
) -> list[ConfidenceComponent]:
    components: list[ConfidenceComponent] = []

    # Two different questions, and both have to be answered before an area is
    # worth anything.
    #
    # Threshold sensitivity asks whether the answer depends on where the boundary
    # was drawn. Separability asks whether there was a boundary to draw. Neither
    # alone is sufficient, and using either alone certifies a measurement that is
    # not defensible:
    #
    # * Sensitivity alone passed a built-up claim where NDBI scored 0.00
    #   separability. The index was cutting a unimodal histogram, the regions it
    #   labelled as newly built were vegetated on both dates, and the verdict came
    #   out supported at 64%. The number agreed with the claim and its own map
    #   contradicted it.
    # * Separability alone failed a robust measurement: on farmland compared a year
    #   apart in the same season, almost the whole scene is one class, so Otsu has
    #   nothing to split and scores low even though nudging the boundary moves the
    #   answer by under one percent.
    #
    # So an index that does not divide the scene is trusted only when moving the
    # boundary barely moves the answer. When it does divide the scene, stability is
    # the whole story.
    sensitivity = measurements.get("threshold_sensitivity")
    separability = measurements.get("class_separability") or measurements.get(
        "grounded_separability"
    )

    if sensitivity is not None:
        stability = max(0.0, 1.0 - sensitivity.value)
        separated = separability is None or separability.value >= WEAK_SEPARABILITY
        if separated or stability >= UNSEPARATED_STABILITY:
            quality = stability
        else:
            # Neither signal vouches for the areas, so neither is given the benefit
            # of the doubt.
            quality = stability * (separability.value / WEAK_SEPARABILITY)

        rationale = (
            f"Moving the threshold by a twentieth of the index range changes the "
            f"answer by {sensitivity.value:.1%} of the starting area, so the finding "
            + (
                "does not rest on where the boundary was drawn."
                if sensitivity.value < 0.25
                else "depends materially on where the boundary was drawn."
            )
        )
        if not separated and stability < UNSEPARATED_STABILITY:
            rationale += (
                f" The index also scores only {separability.value:.2f} at splitting "
                "this scene, so there is no clear boundary to draw and the answer "
                "moves when it is nudged. The areas describe where a line was put, "
                "not two distinguishable populations."
            )
        lever = (
            "A classifier trained on labelled examples, which does not need a "
            "single global threshold at all"
        )
    elif separability is not None:
        quality = separability.value
        rationale = (
            "No threshold-sensitivity test was run, so this falls back to how "
            "cleanly the index splits the scene, which is a weaker signal."
        )
        lever = "A target whose index separates this scene, or a trained classifier"
    else:
        quality = 0.5
        rationale = (
            "Nothing measured how much the answer depends on its threshold, so "
            "this is scored neutrally."
        )
        lever = "A measurement of how sensitive the result is to its threshold"

    components.append(
        ConfidenceComponent(
            name="measurement_quality",
            label="How little the answer depends on its threshold",
            measured=quality,
            weight=WEIGHT_MEASUREMENT_QUALITY,
            best=WEIGHT_MEASUREMENT_QUALITY,
            contribution=WEIGHT_MEASUREMENT_QUALITY * quality,
            rationale=rationale,
            lever=lever,
        )
    )

    if ledger.consistency:
        agreeing = sum(1 for check in ledger.consistency if check.agrees)
        agreement = agreeing / len(ledger.consistency)
        rationale = (
            f"{agreeing} of {len(ledger.consistency)} cross-checked pair(s) agree "
            f"within {CONSISTENCY_TOLERANCE:.0%}."
        )
    else:
        agreement = 0.0
        rationale = (
            "Nothing measured the same quantity twice, so the finding rests on one "
            "method."
        )
    components.append(
        ConfidenceComponent(
            name="independent_agreement",
            label="Whether independent methods agree",
            measured=agreement,
            weight=WEIGHT_INDEPENDENT_AGREEMENT,
            best=WEIGHT_INDEPENDENT_AGREEMENT,
            contribution=WEIGHT_INDEPENDENT_AGREEMENT * agreement,
            rationale=rationale,
            lever="A second sensor or a second index measuring the same quantity",
        )
    )

    observable = measurements.get("observable_fraction")
    observability = observable.value if observable is not None else 1.0
    components.append(
        ConfidenceComponent(
            name="observability",
            label="How much of the area could actually be compared",
            measured=observability,
            weight=WEIGHT_OBSERVABILITY,
            best=WEIGHT_OBSERVABILITY,
            contribution=WEIGHT_OBSERVABILITY * observability,
            rationale=(
                f"{observability:.0%} of the overlap was free of cloud and nodata on "
                "both dates."
                if observable is not None
                else "A single image, so there is nothing to co-observe."
            ),
            lever="A clearer acquisition, or a radar acquisition that sees through cloud",
        )
    )

    effect, effect_note = _effect_size(measurements)
    components.append(
        ConfidenceComponent(
            name="effect_size",
            label="Whether the change is large enough to resolve",
            measured=effect,
            weight=WEIGHT_EFFECT_SIZE,
            best=WEIGHT_EFFECT_SIZE,
            contribution=WEIGHT_EFFECT_SIZE * effect,
            rationale=(
                effect_note
                + " This is a property of the scene rather than of the method, so "
                "no different analysis would improve it."
            ),
            # Deliberately no lever: nothing the system or the user can do changes
            # how big the change on the ground was, and offering a false remedy is
            # worse than offering none.
            lever="",
        )
    )

    penalty, penalty_note = _confounder_penalty(ledger.confounders)
    components.append(
        ConfidenceComponent(
            name="alternative_explanations",
            label="Alternative explanations still standing",
            measured=penalty,
            weight=WEIGHT_CONFOUNDERS,
            best=0.0,
            contribution=-WEIGHT_CONFOUNDERS * penalty,
            rationale=penalty_note,
            lever="The acquisitions named under what would settle it",
        )
    )
    return components


def _effect_size(measurements: dict[str, Measurement]) -> tuple[float, str]:
    """How big the change is relative to what it started from.

    A threshold-derived area has a noise floor. A change well above it is
    resolvable; one near it is not, whatever the index separability.
    """
    before = measurements.get("area_km2_before")
    change = measurements.get("net_change_km2") or measurements.get(
        "area_change_km2"
    )
    if before is None or change is None or before.value <= 1e-9:
        grounded = measurements.get("grounded_area_km2")
        if grounded is not None:
            return 1.0, (
                "A single-image measurement has no change to resolve, so this is "
                "not a limiting factor."
            )
        return 0.5, "There was no baseline to size the change against."

    relative = abs(change.value) / before.value
    # Saturates at a fifth: a 20 per cent shift is unambiguous at this resolution.
    score = min(1.0, relative / 0.2)
    if relative < DIRECTION_DEADBAND:
        note = (
            f"The change is {relative:.1%} of the starting area, inside the "
            f"{DIRECTION_DEADBAND:.0%} band where a thresholded index cannot "
            "resolve a direction."
        )
    else:
        note = (
            f"The change is {relative:.1%} of the starting area, "
            + (
                "large enough to resolve clearly."
                if score >= 0.8
                else "modest relative to what a thresholded index resolves."
            )
        )
    return score, note


def _confounder_penalty(
    tests: list[ConfounderTest],
) -> tuple[float, str]:
    """How much the surviving alternative explanations should cost.

    The worst single surviving explanation dominates rather than the average: one
    alternative that fully accounts for the finding is not offset by three that
    were ruled out.
    """
    if not tests:
        return 1.0, (
            "Nothing tested any alternative explanation, so none can be excluded. "
            "An untested finding is not a confirmed one."
        )

    worst = max(VERDICT_PENALTY[test.verdict] for test in tests)
    likely = [test for test in tests if test.verdict is ConfounderVerdict.LIKELY]
    standing = [test for test in tests if test.survived]

    if likely:
        note = (
            f"{', '.join(test.label.lower() for test in likely)} "
            + ("is" if len(likely) == 1 else "are")
            + " the likely explanation for this finding, which is a reason to doubt "
            "it rather than a caveat on it."
        )
    elif standing:
        note = (
            f"{len(standing)} alternative explanation(s) could not be ruled out: "
            f"{', '.join(test.label.lower() for test in standing)}."
        )
    else:
        note = (
            f"All {len(tests)} alternative explanation(s) tested were ruled out "
            "against thresholds declared before the test ran."
        )
    return worst, note


# ---------------------------------------------------------------------------
# Choosing the label
# ---------------------------------------------------------------------------


def _measured_direction(
    measurements: dict[str, Measurement],
) -> tuple[ChangeDirection, str]:
    """Which way the measurements say it moved, and what that rests on."""
    for key in DIRECTIONAL_KEYS:
        measurement = measurements.get(key)
        if measurement is None:
            continue
        before = measurements.get("area_km2_before")
        if before is not None and before.value > 1e-9 and key != "percentage_change":
            relative = abs(measurement.value) / before.value
        else:
            relative = abs(measurement.value) / 100.0 if key == "percentage_change" else 1.0

        if relative < DIRECTION_DEADBAND:
            return ChangeDirection.UNCHANGED, (
                f"{key} is {measurement.value:.4f} {measurement.unit}, within the "
                f"{DIRECTION_DEADBAND:.0%} band where no direction is resolvable"
            )
        if measurement.value > 0:
            return ChangeDirection.INCREASED, (
                f"{key} is {measurement.value:+.4f} {measurement.unit}"
            )
        return ChangeDirection.DECREASED, (
            f"{key} is {measurement.value:+.4f} {measurement.unit}"
        )
    return ChangeDirection.UNSPECIFIED, "no directional measurement was produced"


def _choose_label(
    *,
    asserted: ChangeDirection,
    measured: ChangeDirection,
    direction_basis: str,
    confidence: float,
    likely: list[ConfounderTest],
    ledger: EvidenceLedger,
    configuration: InputConfiguration,
) -> tuple[VerdictLabel, str]:
    """Commit, or say why not.

    A surviving likely alternative explanation is checked before the
    measurements. If something other than real change accounts for the finding,
    the measurements are not wrong, they are answering a different question.
    """
    if likely:
        names = ", ".join(test.label.lower() for test in likely)
        return VerdictLabel.INCONCLUSIVE, (
            f"The measurements do show a change, but {names} "
            + ("is" if len(likely) == 1 else "are")
            + f" the likely explanation for it. {likely[0].explanation} "
            "Until that is addressed the claim cannot be judged on this data."
        )

    if asserted is ChangeDirection.UNSPECIFIED:
        if not ledger.items:
            return VerdictLabel.UNANSWERABLE, (
                "The request asserts nothing to test and produced no measurement "
                "bearing on one, so there is no claim to rule on."
            )
        return VerdictLabel.INCONCLUSIVE, (
            "The request did not assert a direction, so there is nothing to support "
            f"or refute. The measurements are reported as found: {direction_basis}."
        )

    if measured is ChangeDirection.UNSPECIFIED:
        return VerdictLabel.UNANSWERABLE, (
            f"Nothing measured bears on whether the quantity {asserted.value}. "
            + (
                "A single image cannot establish a change; two acquisitions are "
                "needed."
                if configuration is InputConfiguration.SINGLE
                else "The tools that ran produced no directional measurement."
            )
        )

    if measured is not asserted:
        # Refuting needs the same confidence as supporting. Measuring a decrease
        # with an index that does not separate the class is not grounds to call
        # the claim false; it is grounds to say the measurement cannot settle it.
        if confidence < DECISION_CONFIDENCE:
            return VerdictLabel.INCONCLUSIVE, (
                f"The claim asserts the quantity {asserted.value} and the "
                f"measurements point the other way, {direction_basis}, but "
                f"confidence is {confidence:.0%}, below the "
                f"{DECISION_CONFIDENCE:.0%} needed to commit. Refuting a claim "
                "requires the same standard as supporting one, so this is reported "
                "as undecided rather than as false."
            )
        if measured is ChangeDirection.UNCHANGED:
            return VerdictLabel.REFUTED, (
                f"The claim asserts the quantity {asserted.value}, and the "
                f"measurements show no resolvable change: {direction_basis}. "
                "Everything that could explain it away was tested and cleared, so "
                "the absence of change is itself the finding."
            )
        return VerdictLabel.REFUTED, (
            f"The claim asserts the quantity {asserted.value}; the measurements "
            f"show it {measured.value}. {direction_basis.capitalize()}. "
            f"Confidence {confidence:.0%}."
        )

    if confidence < DECISION_CONFIDENCE:
        weakest = min(
            (c for c in ledger.items if c.weight > 0),
            key=lambda item: item.value,
            default=None,
        )
        return VerdictLabel.INCONCLUSIVE, (
            f"The measurements agree with the claim, {direction_basis}, but "
            f"confidence is {confidence:.0%}, below the {DECISION_CONFIDENCE:.0%} "
            "needed to commit. The confidence breakdown names which component is "
            "short."
            + (f" The weakest admitted figure is {weakest.label.lower()}." if weakest else "")
        )

    return VerdictLabel.SUPPORTED, (
        f"The claim asserts the quantity {asserted.value}, and the measurements "
        f"agree: {direction_basis}. "
        + (
            f"{len([c for c in ledger.consistency if c.agrees])} of "
            f"{len(ledger.consistency)} independent cross-check(s) agree. "
            if ledger.consistency
            else ""
        )
        + f"Every alternative explanation tested was ruled out. Confidence "
        f"{confidence:.0%}."
    )


def _levers(
    verdict: Verdict,
    components: list[ConfidenceComponent],
    requirements: list[str],
) -> list[str]:
    """What would change the answer, in order of how much it would help."""
    levers: list[str] = list(requirements)

    # Only components actually leaving confidence on the table. A penalty that
    # costs nothing is already at its best and naming a lever for it would suggest
    # an improvement that does not exist.
    shortfall = sorted(
        (c for c in components if c.headroom > 0.01 and c.lever),
        key=lambda c: c.headroom,
        reverse=True,
    )
    for component in shortfall[:2]:
        text = (
            f"{component.lever} would recover up to {component.headroom:.2f} of "
            f"confidence ({component.label.lower()})."
        )
        if text not in levers:
            levers.append(text)
    return levers


# ---------------------------------------------------------------------------
# The Never-Guess guard
# ---------------------------------------------------------------------------

# Public because the evidence packet applies the same rule to the figures it
# prints. One definition of "is this number traceable" is the point: a report and
# a narration that disagreed about what counts as traceable would make the
# guarantee meaningless.
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def audit_numbers(
    text: str,
    measurements: dict[str, Measurement],
    verdict: Verdict,
) -> NumericAudit:
    """Check that every figure in the text traces to a measurement.

    Matching allows for reasonable presentation: a value may be rounded to fewer
    decimals, or expressed as a percentage of a fraction. What it may not be is
    absent from the ledger, which is what an invented figure looks like.
    """
    audit = NumericAudit()
    known = admissible_figures(measurements, verdict)

    for token in NUMBER_PATTERN.findall(text):
        if token.lstrip("-") in AUDIT_IGNORE:
            continue
        audit.checked += 1
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - the pattern only matches numbers
            continue
        if figure_matches(value, known, tolerance_for(token)):
            audit.traced += 1
        else:
            audit.untraceable.append(token)

    audit.passed = not audit.untraceable
    if audit.passed:
        audit.note = (
            f"All {audit.traced} figure(s) in the explanation trace to a measurement "
            "in the ledger."
        )
    return audit


def admissible_figures(
    measurements: dict[str, Measurement], verdict: Verdict | None = None
) -> list[float]:
    """Every figure the system itself would print.

    Both the raw value and the value at the precision the measuring tool declared,
    because a measurement of 0.3234 is legitimately shown as 0.32 and a figure the
    system would display is by definition traceable. Percentage forms are included
    for the same reason.
    """
    known: list[float] = []
    for measurement in measurements.values():
        known.append(measurement.value)
        known.append(round(measurement.value, measurement.precision))
        known.append(measurement.value * 100.0)
        known.append(round(measurement.value * 100.0))

    if verdict is None:
        return known

    for value in (
        verdict.confidence,
        *(c.measured for c in verdict.confidence_components),
        *(abs(c.contribution) for c in verdict.confidence_components),
        *(c.relative_difference for c in verdict.consistency),
    ):
        known.append(value)
        known.append(round(value, 2))
        known.append(value * 100.0)
        known.append(round(value * 100.0))

    return known


def tolerance_for(token: str) -> float:
    """Half of the least significant digit the figure claims.

    A figure printed as 32 claims nothing finer than a unit, so it may stand for
    anything within half of one. A figure printed as -1.3182 claims four decimal
    places and has to match to within half of the last one. Precision asserted is
    precision required.
    """
    decimals = len(token.split(".", 1)[1]) if "." in token else 0
    return 0.5 * (10.0**-decimals)


def figure_matches(value: float, known: list[float], tolerance: float) -> bool:
    return any(abs(candidate - value) <= tolerance + 1e-9 for candidate in known)


__all__ = [
    "CONSISTENCY_TOLERANCE",
    "DECISION_CONFIDENCE",
    "DIRECTION_DEADBAND",
    "NUMBER_PATTERN",
    "VerdictEngine",
    "admissible_figures",
    "audit_numbers",
    "figure_matches",
    "tolerance_for",
]
