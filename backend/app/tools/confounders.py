"""The confounder engine.

This is the part of the system that tries to talk itself out of its own finding.

Each test takes the change the previous stage measured and asks whether something
other than real ground change could produce it. The tests are deliberately
adversarial: they are constructed so that a scene where the alternative
explanation is true will trip them, not so that a scene where it is false will
pass them.

The most important of these is the seasonality test, and its logic is worth
stating plainly because it is the whole argument of the project in one
measurement. Real localised change moves a region while leaving the rest of the
scene alone. Phenology moves the whole scene at once. So the test compares how
much the index shifted inside the detected region against how much it shifted
everywhere else. If everything moved, nothing changed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.raster_io import read_role_reflectance, structure_band_index
from app.models.confounders import (
    ConfounderReport,
    ConfounderTest,
    ConfounderVerdict,
)
from app.models.contract import (
    CONFOUNDER_LABEL,
    CONFOUNDER_QUESTION,
    ConfounderKind,
)
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, ToolContext, ToolError, ToolOutcome
from app.tools.gis import adjacency_fraction, mask_area_km2

logger = logging.getLogger(__name__)

# -- seasonality ----------------------------------------------------------

# A pair separated by less than this many months in the annual cycle is
# comparable: crops and canopies are in a similar state.
SEASONAL_SAFE_MONTHS = 2

# If the index shifted outside the detected region by at least this share of the
# shift inside it, the shift is scene-wide and the detection is not localised.
SCENE_WIDE_SHARE_LIKELY = 0.5
SCENE_WIDE_SHARE_PLAUSIBLE = 0.25

# A scene-wide shift smaller than this is within noise and does not indicate
# phenology whatever the calendar says.
SCENE_WIDE_NEGLIGIBLE = 0.02

# When the detected region covers this much of the observable scene, the
# inside-against-outside comparison stops discriminating: "outside" has become, by
# construction, the small part that did not move, so the contrast is guaranteed to
# look large whatever the cause.
#
# At that point extent is the discriminator instead. Land cover does not get
# repainted across half a landscape in a single season by construction,
# deforestation, or flooding of that footprint without other evidence; a whole-
# scene shift over a few months is what a growing cycle looks like.
SCENE_WIDE_EXTENT_LIKELY = 0.5
SCENE_WIDE_EXTENT_CIRCULAR = 0.35

# -- misregistration ------------------------------------------------------

# Detected change concentrated on edges by more than this multiple of the scene's
# own edge density is the signature of a shifted image rather than a changed one.
EDGE_ENRICHMENT_LIKELY = 2.0
EDGE_ENRICHMENT_PLAUSIBLE = 1.5

# Gradient magnitude above this percentile counts as an edge.
#
# Deliberately tight. The baseline this is compared against is the dilated edge
# mask's own coverage, so a loose definition leaves no room to measure with: the
# top decile dilated by one pixel already covers something like 45 per cent of a
# scene, which caps the achievable enrichment at about 2.2 and makes the
# thresholds below unreachable. The top few per cent dilates to roughly 15 per
# cent, which gives the ratio somewhere to move.
EDGE_PERCENTILE = 97.0

# -- cloud and shadow ----------------------------------------------------

# Cloud is excluded from the measurement, but its fringe is not sharply bounded.
# Change hugging the excluded region by more than this is suspect.
CLOUD_FRINGE_SHARE_LIKELY = 0.4
CLOUD_FRINGE_SHARE_PLAUSIBLE = 0.2
CLOUD_FRINGE_PIXELS = 3

# -- radiometry ----------------------------------------------------------

# Reflectance shift on pseudo-invariant features. Bright hard surfaces do not
# change between two dates, so if their reflectance moved, the calibration or the
# atmosphere did.
PIF_SHIFT_LIKELY = 0.04
PIF_SHIFT_PLAUSIBLE = 0.02
PIF_PERCENTILE = 85.0

# -- sensor mismatch -----------------------------------------------------

GSD_RATIO_PLAUSIBLE = 1.5
GSD_RATIO_LIKELY = 3.0


class ConfounderEngine(BaseTool):
    """Tests whether something other than real change produced the finding."""

    name = "confounder-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Tries to explain the finding away. Tests seasonality, misregistration, "
        "cloud fringe, radiometric drift, and sensor mismatch, each against a "
        "threshold declared in advance, and names the data that would settle "
        "anything it cannot rule out."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            configurations=[
                InputConfiguration.BI_TEMPORAL_PAIR,
                InputConfiguration.CROSS_MODAL_PAIR,
            ],
            requires_crs=True,
            description=(
                "A pair, plus a change measurement to test. Runs after the change "
                "engine and reads its output."
            ),
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "kinds": {
                "type": "array",
                "items": {"enum": [kind.value for kind in ConfounderKind]},
                "default": None,
                "description": "Which alternatives to test. Defaults to the contract's.",
            }
        }

    def produces(self) -> list[str]:
        return [
            "scene_wide_index_shift",
            "region_index_shift",
            "scene_wide_share",
            "changed_extent_of_scene",
            "edge_enrichment",
            "cloud_fringe_share",
            "pseudo_invariant_shift",
            "confounders_ruled_out",
        ]

    def depends_on(self) -> tuple[str, ...]:
        return ("change-cva-engine",)

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]:
        """A finding to test, from whichever tool produced one.

        The change engine is the usual source, but not the only one. On a
        cross-modal pair there is no change measurement and yet the alternative
        explanations still matter: sensor mismatch and cloud both bear on whether
        the two instruments should have been expected to agree.
        """
        if _finding_source(context) is None:
            return False, (
                "no tool has produced a finding to test yet; the change engine or "
                "the fusion engine has to run first"
            )
        return True, ""

    def execute(self, context: ToolContext) -> ToolOutcome:
        upstream = context.upstream.get("change-cva-engine")
        finding = _finding_source(context)
        if finding is None:
            raise ToolError(
                "The confounder engine tests a finding, and no tool produced one."
            )
        grid = finding.grid
        delta = finding.delta
        invalid = finding.invalid
        detected = finding.detected
        index_name = finding.index_name
        role_a, role_b = finding.role_a, finding.role_b
        if finding.note:
            logger.info("confounder engine testing %s", finding.note)

        requested = context.param("kinds")
        kinds = (
            [ConfounderKind(value) for value in requested]
            if requested
            else self._default_kinds(context)
        )

        outcome = self.outcome(
            parameters={
                "kinds": [kind.value for kind in kinds],
                "index": index_name,
                "finding": finding.note,
            }
        )
        report = ConfounderReport()

        testers = {
            ConfounderKind.SEASONALITY: self._test_seasonality,
            ConfounderKind.MISREGISTRATION: self._test_misregistration,
            ConfounderKind.CLOUD_SHADOW: self._test_cloud_fringe,
            ConfounderKind.RADIOMETRY: self._test_radiometry,
            ConfounderKind.SENSOR_MISMATCH: self._test_sensor_mismatch,
        }

        state = _State(
            context=context,
            grid=grid,
            delta=delta,
            invalid=invalid,
            detected=detected,
            index_name=index_name,
            role_a=role_a,
            role_b=role_b,
            outcome=outcome,
            has_delta=finding.has_delta,
        )

        # Two of the tests compare an index shift inside the finding against the
        # shift outside it. Across two sensors there is no shared index to
        # difference, so those tests are declared untestable rather than run on a
        # quantity that does not exist.
        needs_delta = {ConfounderKind.SEASONALITY, ConfounderKind.RADIOMETRY}

        for kind in kinds:
            tester = testers.get(kind)
            if tester is not None and kind in needs_delta and not finding.has_delta:
                report.untested_kinds.append(kind)
                report.tests.append(
                    ConfounderTest(
                        kind=kind,
                        label=CONFOUNDER_LABEL[kind],
                        question=CONFOUNDER_QUESTION[kind],
                        verdict=ConfounderVerdict.NOT_TESTED,
                        measured="not measurable across two sensors",
                        threshold="needs one index differenced between two dates",
                        explanation=(
                            "This test compares how much an index shifted inside the "
                            "finding against how much it shifted outside it. "
                            "Reflectance and radar backscatter are not the same "
                            "quantity, so there is no shared index to difference "
                            "here. Testing it would need two acquisitions from the "
                            "same sensor."
                        ),
                        requirement=(
                            "Two acquisitions from the same sensor, so one index can "
                            "be differenced between them"
                        ),
                    )
                )
                continue

            if tester is None:
                report.untested_kinds.append(kind)
                report.tests.append(
                    ConfounderTest(
                        kind=kind,
                        label=CONFOUNDER_LABEL[kind],
                        question=CONFOUNDER_QUESTION[kind],
                        verdict=ConfounderVerdict.NOT_TESTED,
                        measured="not measured",
                        threshold="no test exists in this build",
                        explanation=(
                            "This alternative explanation is named but untested, so "
                            "it is left on the table rather than assumed away."
                        ),
                        requirement=None,
                    )
                )
                continue

            started = time.perf_counter()
            try:
                test = tester(state)
            except ToolError as exc:
                test = ConfounderTest(
                    kind=kind,
                    label=CONFOUNDER_LABEL[kind],
                    question=CONFOUNDER_QUESTION[kind],
                    verdict=ConfounderVerdict.NOT_TESTED,
                    measured="test could not run",
                    threshold="n/a",
                    explanation=str(exc),
                )
            test.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            report.tests.append(test)

        outcome.artifacts["confounders.report"] = report
        outcome.notes.append(report.summary_line().capitalize() + ".")
        for test in report.tests:
            outcome.notes.append(
                f"{test.label}: {test.verdict.value.replace('_', ' ')} "
                f"({test.measured}). {test.explanation}"
            )

        outcome.measurements.append(
            self.measurement(
                key="confounders_ruled_out",
                label="Alternative explanations ruled out",
                value=float(len(report.ruled_out)),
                unit="count",
                formula=(
                    f"{len(report.ruled_out)} of {len(report.tests)} tested "
                    "alternatives fell below their threshold"
                ),
                inputs={
                    "tested": [test.kind.value for test in report.tests],
                    "ruled_out": [test.kind.value for test in report.ruled_out],
                    "surviving": [test.kind.value for test in report.surviving],
                },
                applies_to=[role_a, role_b],
                precision=0,
            )
        )
        return outcome

    def _default_kinds(self, context: ToolContext) -> list[ConfounderKind]:
        """Alternatives worth testing for this input when none were named."""
        kinds = [
            ConfounderKind.SEASONALITY,
            ConfounderKind.MISREGISTRATION,
            ConfounderKind.RADIOMETRY,
        ]
        for role in context.roles:
            if context.metadata(role).band_index(BandRole.SCL) is not None:
                kinds.append(ConfounderKind.CLOUD_SHADOW)
                break
        kinds.append(ConfounderKind.SENSOR_MISMATCH)
        return kinds

    # -- seasonality -------------------------------------------------------
    def _test_seasonality(self, state: _State) -> ConfounderTest:
        """Did the whole scene move, or only the detected region?

        The discriminating measurement, and the reason the system can refuse a
        claim that a simple difference image would happily support.
        """
        context = state.context
        report = context.readiness
        month_delta = report.month_of_year_delta if report else None
        day_delta = report.day_delta if report else None

        usable = ~state.invalid
        inside = state.delta[state.detected & usable]
        outside = state.delta[~state.detected & usable]

        if inside.size < 32:
            raise ToolError(
                "Too little change was detected to ask whether it is seasonal."
            )

        observable = inside.size + outside.size
        extent = inside.size / observable if observable else 0.0

        # Medians, not means: a few extreme pixels should not decide this.
        region_shift = float(np.median(inside))

        if outside.size < 32:
            # The detection is the entire observable scene. There is no outside
            # left to compare against, and that is not a missing measurement: a
            # shift that covers everything is by definition scene-wide.
            scene_shift = region_shift
            share = 1.0
        else:
            scene_shift = float(np.median(outside))
            share = abs(scene_shift) / abs(region_shift) if region_shift else 0.0

        state.outcome.measurements.extend(
            [
                self.measurement(
                    key="region_index_shift",
                    label=f"Median {state.index_name} shift inside the detected region",
                    value=region_shift,
                    unit="index",
                    formula=(
                        f"median({state.index_name} on {state.role_b.value} - on "
                        f"{state.role_a.value}) over detected pixels"
                    ),
                    inputs={"pixels": int(inside.size)},
                    applies_to=[state.role_a, state.role_b],
                    precision=4,
                ),
                self.measurement(
                    key="scene_wide_index_shift",
                    label=f"Median {state.index_name} shift everywhere else",
                    value=scene_shift,
                    unit="index",
                    formula=(
                        f"median({state.index_name} on {state.role_b.value} - on "
                        f"{state.role_a.value}) over observable pixels outside the "
                        "detected region"
                        + (
                            "; the detection covers the whole observable scene, so "
                            "this is the shift inside it"
                            if outside.size < 32
                            else ""
                        )
                    ),
                    inputs={
                        "pixels": int(outside.size),
                        "whole_scene_detected": bool(outside.size < 32),
                    },
                    method=(
                        "real localised change leaves the rest of the scene alone; "
                        "phenology moves all of it"
                    ),
                    applies_to=[state.role_a, state.role_b],
                    precision=4,
                ),
                self.measurement(
                    key="scene_wide_share",
                    label="Share of the shift that is scene-wide",
                    value=share,
                    unit="ratio",
                    formula=(
                        f"|{scene_shift:.4f}| / |{region_shift:.4f}| = {share:.3f}"
                    ),
                    inputs={
                        "likely_above": SCENE_WIDE_SHARE_LIKELY,
                        "plausible_above": SCENE_WIDE_SHARE_PLAUSIBLE,
                        "negligible_shift_below": SCENE_WIDE_NEGLIGIBLE,
                    },
                    applies_to=[state.role_a, state.role_b],
                    precision=3,
                ),
                self.measurement(
                    key="changed_extent_of_scene",
                    label="Share of the observable scene flagged as changed",
                    value=extent,
                    unit="fraction",
                    formula=(
                        f"{inside.size} detected px / "
                        f"{inside.size + outside.size} observable px"
                    ),
                    inputs={
                        "likely_above": SCENE_WIDE_EXTENT_LIKELY,
                        "comparison_becomes_circular_above": (
                            SCENE_WIDE_EXTENT_CIRCULAR
                        ),
                    },
                    method=(
                        "when most of the scene is flagged, the inside-against-"
                        "outside contrast no longer discriminates and extent decides"
                    ),
                    applies_to=[state.role_a, state.role_b],
                    precision=3,
                ),
            ]
        )

        calendar_risk = month_delta is not None and month_delta >= SEASONAL_SAFE_MONTHS
        scene_moved = abs(scene_shift) >= SCENE_WIDE_NEGLIGIBLE

        if calendar_risk and extent >= SCENE_WIDE_EXTENT_LIKELY:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"{extent:.0%} of the observable scene is flagged as changed, and the "
                f"acquisitions are {month_delta} month(s) apart in the annual cycle. "
                "A change of that extent is not a change to one region: land cover is "
                "not repainted across most of a landscape in a season. This is what a "
                f"growing cycle looks like. The inside-against-outside contrast reads "
                f"{share:.0%} here, but at this extent that comparison is circular, "
                "because everything that did not move is what defines the outside."
            )
        elif scene_moved and share >= SCENE_WIDE_SHARE_LIKELY:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"{state.index_name} shifted by {scene_shift:+.3f} across the rest of "
                f"the scene against {region_shift:+.3f} inside the detected region, "
                f"which is {share:.0%} as much. A change that is happening everywhere "
                "is not a change in this region."
            )
        elif calendar_risk and scene_moved and share >= SCENE_WIDE_SHARE_PLAUSIBLE:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"The acquisitions are {month_delta} month(s) apart in the annual "
                f"cycle and the rest of the scene shifted by {scene_shift:+.3f}, "
                f"{share:.0%} of the shift inside the region. Part of what was "
                "detected may be phenology."
            )
        elif calendar_risk:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"The acquisitions are {month_delta} month(s) apart in the annual "
                "cycle, so crops and canopies need not be in the same state, even "
                f"though the rest of the scene only shifted by {scene_shift:+.3f}."
            )
        else:
            verdict = ConfounderVerdict.RULED_OUT
            explanation = (
                f"The acquisitions sit "
                f"{month_delta if month_delta is not None else 0} month(s) apart in "
                f"the annual cycle and {state.index_name} shifted by only "
                f"{scene_shift:+.3f} outside the detected region, so the detected "
                "change is localised rather than seasonal."
            )

        requirement = None
        if verdict is not ConfounderVerdict.RULED_OUT:
            requirement = (
                "An acquisition from the same part of the growing season as "
                f"{state.role_a.value}"
                + (f", which was {day_delta} days before {state.role_b.value}" if day_delta else "")
                + ". Comparing like with like removes phenology as an explanation."
            )

        return ConfounderTest(
            kind=ConfounderKind.SEASONALITY,
            label=CONFOUNDER_LABEL[ConfounderKind.SEASONALITY],
            question=CONFOUNDER_QUESTION[ConfounderKind.SEASONALITY],
            verdict=verdict,
            measured=(
                f"{extent:.0%} of the scene flagged, {share:.0%} of the shift "
                f"scene-wide ({scene_shift:+.3f} outside against "
                f"{region_shift:+.3f} inside)"
            ),
            measured_numeric=extent if calendar_risk else share,
            unit="fraction" if calendar_risk else "ratio",
            threshold=(
                f"likely when over {SCENE_WIDE_EXTENT_LIKELY:.0%} of the scene is "
                f"flagged and the dates are {SEASONAL_SAFE_MONTHS}+ months apart in "
                f"the annual cycle, or when over {SCENE_WIDE_SHARE_LIKELY:.0%} of the "
                f"shift is scene-wide; plausible above "
                f"{SCENE_WIDE_SHARE_PLAUSIBLE:.0%}"
            ),
            formula=(
                "detected extent / observable extent, and |median shift outside the "
                "region| / |median shift inside it|"
            ),
            inputs={
                "scene_wide_shift": round(scene_shift, 6),
                "region_shift": round(region_shift, 6),
                "changed_extent_of_scene": round(extent, 6),
                "scene_wide_share": round(share, 6),
                "month_of_year_delta": month_delta,
                "day_delta": day_delta,
                "index": state.index_name,
            },
            method=(
                "compares the index shift inside the detected region against the "
                "shift over the rest of the observable scene, and falls back to the "
                "extent of the detection when that comparison would be circular"
            ),
            explanation=explanation,
            requirement=requirement,
            applies_to=[state.role_a, state.role_b],
        )

    # -- misregistration ---------------------------------------------------
    def _test_misregistration(self, state: _State) -> ConfounderTest:
        """Does the detected change hug edges?

        A shifted image produces change along every boundary in the scene and
        almost nowhere else, so edge concentration separates a misalignment
        artifact from a real region that happens to have edges.
        """
        context = state.context
        report = context.readiness
        shift_px = None
        if report is not None:
            check = next(
                (c for c in report.checks if c.id == "pair.coregistration"), None
            )
            if check is not None:
                shift_px = check.measured_numeric

        if not state.detected.any():
            raise ToolError("No change was detected, so there is no pattern to test.")

        edges = self._edge_mask(state)
        usable = ~state.invalid

        # Both the share and the baseline are measured against the *same* dilated
        # edge mask. Measuring the change's proximity to a dilated edge and
        # comparing it against the undilated edge density inflates the ratio by
        # the dilation factor alone: on a featureless scene the top decile of
        # gradient is scattered noise, and dilating it covers half the image, so a
        # perfectly real change scores five times "chance" against a baseline of
        # 10 per cent. The comparison is only meaningful like against like.
        from scipy.ndimage import binary_dilation

        near_edge = binary_dilation(edges, structure=np.ones((3, 3), dtype=bool))

        observable = int(np.count_nonzero(usable))
        detected_observable = int(np.count_nonzero(state.detected & usable))
        edge_density = (
            int(np.count_nonzero(near_edge & usable)) / observable
            if observable
            else 0.0
        )
        edge_share = (
            int(np.count_nonzero(state.detected & near_edge & usable))
            / detected_observable
            if detected_observable
            else 0.0
        )
        enrichment = edge_share / edge_density if edge_density > 0 else 0.0

        state.outcome.measurements.append(
            self.measurement(
                key="edge_enrichment",
                label="Detected change concentrated on edges",
                value=enrichment,
                unit="ratio",
                formula=(
                    f"{edge_share:.3f} of detected change lies near an edge / "
                    f"{edge_density:.3f} of the observable scene lies near an edge"
                ),
                inputs={
                    "edge_percentile": EDGE_PERCENTILE,
                    "edge_density": round(edge_density, 6),
                    "edge_share_of_change": round(edge_share, 6),
                    "measured_shift_px": shift_px,
                    "likely_above": EDGE_ENRICHMENT_LIKELY,
                },
                method=(
                    "Sobel gradient magnitude on the structure band of the first "
                    "date, thresholded at its 90th percentile, then dilated by one "
                    "pixel; the same dilated mask sets both the share and the "
                    "baseline so the ratio is not inflated by the dilation"
                ),
                applies_to=[state.role_a, state.role_b],
                precision=2,
            )
        )

        misaligned = shift_px is not None and shift_px > 1.0

        if enrichment >= EDGE_ENRICHMENT_LIKELY and misaligned:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"The images are offset by {shift_px:.2f} px and the detected change "
                f"sits on edges {enrichment:.1f} times more than chance would put it "
                "there. This is what a shifted image looks like, not a changed one."
            )
        elif enrichment >= EDGE_ENRICHMENT_LIKELY:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"The detected change sits on edges {enrichment:.1f} times more than "
                "chance, although alignment measured at "
                f"{shift_px if shift_px is not None else 0:.2f} px. Narrow linear "
                "features change along their edges too, so this is suggestive rather "
                "than conclusive."
            )
        elif enrichment >= EDGE_ENRICHMENT_PLAUSIBLE and misaligned:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"Alignment is off by {shift_px:.2f} px and edge concentration is "
                f"{enrichment:.1f} times chance, enough that some of the detected "
                "change may be an alignment artifact."
            )
        else:
            verdict = ConfounderVerdict.RULED_OUT
            explanation = (
                f"The detected change is spread across the scene rather than following "
                f"edges ({enrichment:.1f} times chance"
                + (
                    f", with alignment within {shift_px:.2f} px"
                    if shift_px is not None
                    else ""
                )
                + "), so it is not an alignment artifact."
            )

        return ConfounderTest(
            kind=ConfounderKind.MISREGISTRATION,
            label=CONFOUNDER_LABEL[ConfounderKind.MISREGISTRATION],
            question=CONFOUNDER_QUESTION[ConfounderKind.MISREGISTRATION],
            verdict=verdict,
            measured=f"{enrichment:.1f}x edge concentration",
            measured_numeric=enrichment,
            unit="ratio",
            threshold=(
                f"likely above {EDGE_ENRICHMENT_LIKELY:.1f}x with an offset over 1 px, "
                f"plausible above {EDGE_ENRICHMENT_PLAUSIBLE:.1f}x"
            ),
            formula="edge share of detected change / edge density of the scene",
            inputs={
                "edge_share_of_change": round(edge_share, 6),
                "edge_density": round(edge_density, 6),
                "measured_shift_px": shift_px,
            },
            method="Sobel edges from the first date, 90th percentile",
            explanation=explanation,
            requirement=(
                None
                if verdict is ConfounderVerdict.RULED_OUT
                else "A co-registered pair, or ground control points to align these two"
            ),
            applies_to=[state.role_a, state.role_b],
        )

    def _edge_mask(self, state: _State) -> np.ndarray:
        from scipy.ndimage import sobel

        context = state.context
        meta = context.metadata(state.role_a)
        index = structure_band_index(meta)
        role = next(
            (
                candidate
                for candidate in meta.resolved_roles
                if meta.band_index(candidate) == index
            ),
            None,
        )
        band = None
        if role is not None:
            band = read_role_reflectance(
                context.path(state.role_a), meta, role, **state.grid.read_kwargs()
            )
        if band is None:
            raise ToolError("No band could be read to locate edges.")

        filled = np.ma.filled(band, 0.0).astype("float64")
        magnitude = np.hypot(sobel(filled, axis=0), sobel(filled, axis=1))
        usable = magnitude[~state.invalid]
        if usable.size < 32:
            raise ToolError("Too few observable pixels to locate edges.")
        cutoff = float(np.percentile(usable, EDGE_PERCENTILE))
        return magnitude >= cutoff

    # -- cloud fringe ------------------------------------------------------
    def _test_cloud_fringe(self, state: _State) -> ConfounderTest:
        """Does the detected change cling to the edge of what was masked out?

        Cloud itself is excluded from the measurement. Its fringe is not: thin
        cirrus and the penumbra of a shadow are not sharply bounded, and change
        detected right against the mask boundary is more likely to be that fringe
        than real ground change.
        """
        if not state.invalid.any():
            return ConfounderTest(
                kind=ConfounderKind.CLOUD_SHADOW,
                label=CONFOUNDER_LABEL[ConfounderKind.CLOUD_SHADOW],
                question=CONFOUNDER_QUESTION[ConfounderKind.CLOUD_SHADOW],
                verdict=ConfounderVerdict.RULED_OUT,
                measured="no pixel was excluded on either date",
                measured_numeric=0.0,
                unit="fraction",
                threshold="any excluded region would have to be adjacent to the finding",
                formula="count of excluded pixels = 0",
                method="scene classification layer, cloud and shadow classes",
                explanation=(
                    "Both dates were clear over the whole overlap, so neither cloud "
                    "nor shadow can account for the finding."
                ),
                applies_to=[state.role_a, state.role_b],
            )

        if not state.detected.any():
            raise ToolError("No change was detected, so there is no pattern to test.")

        share = adjacency_fraction(
            state.detected, state.invalid, within_pixels=CLOUD_FRINGE_PIXELS
        )
        _, excluded_km2 = mask_area_km2(state.invalid, state.grid.transform)

        state.outcome.measurements.append(
            self.measurement(
                key="cloud_fringe_share",
                label="Detected change adjacent to excluded pixels",
                value=share,
                unit="fraction",
                formula=(
                    f"detected pixels within {CLOUD_FRINGE_PIXELS} px of an excluded "
                    "pixel / detected pixels"
                ),
                inputs={
                    "fringe_pixels": CLOUD_FRINGE_PIXELS,
                    "excluded_km2": round(excluded_km2 or 0.0, 6),
                    "likely_above": CLOUD_FRINGE_SHARE_LIKELY,
                },
                method="adjacency to the union of both dates' unobservable pixels",
                applies_to=[state.role_a, state.role_b],
                precision=3,
            )
        )

        if share >= CLOUD_FRINGE_SHARE_LIKELY:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"{share:.0%} of the detected change lies within "
                f"{CLOUD_FRINGE_PIXELS} px of a pixel that was excluded as cloud or "
                "shadow. Thin cloud and shadow penumbra are not sharply bounded, so "
                "this is most likely the fringe of what was masked rather than ground "
                "change."
            )
        elif share >= CLOUD_FRINGE_SHARE_PLAUSIBLE:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"{share:.0%} of the detected change sits against the excluded region, "
                "so part of it may be cloud or shadow fringe."
            )
        else:
            verdict = ConfounderVerdict.RULED_OUT
            explanation = (
                f"Only {share:.0%} of the detected change is near an excluded pixel, "
                f"and {excluded_km2 or 0.0:.2f} km2 was excluded outright, so cloud "
                "and shadow do not account for the finding."
            )

        return ConfounderTest(
            kind=ConfounderKind.CLOUD_SHADOW,
            label=CONFOUNDER_LABEL[ConfounderKind.CLOUD_SHADOW],
            question=CONFOUNDER_QUESTION[ConfounderKind.CLOUD_SHADOW],
            verdict=verdict,
            measured=f"{share:.0%} of the finding is adjacent to excluded pixels",
            measured_numeric=share,
            unit="fraction",
            threshold=(
                f"likely above {CLOUD_FRINGE_SHARE_LIKELY:.0%}, plausible above "
                f"{CLOUD_FRINGE_SHARE_PLAUSIBLE:.0%}"
            ),
            formula=(
                f"detected pixels within {CLOUD_FRINGE_PIXELS} px of an excluded "
                "pixel / detected pixels"
            ),
            inputs={"excluded_km2": round(excluded_km2 or 0.0, 6)},
            method="scene classification layer, cloud and shadow classes",
            explanation=explanation,
            requirement=(
                None
                if verdict is ConfounderVerdict.RULED_OUT
                else "A cloud-free acquisition of either date, or a SAR acquisition "
                "that sees through cloud"
            ),
            applies_to=[state.role_a, state.role_b],
        )

    # -- radiometry --------------------------------------------------------
    def _test_radiometry(self, state: _State) -> ConfounderTest:
        """Did the invariant part of the scene change brightness?

        Bright hard surfaces, roofs, rock, and pavement, do not change over a few
        years. If their reflectance moved between the two acquisitions, the
        difference is in the calibration or the atmosphere rather than the ground,
        and every difference measured from these images inherits that offset.
        """
        context = state.context
        role_for_band = BandRole.RED
        kwargs = state.grid.read_kwargs()
        first = read_role_reflectance(
            context.path(state.role_a),
            context.metadata(state.role_a),
            role_for_band,
            **kwargs,
        )
        second = read_role_reflectance(
            context.path(state.role_b),
            context.metadata(state.role_b),
            role_for_band,
            **kwargs,
        )
        if first is None or second is None:
            raise ToolError(
                "The red band is absent on at least one date, so radiometric drift "
                "cannot be checked against a pseudo-invariant surface."
            )

        usable = ~state.invalid & ~state.detected
        values = np.ma.filled(first, np.nan)
        candidates = values[usable & np.isfinite(values)]
        if candidates.size < 64:
            raise ToolError(
                "Too few unchanged observable pixels to identify a pseudo-invariant "
                "surface."
            )

        cutoff = float(np.percentile(candidates, PIF_PERCENTILE))
        pif = usable & (np.ma.filled(first, -1.0) >= cutoff)
        if int(np.count_nonzero(pif)) < 64:
            raise ToolError("The pseudo-invariant surface sample is too small.")

        before = np.ma.filled(first, np.nan)[pif]
        after = np.ma.filled(second, np.nan)[pif]
        finite = np.isfinite(before) & np.isfinite(after)
        if int(np.count_nonzero(finite)) < 64:
            raise ToolError("The pseudo-invariant surface sample is too small.")

        shift = float(np.median(after[finite] - before[finite]))

        state.outcome.measurements.append(
            self.measurement(
                key="pseudo_invariant_shift",
                label="Reflectance shift on surfaces that should not have changed",
                value=shift,
                unit="reflectance",
                formula=(
                    f"median(red on {state.role_b.value} - red on "
                    f"{state.role_a.value}) over the brightest "
                    f"{100 - PIF_PERCENTILE:.0f}% of unchanged pixels"
                ),
                inputs={
                    "sample_pixels": int(np.count_nonzero(finite)),
                    "brightness_percentile": PIF_PERCENTILE,
                    "likely_above": PIF_SHIFT_LIKELY,
                    "plausible_above": PIF_SHIFT_PLAUSIBLE,
                },
                method=(
                    "pseudo-invariant feature comparison: bright hard surfaces are "
                    "stable over the interval, so a shift in their reflectance is "
                    "instrumental or atmospheric"
                ),
                applies_to=[state.role_a, state.role_b],
                precision=4,
            )
        )

        magnitude = abs(shift)
        if magnitude >= PIF_SHIFT_LIKELY:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"Surfaces that should not have changed are {shift:+.3f} reflectance "
                f"different between the two dates. An offset that large affects every "
                "difference measured from this pair, so the finding may be an artifact "
                "of calibration or atmosphere."
            )
        elif magnitude >= PIF_SHIFT_PLAUSIBLE:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"Unchanged bright surfaces differ by {shift:+.3f} reflectance, enough "
                "to shift a threshold-based comparison at the margins."
            )
        else:
            verdict = ConfounderVerdict.RULED_OUT
            explanation = (
                f"Unchanged bright surfaces read within {magnitude:.3f} reflectance of "
                "each other across the two dates, so the acquisitions are "
                "radiometrically comparable and the finding is not a calibration "
                "offset."
            )

        return ConfounderTest(
            kind=ConfounderKind.RADIOMETRY,
            label=CONFOUNDER_LABEL[ConfounderKind.RADIOMETRY],
            question=CONFOUNDER_QUESTION[ConfounderKind.RADIOMETRY],
            verdict=verdict,
            measured=f"{shift:+.3f} reflectance on invariant surfaces",
            measured_numeric=shift,
            unit="reflectance",
            threshold=(
                f"likely above {PIF_SHIFT_LIKELY:.02f}, plausible above "
                f"{PIF_SHIFT_PLAUSIBLE:.02f}"
            ),
            formula="median red reflectance difference over invariant bright pixels",
            inputs={"sample_pixels": int(np.count_nonzero(finite))},
            method="pseudo-invariant feature comparison",
            explanation=explanation,
            requirement=(
                None
                if verdict is ConfounderVerdict.RULED_OUT
                else "Surface reflectance products processed with the same atmospheric "
                "correction, or an overlapping acquisition to cross-calibrate against"
            ),
            applies_to=[state.role_a, state.role_b],
        )

    # -- sensor mismatch ---------------------------------------------------
    def _test_sensor_mismatch(self, state: _State) -> ConfounderTest:
        context = state.context
        meta_a = context.metadata(state.role_a)
        meta_b = context.metadata(state.role_b)
        gsd_a = meta_a.geo.gsd_m
        gsd_b = meta_b.geo.gsd_m
        if not gsd_a or not gsd_b:
            raise ToolError("At least one date has no resolvable ground sample distance.")

        ratio = max(gsd_a, gsd_b) / min(gsd_a, gsd_b)
        shared = sorted(
            {role.value for role in meta_a.resolved_roles}
            & {role.value for role in meta_b.resolved_roles}
        )

        if ratio >= GSD_RATIO_LIKELY:
            verdict = ConfounderVerdict.LIKELY
            explanation = (
                f"The two dates differ in resolution by {ratio:.1f} times "
                f"({gsd_a:.1f} m against {gsd_b:.1f} m). At that gap the coarser "
                "image cannot resolve features the finer one can, and the difference "
                "between them is partly a difference in what each can see."
            )
        elif ratio >= GSD_RATIO_PLAUSIBLE:
            verdict = ConfounderVerdict.PLAUSIBLE
            explanation = (
                f"Resolutions differ by {ratio:.1f} times ({gsd_a:.1f} m against "
                f"{gsd_b:.1f} m), so small features are represented differently on "
                "the two dates."
            )
        else:
            verdict = ConfounderVerdict.RULED_OUT
            explanation = (
                f"Both dates are at effectively the same resolution "
                f"({gsd_a:.1f} m and {gsd_b:.1f} m) and share "
                f"{len(shared)} band role(s), so the comparison is between like and "
                "like."
            )

        return ConfounderTest(
            kind=ConfounderKind.SENSOR_MISMATCH,
            label=CONFOUNDER_LABEL[ConfounderKind.SENSOR_MISMATCH],
            question=CONFOUNDER_QUESTION[ConfounderKind.SENSOR_MISMATCH],
            verdict=verdict,
            measured=f"{ratio:.2f}x resolution ratio, {len(shared)} shared band role(s)",
            measured_numeric=ratio,
            unit="ratio",
            threshold=(
                f"likely above {GSD_RATIO_LIKELY:.1f}x, plausible above "
                f"{GSD_RATIO_PLAUSIBLE:.1f}x"
            ),
            formula=f"max({gsd_a:.2f}, {gsd_b:.2f}) / min({gsd_a:.2f}, {gsd_b:.2f})",
            inputs={
                "gsd_a_m": gsd_a,
                "gsd_b_m": gsd_b,
                "shared_band_roles": shared,
                "gsd_method_a": meta_a.geo.gsd_method,
                "gsd_method_b": meta_b.geo.gsd_method,
            },
            method="ground sample distance as measured from each file's transform",
            explanation=explanation,
            requirement=(
                None
                if verdict is ConfounderVerdict.RULED_OUT
                else "A pair from the same sensor, or from sensors of comparable "
                "resolution"
            ),
            applies_to=[state.role_a, state.role_b],
        )


@dataclass
class _Finding:
    """The result under test, however it was produced.

    Gathered into one shape so the individual tests do not have to know which
    tool found the thing they are trying to explain away.
    """

    grid: Any
    delta: np.ndarray
    invalid: np.ndarray
    detected: np.ndarray
    index_name: str
    role_a: ImageRole
    role_b: ImageRole
    note: str = ""
    # False when there is no index difference to work with, which rules out the
    # tests that compare a shift inside the finding against one outside it.
    has_delta: bool = True


def _finding_source(context: ToolContext) -> _Finding | None:
    """Find something to test, preferring a change measurement.

    The change engine is the usual source. Where there is no change to measure,
    as on a cross-modal pair, the fusion engine's disagreement is the finding
    instead: asking whether cloud or a resolution gap explains why two sensors
    differ is exactly the same kind of question.
    """
    change = context.upstream.get("change-cva-engine")
    if change is not None and change.ok:
        artifacts = change.artifacts
        required = ("change.grid", "change.delta", "change.invalid", "change.index")
        if all(key in artifacts for key in required):
            roles = context.temporal_roles()
            if roles is not None:
                grid = artifacts["change.grid"]
                detected = np.zeros(grid.shape, dtype=bool)
                for key in ("change_gain", "change_loss"):
                    layer = change.mask(key)
                    if layer is not None:
                        detected |= layer.array
                return _Finding(
                    grid=grid,
                    delta=artifacts["change.delta"],
                    invalid=artifacts["change.invalid"],
                    detected=detected,
                    index_name=artifacts["change.index"],
                    role_a=roles[0],
                    role_b=roles[1],
                    note="the change measured between the two dates",
                )

    fusion = context.upstream.get("optical-sar-fusion")
    if fusion is not None and fusion.ok:
        disagreement = np.zeros((0, 0), dtype=bool)
        reference = None
        for key in ("fusion_disagree_sar_only", "fusion_disagree_optical_only"):
            layer = fusion.mask(key)
            if layer is None:
                continue
            reference = reference or layer
            if disagreement.size == 0:
                disagreement = np.zeros_like(layer.array)
            disagreement |= layer.array
        if reference is not None:
            optical = next(iter(context.optical_roles()), None)
            radar = next(iter(context.sar_roles()), None)
            if optical is not None and radar is not None:
                invalid = (
                    reference.invalid
                    if reference.invalid is not None
                    else np.zeros_like(reference.array)
                )
                return _Finding(
                    grid=_GridFromLayer(reference),
                    # No index difference exists across two sensors: reflectance
                    # and backscatter are not the same quantity, so subtracting
                    # them would produce a number with no meaning.
                    delta=np.zeros_like(reference.array, dtype="float64"),
                    invalid=invalid,
                    detected=disagreement,
                    index_name="cross-sensor",
                    role_a=optical,
                    role_b=radar,
                    note="where the two sensors disagree",
                    has_delta=False,
                )

    return None


class _GridFromLayer:
    """Grid view of a mask layer, matching what the change engine hands over."""

    def __init__(self, layer: Any) -> None:
        self.transform = layer.transform
        self.crs = layer.crs
        self.height, self.width = layer.array.shape
        self.pixel_m = abs(layer.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def read_kwargs(self) -> dict[str, Any]:
        # The masks are already on the source grid, so no windowing is needed.
        return {}


class _State:
    """Everything the individual tests share, gathered once."""

    def __init__(
        self,
        *,
        context: ToolContext,
        grid: Any,
        delta: np.ndarray,
        invalid: np.ndarray,
        detected: np.ndarray,
        index_name: str,
        role_a: ImageRole,
        role_b: ImageRole,
        outcome: ToolOutcome,
        has_delta: bool = True,
    ) -> None:
        self.context = context
        self.grid = grid
        self.delta = delta
        self.invalid = invalid
        self.detected = detected
        self.index_name = index_name
        self.role_a = role_a
        self.role_b = role_b
        self.outcome = outcome
        self.has_delta = has_delta


__all__ = [
    "EDGE_ENRICHMENT_LIKELY",
    "PIF_SHIFT_LIKELY",
    "SCENE_WIDE_SHARE_LIKELY",
    "SEASONAL_SAFE_MONTHS",
    "ConfounderEngine",
]
