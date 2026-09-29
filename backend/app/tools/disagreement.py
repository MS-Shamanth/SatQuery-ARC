"""The disagreement engine: where the methods conflict, and why.

Every method already in this build answers the same kind of question about the same
pixels, by a different route. The spectral indices threshold a published formula.
The grounding adapter resolves a named target to one of those indices and drops
small regions. The change engine differences two dates. The radar engine thresholds
backscatter. The learned probe weighs every band at once. On most ground they
agree, and where they do not, that is the single most informative thing the system
can show.

So this engine does not measure the scene. It compares what the other tools already
measured, pixel by pixel, and reports three states: the methods agree, the methods
disagree, or no method was confident enough for its answer to count. The
disagreement is then broken into contiguous regions, each carrying which method
said what, the measured reason they differ, and a recommended action.

Three rules keep it honest.

**A method that could not see is not dissenting.** Cloud, nodata and out-of-swath
pixels are excluded from the comparison rather than counted as disagreement, the
same rule the optical-versus-radar fusion follows.

**No winner is declared.** There is no labelled reference for these scenes, so
scoring the methods against each other would be inventing ground truth. The
recommendation is human review, which is the honest action.

**The reason is measured.** "Low spectral separability" is only written where the
separability was actually computed and is actually low. Where no measurement
explains the conflict, the engine says the reason is not established.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.disagreement import (
    Agreement,
    ConflictRegion,
    DisagreementReport,
    MethodOpinion,
)
from app.models.schemas import ImageRole, ToolImplementation, ToolRequirement
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import connected_components, mask_area_km2
from app.tools.indices import WEAK_SEPARABILITY

logger = logging.getLogger(__name__)

# Regions below this are not worth a reviewer's attention: at ten metres this is a
# couple of dozen pixels, which is the scale at which methods differ by rounding
# rather than by substance.
MIN_CONFLICT_AREA_M2 = 5000.0

# How many regions to report. The list is a work queue for a human, and one that
# cannot be worked through is not a work queue.
MAX_REGIONS = 40

# A probability at or below this is not a position. Below it the learned probe's
# preference is too weak to count either as agreement or as dissent.
UNCERTAIN_CONFIDENCE = 0.55

# Mask keys each method contributes, mapped to the class the mask asserts. Only
# masks that make a claim about a land-cover class belong here: a change-magnitude
# layer says something changed, not what it is.
CLASS_OF_MASK: tuple[tuple[str, str], ...] = (
    ("grounded_", "target"),
    ("probe_water", "water"),
    ("probe_vegetation", "vegetation"),
    ("probe_built_up", "built-up"),
    ("probe_bare", "bare"),
    ("mndwi_mask", "water"),
    ("ndwi_mask", "water"),
    ("ndvi_mask", "vegetation"),
    ("ndbi_mask", "built-up"),
    ("sar_water", "water"),
    ("fusion_agree_water", "water"),
)

# How each tool is named to the reader. The point of the panel is that a person can
# see which kind of evidence dissented, so the names are the kinds of evidence.
METHOD_LABEL: dict[str, str] = {
    "spectral-index-engine": "Spectral index",
    "grounding-engine": "Grounding",
    "sar-backscatter-engine": "Radar backscatter",
    "optical-sar-fusion": "Optical and radar fused",
    "change-cva-engine": "Change vector",
    "rs-landcover-probe": "Learned probe",
}


@dataclass
class _Opinion:
    """One method's mask, with everything needed to describe and compare it."""

    method: str
    tool: str
    version: str
    verdict: str
    basis: str
    selected: np.ndarray
    invalid: np.ndarray | None
    confidence: np.ndarray | None
    separability: float | None


def _class_for(key: str, targets: list[str]) -> str | None:
    for prefix, class_name in CLASS_OF_MASK:
        if key.startswith(prefix):
            if class_name == "target":
                return targets[0].lower() if targets else "the named target"
            return class_name
    return None


class DisagreementEngine(BaseTool):
    """Compares what every other method concluded about the same pixels."""

    name = "evidence-disagreement-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Compares the conclusions of every method that ran on this scene, pixel by "
        "pixel, and reports where they agree, where they conflict, and why. Declares "
        "no winner: without a labelled reference, the honest output of a conflict is "
        "a region for a human to review."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            description=(
                "At least two other methods that each produced a mask asserting a "
                "land-cover class on the same grid."
            ),
        )

    def depends_on(self) -> tuple[str, ...]:
        return (
            "spectral-index-engine",
            "grounding-engine",
            "sar-backscatter-engine",
            "rs-landcover-probe",
        )

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]:
        if len(self._gather(context)) < 2:
            return False, (
                "fewer than two methods produced a class mask on this scene, so "
                "there is nothing to compare"
            )
        return True, ""

    def produces(self) -> list[str]:
        return [
            "conflicting_area_km2",
            "conflicting_fraction",
            "agreeing_area_km2",
            "uncertain_area_km2",
            "methods_compared",
            "conflict_regions",
        ]

    @staticmethod
    def _label(tool: str, mask_key: str) -> str:
        """How this opinion is named to the reader."""
        base = METHOD_LABEL.get(tool, tool)
        if tool != "spectral-index-engine":
            return base
        index = mask_key.split("_", 1)[0].upper()
        return f"{base} ({index})"

    # -- gathering ---------------------------------------------------------
    def _gather(self, context: ToolContext) -> list[_Opinion]:
        """Every upstream mask that asserts a land-cover class, on one grid.

        Masks of different shapes cannot be compared pixel by pixel, so the most
        common shape wins and the rest are dropped with a note. Silently
        comparing mismatched grids would produce a disagreement map of the
        resampling rather than of the evidence.
        """
        found: list[_Opinion] = []
        targets = list(context.target_classes)

        for tool, outcome in context.upstream.items():
            if not getattr(outcome, "ok", False) or tool == self.name:
                continue
            for layer in getattr(outcome, "masks", []):
                verdict = _class_for(layer.key, targets)
                if verdict is None:
                    continue
                found.append(
                    _Opinion(
                        # Qualified by the mask, because one tool can hold two
                        # opinions about the same class: MNDWI and NDWI are both
                        # water indices and they disagree over cities, which is one
                        # of the more interesting conflicts this panel can show.
                        method=self._label(tool, layer.key),
                        tool=tool,
                        version=outcome.version,
                        verdict=verdict,
                        basis=layer.description or layer.label,
                        selected=np.asarray(layer.array, dtype=bool),
                        invalid=(
                            None
                            if layer.invalid is None
                            else np.asarray(layer.invalid, dtype=bool)
                        ),
                        confidence=None,
                        separability=layer.separability,
                    )
                )

        if not found:
            return []

        shapes = [opinion.selected.shape for opinion in found]
        common = max(set(shapes), key=shapes.count)
        return [
            opinion for opinion in found if opinion.selected.shape == common
        ]

    # -- the comparison ----------------------------------------------------
    def execute(self, context: ToolContext) -> ToolOutcome:
        opinions = self._gather(context)
        if len(opinions) < 2:
            raise ToolError(
                "Fewer than two methods produced a comparable class mask, so there "
                "is no disagreement to report."
            )

        # Methods are only ever compared about the same class.
        #
        # This is the whole correctness of the panel. A first version pooled every
        # mask together, so the vegetation index "disagreeing" with the water index
        # counted as a conflict and the answer came out as 100% of the scene
        # contested, which is true of no scene and useful for nothing. Two methods
        # conflict when one says a pixel is water and another says that same pixel
        # is not water. An index that was measuring vegetation has not been asked.
        by_class: dict[str, list[_Opinion]] = {}
        for opinion in opinions:
            by_class.setdefault(opinion.verdict, []).append(opinion)

        contested_classes = {
            name: group for name, group in by_class.items() if len(group) >= 2
        }
        if not contested_classes:
            raise ToolError(
                "No class was assessed by two different methods, so there is no "
                "conflict to report. Compared: "
                + ", ".join(f"{name} by {len(g)}" for name, g in by_class.items())
            )

        role = context.primary_role()
        transform, crs = self._geometry(context, role)
        shape = opinions[0].selected.shape

        unseen = np.zeros(shape, dtype=bool)
        for opinion in opinions:
            if opinion.invalid is not None:
                unseen |= opinion.invalid

        candidate = np.zeros(shape, dtype=bool)
        agree = np.zeros(shape, dtype=bool)
        disagree = np.zeros(shape, dtype=bool)
        uncertain = np.zeros(shape, dtype=bool)

        for group in contested_classes.values():
            votes = np.zeros(shape, dtype="int16")
            for opinion in group:
                votes += opinion.selected.astype("int16")

            proposed = (votes > 0) & ~unseen
            split = proposed & (votes < len(group))
            unanimous = proposed & (votes == len(group))

            # Contested ground where the split is explained by a method's own
            # weakness rather than by a genuine difference of opinion. An index
            # that does not separate this scene is not really taking a position, so
            # its dissent is reported as uncertainty instead of conflict.
            weak = np.zeros(shape, dtype=bool)
            for opinion in group:
                if (
                    opinion.separability is not None
                    and opinion.separability < WEAK_SEPARABILITY
                ):
                    weak |= split & opinion.selected

            candidate |= proposed
            agree |= unanimous
            disagree |= split & ~weak
            uncertain |= weak

        # A pixel contested for any class is contested, whatever another class
        # agreed about it. Precedence runs conflict, then uncertainty, then
        # agreement, so the map never shows green over ground that is in dispute.
        uncertain &= ~disagree
        agree &= ~(disagree | uncertain)

        outcome = self.outcome(
            parameters={
                "methods": [opinion.method for opinion in opinions],
                "min_conflict_area_m2": MIN_CONFLICT_AREA_M2,
            }
        )

        _, candidate_km2 = mask_area_km2(candidate, transform)
        _, agree_km2 = mask_area_km2(agree, transform)
        _, disagree_km2 = mask_area_km2(disagree, transform)
        _, uncertain_km2 = mask_area_km2(uncertain, transform)
        candidate_km2 = candidate_km2 or 0.0
        contested = (disagree_km2 or 0.0) + (uncertain_km2 or 0.0)
        fraction = (contested / candidate_km2) if candidate_km2 else 0.0

        report = DisagreementReport(
            candidate_area_km2=candidate_km2,
            agree_area_km2=agree_km2 or 0.0,
            disagree_area_km2=disagree_km2 or 0.0,
            uncertain_area_km2=uncertain_km2 or 0.0,
            conflicting_fraction=fraction,
            # Only the methods that actually took part in a comparison.
            #
            # Not every gathered mask does. A vegetation index on a scene where no
            # second method assessed vegetation was never contradicted by anything,
            # so counting it would inflate "across N methods" and the
            # methods_compared measurement with a method that compared nothing.
            #
            # Named once each: several masks from one tool are one method as far as
            # the reader is concerned, and listing "Spectral index" four times says
            # nothing except that four indices were computed.
            methods=sorted(
                {
                    opinion.method
                    for group in contested_classes.values()
                    for opinion in group
                }
            ),
            applies_to=[role],
            agree_layer="evidence_agree",
            disagree_layer="evidence_disagree",
            uncertain_layer="evidence_uncertain",
        )

        self._describe_regions(
            report, disagree, uncertain, contested_classes, transform, crs
        )
        self._add_layers(outcome, agree, disagree, uncertain, unseen, transform, crs, role)
        self._add_measurements(
            outcome, report, candidate, agree, disagree, uncertain, transform, role
        )

        report.notes.append(
            "A method that could not see a pixel is excluded from the comparison "
            "rather than counted as disagreeing. Cloud and missing data are not "
            "dissent."
        )
        report.notes.append(
            "No method is treated as correct. There is no labelled reference for "
            "this scene, so a conflict is reported as ground for review rather than "
            "resolved by preferring one instrument."
        )
        if len(by_class) > 1:
            report.notes.append(
                "Methods are compared only where they describe the same class: "
                + ", ".join(sorted(by_class))
                + "."
            )

        outcome.notes.extend(report.notes)
        outcome.notes.insert(0, report.summary_line() + ".")
        outcome.artifacts["disagreement.report"] = report
        return outcome

    # -- regions -----------------------------------------------------------
    def _describe_regions(
        self,
        report: DisagreementReport,
        disagree: np.ndarray,
        uncertain: np.ndarray,
        contested: dict[str, list[_Opinion]],
        transform: Any,
        crs: Any,
    ) -> None:
        """Turn the contested pixels into a list a person can work through.

        Each region is attributed to the class it is actually in dispute about, and
        only the methods that assessed that class are listed. Showing every method
        would put an index that was measuring something else in the witness box.
        """
        region_id = 0
        for mask, agreement in (
            (disagree, Agreement.DISAGREE),
            (uncertain, Agreement.UNCERTAIN),
        ):
            if not mask.any():
                continue
            components = connected_components(
                mask, transform, crs, min_area_m2=MIN_CONFLICT_AREA_M2
            )
            for component in components:
                if len(report.regions) >= MAX_REGIONS:
                    report.notes.append(
                        f"Only the {MAX_REGIONS} largest contested regions are "
                        "listed; the overlay shows all of them."
                    )
                    return
                region_id += 1
                row, col = self._sample_pixel(mask, component)
                group = self._class_in_dispute(contested, row, col)
                report.regions.append(
                    ConflictRegion(
                        region_id=region_id,
                        agreement=agreement,
                        area_km2=component.area_km2,
                        pixel_count=component.pixel_count,
                        centroid_wgs84=(
                            list(component.centroid_lonlat)
                            if component.centroid_lonlat
                            else []
                        ),
                        bounds_wgs84=[],
                        opinions=self._opinions_at(group, row, col),
                        **self._reason_at(group, agreement, row, col),
                    )
                )

    @staticmethod
    def _sample_pixel(mask: np.ndarray, component: Any) -> tuple[int, int]:
        """A pixel that is genuinely inside the region.

        Not the centroid. A centroid lies outside any region that is not convex,
        and a shoreline or a field boundary rarely is, so sampling there reported
        the opinions of methods about ground the region does not cover: four
        methods all saying "not water", listed under a heading that said they
        disagreed. Taking a pixel the region actually contains is the difference
        between describing the conflict and describing its neighbourhood.
        """
        top, left, bottom, right = component.bbox_rowcol
        window = mask[top : bottom + 1, left : right + 1]
        found = np.argwhere(window)
        if found.size == 0:  # pragma: no cover - a component has pixels by
            return (                                 # construction
                int(component.centroid_rowcol[0]),
                int(component.centroid_rowcol[1]),
            )
        # The middle of the list, which for a contiguous region is well inside it.
        row_offset, col_offset = found[len(found) // 2]
        return int(top + row_offset), int(left + col_offset)

    def _class_in_dispute(
        self, contested: dict[str, list[_Opinion]], row: int, col: int
    ) -> list[_Opinion]:
        """The methods actually split about this pixel.

        Where more than one class is in dispute at the same pixel the most divided
        is chosen, because that is the conflict a reviewer needs to see first.
        """
        best: list[_Opinion] = []
        best_split = -1
        for group in contested.values():
            yes = sum(1 for o in group if bool(o.selected[row, col]))
            if yes == 0 or yes == len(group):
                continue
            split = min(yes, len(group) - yes)
            if split > best_split:
                best, best_split = group, split
        if best:
            return best
        # No class is split here, which happens for a pixel that is contested only
        # because a weak method dissented. Fall back to the largest group so the
        # region still names its witnesses.
        return max(contested.values(), key=len)

    def _opinions_at(
        self, opinions: list[_Opinion], row: int, col: int
    ) -> list[MethodOpinion]:
        """What each method says about the pixel at the region's centre.

        The centroid is a fair sample of a contiguous region and keeps the payload
        small. A per-region majority vote would hide exactly the split the panel
        exists to show.
        """
        described: list[MethodOpinion] = []
        for opinion in opinions:
            blind = (
                opinion.invalid is not None
                and bool(opinion.invalid[row, col])
            )
            if blind:
                verdict = "could not see it"
            elif bool(opinion.selected[row, col]):
                verdict = opinion.verdict.upper()
            else:
                verdict = f"not {opinion.verdict}"

            weak = (
                opinion.separability is not None
                and opinion.separability < WEAK_SEPARABILITY
            )
            described.append(
                MethodOpinion(
                    method=opinion.method,
                    verdict="UNCERTAIN" if (weak and not blind) else verdict,
                    source_tool=opinion.tool,
                    source_version=opinion.version,
                    basis=opinion.basis,
                    confidence=opinion.separability,
                    could_not_see=blind,
                )
            )
        return described

    def _reason_at(
        self,
        opinions: list[_Opinion],
        agreement: Agreement,
        row: int,
        col: int,
    ) -> dict[str, Any]:
        """Why these methods differ here, from a measurement or not at all."""
        weakest = min(
            (o for o in opinions if o.separability is not None),
            key=lambda o: o.separability or 1.0,
            default=None,
        )
        blind = [
            opinion
            for opinion in opinions
            if opinion.invalid is not None and bool(opinion.invalid[row, col])
        ]

        if blind:
            return {
                "reason": (
                    f"{blind[0].method} could not observe this ground, so the "
                    "methods are not really in conflict about it."
                ),
                "reason_measurement_key": "",
                "reason_value": None,
                "action": (
                    "Treat the remaining methods as the only evidence here, or "
                    "supply an acquisition that sees it."
                ),
            }

        if weakest is not None and (weakest.separability or 1.0) < WEAK_SEPARABILITY:
            return {
                "reason": (
                    f"Low spectral separability: {weakest.method} splits this scene "
                    f"at only {weakest.separability:.2f}, so its boundary here is a "
                    "line through one population rather than between two."
                ),
                "reason_measurement_key": "class_separability",
                "reason_value": weakest.separability,
                "action": "Human review recommended.",
            }

        if agreement is Agreement.UNCERTAIN:
            return {
                "reason": (
                    "No method reached the confidence needed for its answer to "
                    "count as a position here."
                ),
                "reason_measurement_key": "",
                "reason_value": None,
                "action": "Human review recommended.",
            }

        return {
            "reason": (
                "The methods measure different physical quantities and reach "
                "different conclusions on this ground. Nothing measured here "
                "explains which is closer to the truth."
            ),
            "reason_measurement_key": "",
            "reason_value": None,
            "action": (
                "Human review recommended, or an independent observation of this "
                "region."
            ),
        }

    # -- outputs -----------------------------------------------------------
    def _add_layers(
        self,
        outcome: ToolOutcome,
        agree: np.ndarray,
        disagree: np.ndarray,
        uncertain: np.ndarray,
        unseen: np.ndarray,
        transform: Any,
        crs: Any,
        role: ImageRole,
    ) -> None:
        for key, label, array, description in (
            (
                "evidence_agree",
                "Methods agree",
                agree,
                "Every method that could see this ground reached the same conclusion.",
            ),
            (
                "evidence_disagree",
                "Methods disagree",
                disagree,
                "At least one method said yes and another said no, over ground both "
                "could see.",
            ),
            (
                "evidence_uncertain",
                "Not enough confidence to say",
                uncertain,
                "Contested ground where a method's own index does not separate the "
                "scene, so its position is too weak to count as dissent.",
            ),
        ):
            outcome.masks.append(
                MaskLayer(
                    key=key,
                    label=label,
                    array=array,
                    transform=transform,
                    crs=crs,
                    description=description,
                    applies_to=[role],
                    invalid=unseen,
                )
            )

    def _add_measurements(
        self,
        outcome: ToolOutcome,
        report: DisagreementReport,
        candidate: np.ndarray,
        agree: np.ndarray,
        disagree: np.ndarray,
        uncertain: np.ndarray,
        transform: Any,
        role: ImageRole,
    ) -> None:
        pixels = {
            "candidate": int(np.count_nonzero(candidate)),
            "agree": int(np.count_nonzero(agree)),
            "disagree": int(np.count_nonzero(disagree)),
            "uncertain": int(np.count_nonzero(uncertain)),
        }

        for key, label, value, unit, formula in (
            (
                "conflicting_area_km2",
                "Area with conflicting evidence",
                report.disagree_area_km2 + report.uncertain_area_km2,
                "km2",
                f"({pixels['disagree']} + {pixels['uncertain']}) px x pixel area / 1e6",
            ),
            (
                "agreeing_area_km2",
                "Area every method agrees on",
                report.agree_area_km2,
                "km2",
                f"{pixels['agree']} px x pixel area / 1e6",
            ),
            (
                "uncertain_area_km2",
                "Area no method is confident about",
                report.uncertain_area_km2,
                "km2",
                f"{pixels['uncertain']} px x pixel area / 1e6",
            ),
            (
                "conflicting_fraction",
                "Share of the candidate area that is contested",
                report.conflicting_fraction,
                "fraction",
                f"({pixels['disagree']} + {pixels['uncertain']}) contested px / "
                f"{pixels['candidate']} candidate px",
            ),
            (
                "methods_compared",
                "Methods compared",
                float(len(report.methods)),
                "count",
                "distinct methods that assessed a class another method also assessed",
            ),
            (
                "conflict_regions",
                "Contested regions listed for review",
                float(len(report.regions)),
                "count",
                f"connected components above {MIN_CONFLICT_AREA_M2:.0f} m2",
            ),
        ):
            outcome.measurements.append(
                self.measurement(
                    key=key,
                    label=label,
                    value=value,
                    unit=unit,
                    formula=formula,
                    inputs={**pixels, "methods": report.methods},
                    method=(
                        "counted from the per-pixel comparison of every method's "
                        "own mask"
                    ),
                    applies_to=[role],
                    precision=4 if unit == "fraction" else 3,
                )
            )

    def _geometry(self, context: ToolContext, role: ImageRole) -> tuple[Any, Any]:
        """The transform and CRS the masks share, taken from an upstream layer."""
        for outcome in context.upstream.values():
            for layer in getattr(outcome, "masks", []):
                if layer.crs is not None:
                    return layer.transform, layer.crs
        meta = context.metadata(role)
        raise ToolError(
            "No upstream mask carried a coordinate reference system, so the "
            f"comparison cannot be placed on the ground ({meta.original_filename})."
        )


__all__ = [
    "CLASS_OF_MASK",
    "MAX_REGIONS",
    "METHOD_LABEL",
    "MIN_CONFLICT_AREA_M2",
    "UNCERTAIN_CONFIDENCE",
    "DisagreementEngine",
]
