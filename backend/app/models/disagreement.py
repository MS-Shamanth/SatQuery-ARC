"""Where the methods do not agree, and why.

Every other panel in this system reports what was found. This one reports where
the findings conflict, which is a different and more useful thing to show. A map
with one mask on it invites the reader to believe the mask. A map that marks the
ground where two independent methods reached opposite conclusions tells them
exactly where believing it would be a mistake.

The conflicts are not errors. Each method is applied correctly and reports what it
measures; they disagree because the ground is genuinely ambiguous at this
resolution, or because the methods measure different physical quantities. The
honest presentation is to draw the conflict, name which method said what, give the
measured reason, and recommend review rather than picking a winner. There is no
labelled reference for these scenes, so picking a winner would mean inventing the
one thing this system refuses to invent.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.schemas import ImageRole


class Agreement(str, Enum):
    """How the methods stand on one piece of ground."""

    # Every method that could see it reached the same conclusion.
    AGREE = "agree"
    # At least two methods reached opposite conclusions.
    DISAGREE = "disagree"
    # No method was confident enough for its answer to count as a position.
    UNCERTAIN = "uncertain"


AGREEMENT_LABEL: dict[Agreement, str] = {
    Agreement.AGREE: "Methods agree",
    Agreement.DISAGREE: "Methods disagree",
    Agreement.UNCERTAIN: "Not enough confidence to say",
}


class MethodOpinion(BaseModel):
    """One method's reading of one region, in its own terms."""

    method: str
    # What the method calls this ground, e.g. "built-up", "vegetation", "water".
    verdict: str
    # The tool that produced it, so the opinion is attributable.
    source_tool: str
    source_version: str
    # How the method arrived at it: the index and threshold, or the model.
    basis: str
    # The method's own confidence where it reports one. Left unset rather than
    # invented for methods that do not produce a probability.
    confidence: float | None = None
    # True when the method could not see this ground at all, as opposed to seeing
    # it and disagreeing. Cloud and nodata are not dissent.
    could_not_see: bool = False


class ConflictRegion(BaseModel):
    """A contiguous patch of ground the methods do not agree about."""

    region_id: int
    agreement: Agreement
    area_km2: float
    pixel_count: int
    # Where to look, in degrees, so the region can be found on the map.
    centroid_wgs84: list[float] = Field(default_factory=list)
    bounds_wgs84: list[float] = Field(default_factory=list)

    opinions: list[MethodOpinion] = Field(default_factory=list)

    # Why they disagree, measured rather than guessed.
    reason: str
    reason_measurement_key: str = ""
    reason_value: float | None = None

    # What a human should do about it. Always an action, never a verdict.
    action: str

    @property
    def dissenting_methods(self) -> list[str]:
        return sorted({opinion.verdict for opinion in self.opinions if not opinion.could_not_see})


class DisagreementReport(BaseModel):
    """The whole scene's agreement picture."""

    # The ground any method put forward as belonging to the class under question.
    candidate_area_km2: float
    agree_area_km2: float
    disagree_area_km2: float
    uncertain_area_km2: float

    # The headline: how much of the candidate area is contested.
    conflicting_fraction: float

    methods: list[str] = Field(default_factory=list)
    regions: list[ConflictRegion] = Field(default_factory=list)
    applies_to: list[ImageRole] = Field(default_factory=list)

    # Mask keys for the three overlays, so the client can colour the map from the
    # same arrays these figures were counted from.
    agree_layer: str = ""
    disagree_layer: str = ""
    uncertain_layer: str = ""

    notes: list[str] = Field(default_factory=list)

    @property
    def has_conflict(self) -> bool:
        return self.disagree_area_km2 > 0.0

    def summary_line(self) -> str:
        if not self.methods:
            return "Only one method produced a result, so there was nothing to compare."
        return (
            f"{self.conflicting_fraction:.1%} of the candidate area has conflicting "
            f"evidence across {len(self.methods)} methods"
        )


__all__ = [
    "AGREEMENT_LABEL",
    "Agreement",
    "ConflictRegion",
    "DisagreementReport",
    "MethodOpinion",
]
