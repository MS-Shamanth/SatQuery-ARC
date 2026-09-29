"""The evidence ledger and the verdict.

This is where the system commits. Everything before it measures; this decides.

Four ideas hold it together:

* **The ledger is explicit.** Every measurement admitted as evidence is listed
  with the direction it points and why it was given the weight it was. Nothing
  contributes to a conclusion without appearing here.
* **Independent measurements are compared, not averaged.** Two estimates of the
  same quantity that disagree are more informative than their mean, so a
  ``ConsistencyCheck`` records the disagreement rather than hiding it.
* **Confidence is a sum of named parts.** A single opaque percentage is
  unfalsifiable. Each component states what it measured, what it contributed, and
  what would change it.
* **A number that cannot be traced is not printed.** ``NumericAudit`` checks every
  figure in the written explanation against the ledger, and an explanation that
  fails is discarded rather than shown with a caveat.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.confounders import ConfounderTest
from app.models.contract import ChangeDirection, ConfounderKind


class VerdictLabel(str, Enum):
    """The conclusion.

    ``UNANSWERABLE`` is distinct from ``INCONCLUSIVE`` on purpose. Inconclusive
    means the system looked and could not decide. Unanswerable means the question
    cannot be settled with the data supplied, whatever the analysis does, and the
    honest response is to say which data would be needed.
    """

    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    UNANSWERABLE = "unanswerable"


VERDICT_LABEL_TEXT: dict[VerdictLabel, str] = {
    VerdictLabel.SUPPORTED: "Supported",
    VerdictLabel.REFUTED: "Refuted",
    VerdictLabel.INCONCLUSIVE: "Inconclusive",
    VerdictLabel.UNANSWERABLE: "Unanswerable with this data",
}


class EvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class EvidenceItem(BaseModel):
    """One measurement admitted as evidence, and what it argues for."""

    measurement_key: str
    label: str
    value: float
    unit: str
    # The value as it should be written, at the precision the measuring tool
    # declared. Narration is handed this rather than the raw float, so a phrased
    # explanation reads cleanly without the model having to round anything itself.
    display: str
    direction: EvidenceDirection
    # Why this measurement bears on the claim at all.
    relevance: str
    # What it says, in plain language, with its own number.
    statement: str
    source_tool: str
    source_version: str
    formula: str
    # Independent of the other items, or derived from the same pixels and
    # therefore not a second opinion. Stated because it governs the weight.
    independent: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ConsistencyCheck(BaseModel):
    """Two independent estimates of the same quantity, compared.

    Averaging two disagreeing estimates produces a number neither method
    supports. Recording the disagreement keeps the uncertainty visible where it
    belongs.
    """

    quantity: str
    label: str
    first_key: str
    first_value: float
    second_key: str
    second_value: float
    unit: str
    relative_difference: float = Field(
        description="|a - b| / max(|a|, |b|), so 0 is perfect agreement"
    )
    tolerance: float
    agrees: bool
    method: str
    explanation: str


class ConfidenceComponent(BaseModel):
    """One named contribution to the confidence figure."""

    name: str
    label: str
    measured: float
    # Contribution after weighting, signed: negative components reduce confidence.
    contribution: float
    weight: float
    # The best this component could contribute. For a positive component that is
    # its weight; for a penalty it is zero, since the best outcome is costing
    # nothing. Without this a penalty that costs nothing still looks like a
    # shortfall, and the "what would change it" list fills up with levers that
    # would change nothing.
    best: float = 0.0
    rationale: str
    # What would move this component.
    lever: str = ""

    @property
    def headroom(self) -> float:
        """How much confidence this component is currently leaving on the table."""
        return max(0.0, self.best - self.contribution)


class NumericAudit(BaseModel):
    """Whether every number in the written explanation traces to a measurement.

    The Never-Guess Rule turned into a check that can fail. A language model
    asked to phrase a finding will occasionally round, restate, or invent a
    figure; this catches all three by requiring each numeric token in the text to
    match a value already in the ledger.
    """

    checked: int = 0
    traced: int = 0
    untraceable: list[str] = Field(default_factory=list)
    passed: bool = True
    note: str = ""


class Verdict(BaseModel):
    """The conclusion, with everything it rests on."""

    label: VerdictLabel
    claim: str
    asserted_direction: ChangeDirection = ChangeDirection.UNSPECIFIED
    measured_direction: ChangeDirection = ChangeDirection.UNSPECIFIED

    confidence: float = Field(ge=0.0, le=1.0)
    confidence_components: list[ConfidenceComponent] = Field(default_factory=list)

    evidence: list[EvidenceItem] = Field(default_factory=list)
    consistency: list[ConsistencyCheck] = Field(default_factory=list)

    # Alternative explanations that are still standing, and what would close them.
    surviving_confounders: list[ConfounderKind] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)

    # The deterministic explanation, always present.
    reasoning: str = ""
    # The phrased explanation, present only when it survived the numeric audit.
    narrative: str | None = None
    narrative_source: str = "template"
    narrative_audit: NumericAudit | None = None

    # Stated plainly so the reader knows what would flip the answer.
    what_would_change_it: list[str] = Field(default_factory=list)

    @property
    def decided(self) -> bool:
        return self.label in {VerdictLabel.SUPPORTED, VerdictLabel.REFUTED}

    @property
    def supporting(self) -> list[EvidenceItem]:
        return [
            item for item in self.evidence
            if item.direction is EvidenceDirection.SUPPORTS
        ]

    @property
    def refuting(self) -> list[EvidenceItem]:
        return [
            item for item in self.evidence
            if item.direction is EvidenceDirection.REFUTES
        ]

    def summary_line(self) -> str:
        return (
            f"{VERDICT_LABEL_TEXT[self.label]} at {self.confidence:.0%} confidence, "
            f"on {len(self.evidence)} piece(s) of evidence with "
            f"{len(self.surviving_confounders)} alternative explanation(s) standing"
        )


class EvidenceLedger(BaseModel):
    """Everything the verdict was allowed to consider."""

    items: list[EvidenceItem] = Field(default_factory=list)
    consistency: list[ConsistencyCheck] = Field(default_factory=list)
    confounders: list[ConfounderTest] = Field(default_factory=list)
    # Measurements present in the run but not admitted, with the reason.
    excluded: dict[str, str] = Field(default_factory=dict)

    @property
    def independent_count(self) -> int:
        return sum(1 for item in self.items if item.independent)

    def by_key(self, key: str) -> EvidenceItem | None:
        for item in self.items:
            if item.measurement_key == key:
                return item
        return None


__all__ = [
    "VERDICT_LABEL_TEXT",
    "ConfidenceComponent",
    "ConsistencyCheck",
    "EvidenceDirection",
    "EvidenceItem",
    "EvidenceLedger",
    "NumericAudit",
    "Verdict",
    "VerdictLabel",
]
