"""Confounder test results.

A confounder is an alternative explanation for a finding. Naming one is easy;
the value is in testing it and saying what the test measured.

Each test reports four things, and all four are load-bearing:

* a **verdict** on whether the alternative explanation survives,
* the **number** the verdict rests on, with the formula that produced it,
* the **threshold** it was compared against, stated up front rather than
  chosen to fit,
* and, when the alternative cannot be ruled out, **what data would settle it**.

That last field is the one that turns a refusal into something useful. A system
that says "inconclusive" is unhelpful; one that says "inconclusive, and here is
the acquisition that would decide it" is doing the job.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.contract import ConfounderKind
from app.models.schemas import ImageRole


class ConfounderVerdict(str, Enum):
    """Whether the alternative explanation survived its test."""

    RULED_OUT = "ruled_out"
    PLAUSIBLE = "plausible"
    LIKELY = "likely"
    NOT_TESTED = "not_tested"


VERDICT_LABEL: dict[ConfounderVerdict, str] = {
    ConfounderVerdict.RULED_OUT: "ruled out",
    ConfounderVerdict.PLAUSIBLE: "cannot be ruled out",
    ConfounderVerdict.LIKELY: "likely explanation",
    ConfounderVerdict.NOT_TESTED: "not tested",
}

# How much each verdict should reduce confidence in the finding. Used by the
# verdict engine so that the weighting is declared in one place rather than
# improvised where it is applied.
VERDICT_PENALTY: dict[ConfounderVerdict, float] = {
    ConfounderVerdict.RULED_OUT: 0.0,
    ConfounderVerdict.PLAUSIBLE: 0.25,
    ConfounderVerdict.LIKELY: 0.7,
    ConfounderVerdict.NOT_TESTED: 0.15,
}


class ConfounderTest(BaseModel):
    """One alternative explanation, tested."""

    kind: ConfounderKind
    label: str
    question: str
    verdict: ConfounderVerdict

    measured: str = Field(description="The result as it should be displayed")
    measured_numeric: float | None = None
    unit: str | None = None
    threshold: str = Field(description="What the measurement was compared against")

    formula: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    method: str = ""
    # What the number means for the finding, in plain language.
    explanation: str = ""
    # Present when the alternative could not be ruled out: the acquisition or
    # ancillary data that would settle it.
    requirement: str | None = None

    applies_to: list[ImageRole] = Field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def survived(self) -> bool:
        """True when this alternative explanation is still on the table."""
        return self.verdict in {
            ConfounderVerdict.PLAUSIBLE,
            ConfounderVerdict.LIKELY,
            ConfounderVerdict.NOT_TESTED,
        }

    @property
    def penalty(self) -> float:
        return VERDICT_PENALTY[self.verdict]


class ConfounderReport(BaseModel):
    """Every alternative explanation considered for one finding."""

    tests: list[ConfounderTest] = Field(default_factory=list)
    # Alternatives named by the contract that nothing in this build can test.
    untested_kinds: list[ConfounderKind] = Field(default_factory=list)

    def by_kind(self, kind: ConfounderKind) -> ConfounderTest | None:
        for test in self.tests:
            if test.kind is kind:
                return test
        return None

    @property
    def ruled_out(self) -> list[ConfounderTest]:
        return [t for t in self.tests if t.verdict is ConfounderVerdict.RULED_OUT]

    @property
    def surviving(self) -> list[ConfounderTest]:
        return [t for t in self.tests if t.survived]

    @property
    def likely(self) -> list[ConfounderTest]:
        return [t for t in self.tests if t.verdict is ConfounderVerdict.LIKELY]

    @property
    def requirements(self) -> list[str]:
        seen: list[str] = []
        for test in self.tests:
            if test.requirement and test.requirement not in seen:
                seen.append(test.requirement)
        return seen

    def summary_line(self) -> str:
        return (
            f"{len(self.ruled_out)} of {len(self.tests)} alternative explanation(s) "
            f"ruled out, {len(self.likely)} likely"
        )


__all__ = [
    "VERDICT_LABEL",
    "VERDICT_PENALTY",
    "ConfounderReport",
    "ConfounderTest",
    "ConfounderVerdict",
]
