"""Turning a refusal into something the user can act on.

The confounder engine already says what data would settle an open question. That
is most of the value, but only most: "you would need an acquisition from the same
part of the growing season" is advice, and advice that the system could satisfy
itself is better delivered as an offer.

This module matches surviving alternative explanations against the sample library
and returns the scenes that remove them. On the seasonal pair the result is the
same fields a year apart at the same point in the growing cycle, which is exactly
what the refusal asked for. Loading it and re-running is then a single action, and
the answer changes because the objection is gone rather than because anything was
adjusted to make it go away.

Nothing here weakens a refusal. A remedy is offered only when a scene genuinely
removes the confounder by construction, and the scene says why.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.models.confounders import ConfounderReport, ConfounderVerdict
from app.models.contract import ConfounderKind
from app.models.schemas import InputConfiguration, SampleManifest

logger = logging.getLogger(__name__)


class RemedyOffer(BaseModel):
    """Imagery in the library that removes a surviving objection."""

    confounder: ConfounderKind
    confounder_label: str
    # What the confounder test said was needed, verbatim.
    requirement: str

    sample_key: str
    title: str
    place: str
    configuration: InputConfiguration
    # Why this scene settles it, from the scene's own declaration.
    why: str
    # True when the scene exists specifically to answer this input's objection.
    corrects_current: bool = False
    suggested_query: str | None = None


class RemedySet(BaseModel):
    offers: list[RemedyOffer] = Field(default_factory=list)
    # Objections with no scene in the library that answers them. Listed so the
    # absence of an offer is not read as the absence of an objection.
    unmet: list[str] = Field(default_factory=list)

    @property
    def any_offered(self) -> bool:
        return bool(self.offers)


def offers_for(
    report: ConfounderReport | None,
    manifest: SampleManifest,
    *,
    current_sample: str | None = None,
) -> RemedySet:
    """Scenes that would settle whatever is still standing.

    Only cached scenes are offered. A scene the operator has not fetched cannot
    be loaded, and offering it would be a button that does nothing.
    """
    result = RemedySet()
    if report is None:
        return result

    surviving = [
        test
        for test in report.tests
        if test.verdict
        in {ConfounderVerdict.LIKELY, ConfounderVerdict.PLAUSIBLE}
        and test.requirement
    ]
    if not surviving:
        return result

    for test in surviving:
        matched = False
        for scene in manifest.scenes.values():
            if test.kind.value not in scene.remedies:
                continue
            if not scene.assets:
                logger.debug(
                    "scene %s remedies %s but is not cached", scene.key, test.kind
                )
                continue
            if scene.key == current_sample:
                # Already looking at it.
                matched = True
                continue

            result.offers.append(
                RemedyOffer(
                    confounder=test.kind,
                    confounder_label=test.label,
                    requirement=test.requirement or "",
                    sample_key=scene.key,
                    title=scene.title,
                    place=scene.place,
                    configuration=scene.configuration,
                    why=scene.remedy_note or scene.description,
                    corrects_current=(
                        current_sample is not None
                        and scene.corrects == current_sample
                    ),
                    suggested_query=(
                        scene.suggested_queries[0]
                        if scene.suggested_queries
                        else None
                    ),
                )
            )
            matched = True

        if not matched and test.requirement:
            result.unmet.append(test.requirement)

    # A scene built specifically to answer this input comes first: it is the one
    # the refusal was actually asking for.
    result.offers.sort(key=lambda offer: (not offer.corrects_current, offer.title))
    return result


__all__ = ["RemedyOffer", "RemedySet", "offers_for"]
