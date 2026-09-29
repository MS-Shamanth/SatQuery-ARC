"""Remedy offer tests.

A refusal that names the data it needs is useful. One that can hand you that data
is actionable. These tests guard the line between the two: an offer may only be
made when a scene genuinely removes the objection, and making one must never
soften the objection itself.
"""

from __future__ import annotations

import pytest

from app.core.remedy import offers_for
from app.models.confounders import (
    ConfounderReport,
    ConfounderTest,
    ConfounderVerdict,
)
from app.models.contract import ConfounderKind
from app.models.schemas import (
    ImageRole,
    InputConfiguration,
    SampleAssetRecord,
    SampleManifest,
    SampleProvenance,
    SampleScene,
)
from app.core.sample_sources import now_utc

SEASONAL_REQUIREMENT = (
    "An acquisition from the same part of the growing season as date_a"
)


def probe(
    kind: ConfounderKind,
    verdict: ConfounderVerdict,
    *,
    requirement: str | None = SEASONAL_REQUIREMENT,
) -> ConfounderTest:
    return ConfounderTest(
        kind=kind,
        label=kind.value.replace("_", " ").title(),
        question="Could something else explain this?",
        verdict=verdict,
        measured="83% of the scene flagged",
        threshold="likely above 50%",
        explanation="Because of what was measured.",
        requirement=requirement,
    )


def asset() -> SampleAssetRecord:
    return SampleAssetRecord(
        role=ImageRole.DATE_A,
        filename="x.tif",
        sha256="a" * 64,
        size_bytes=1024,
        width=512,
        height=512,
        band_count=7,
        provenance=SampleProvenance(source="test"),
    )


def scene(
    key: str,
    *,
    remedies: list[str] | None = None,
    corrects: str | None = None,
    cached: bool = True,
) -> SampleScene:
    return SampleScene(
        key=key,
        title=key.replace("_", " ").title(),
        description="A scene.",
        demo="test",
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        suggested_queries=["Did vegetation decrease?"],
        place="Somewhere",
        assets={ImageRole.DATE_A: asset()} if cached else {},
        remedies=remedies or [],
        corrects=corrects,
        remedy_note="Both dates sit at the same point in the growing cycle.",
    )


def manifest(*scenes: SampleScene) -> SampleManifest:
    return SampleManifest(
        generated_at=now_utc(), scenes={item.key: item for item in scenes}
    )


LIBRARY = manifest(
    scene("seasonal_farmland"),
    scene(
        "seasonal_farmland_same_season",
        remedies=[ConfounderKind.SEASONALITY.value],
        corrects="seasonal_farmland",
    ),
)


def test_a_likely_confounder_is_matched_to_the_scene_that_settles_it():
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY)]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")

    assert len(result.offers) == 1
    offer = result.offers[0]
    assert offer.sample_key == "seasonal_farmland_same_season"
    assert offer.confounder is ConfounderKind.SEASONALITY
    assert offer.requirement == SEASONAL_REQUIREMENT
    assert offer.why
    assert offer.corrects_current is True


def test_a_merely_plausible_confounder_is_also_worth_offering_for():
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.PLAUSIBLE)]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")
    assert result.any_offered


def test_nothing_is_offered_when_everything_was_ruled_out():
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.RULED_OUT)]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")
    assert not result.any_offered
    assert result.unmet == []


def test_the_scene_already_loaded_is_not_offered_to_itself():
    """Offering the imagery under examination would be a button that changes nothing."""
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY)]
    )
    result = offers_for(
        report, LIBRARY, current_sample="seasonal_farmland_same_season"
    )
    assert not result.any_offered
    # And the objection is not recorded as unmet either: something does answer it.
    assert result.unmet == []


def test_an_uncached_scene_is_not_offered():
    """A button that cannot load anything is worse than no button."""
    library = manifest(
        scene(
            "seasonal_farmland_same_season",
            remedies=[ConfounderKind.SEASONALITY.value],
            corrects="seasonal_farmland",
            cached=False,
        )
    )
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY)]
    )
    result = offers_for(report, library, current_sample="seasonal_farmland")

    assert not result.any_offered
    assert SEASONAL_REQUIREMENT in result.unmet


def test_an_objection_with_no_answer_in_the_library_is_recorded_as_unmet():
    """The absence of an offer must not read as the absence of an objection."""
    report = ConfounderReport(
        tests=[
            probe(
                ConfounderKind.RADIOMETRY,
                ConfounderVerdict.LIKELY,
                requirement="Consistently corrected reflectance products",
            )
        ]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")

    assert not result.any_offered
    assert result.unmet == ["Consistently corrected reflectance products"]


def test_a_confounder_with_no_stated_requirement_produces_no_offer():
    report = ConfounderReport(
        tests=[
            probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY, requirement=None)
        ]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")
    assert not result.any_offered


def test_the_scene_built_for_this_input_is_offered_first():
    library = manifest(
        scene(
            "generic_same_season",
            remedies=[ConfounderKind.SEASONALITY.value],
        ),
        scene(
            "seasonal_farmland_same_season",
            remedies=[ConfounderKind.SEASONALITY.value],
            corrects="seasonal_farmland",
        ),
    )
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY)]
    )
    result = offers_for(report, library, current_sample="seasonal_farmland")

    assert len(result.offers) == 2
    assert result.offers[0].sample_key == "seasonal_farmland_same_season"
    assert result.offers[0].corrects_current is True
    assert result.offers[1].corrects_current is False


def test_an_offer_carries_a_query_to_ask_of_it():
    """The point is to re-ask the same question, so the question travels with it."""
    report = ConfounderReport(
        tests=[probe(ConfounderKind.SEASONALITY, ConfounderVerdict.LIKELY)]
    )
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")
    assert result.offers[0].suggested_query == "Did vegetation decrease?"


def test_no_report_means_no_offers():
    assert not offers_for(None, LIBRARY).any_offered


@pytest.mark.parametrize(
    "verdict",
    [ConfounderVerdict.RULED_OUT, ConfounderVerdict.NOT_TESTED],
)
def test_only_surviving_tested_objections_generate_offers(verdict):
    """An untested objection has no measured requirement to satisfy."""
    report = ConfounderReport(tests=[probe(ConfounderKind.SEASONALITY, verdict)])
    result = offers_for(report, LIBRARY, current_sample="seasonal_farmland")
    assert not result.any_offered
