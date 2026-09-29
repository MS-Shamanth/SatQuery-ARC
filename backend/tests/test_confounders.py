"""Confounder engine tests.

Each test here constructs a scene where the alternative explanation is *true* and
asserts the engine catches it, then constructs one where it is false and asserts
the engine clears it. A test that only ever checks the clean case would pass for a
confounder engine that always says "ruled out".
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.readiness import evaluate_readiness
from app.models.confounders import ConfounderVerdict
from app.models.contract import ConfounderKind
from app.models.schemas import ImageRole
from app.tools.change import ChangeCvaEngine
from app.tools.confounders import (
    SCENE_WIDE_EXTENT_LIKELY,
    ConfounderEngine,
)
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    optical_from_labels,
    write_raster,
)

MARCH = datetime(2024, 3, 12, tzinfo=timezone.utc)
MAY = datetime(2024, 5, 31, tzinfo=timezone.utc)
NEXT_MARCH = datetime(2025, 3, 14, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    return ConfounderEngine()


def run_pair(
    make_context,
    store,
    path_a,
    path_b,
    *,
    targets: list[str],
    change_parameters: dict | None = None,
    confounder_kinds: list[ConfounderKind] | None = None,
):
    """Run the change engine, then the confounder engine on its output."""
    context = make_context(
        {ImageRole.DATE_A: path_a, ImageRole.DATE_B: path_b},
        target_classes=targets,
        with_readiness=True,
    )
    change = ChangeCvaEngine().run(
        replace(context, parameters=change_parameters or {})
    )
    assert change.ok, change.skipped_reason

    parameters: dict = {}
    if confounder_kinds is not None:
        parameters["kinds"] = [kind.value for kind in confounder_kinds]

    probe = replace(
        context,
        parameters=parameters,
        upstream={"change-cva-engine": change},
    )
    outcome = ConfounderEngine().run(probe)
    assert outcome.ok, outcome.skipped_reason
    return outcome, outcome.artifacts["confounders.report"], change


# -- seasonality: the discriminating test ---------------------------------


def test_a_localised_change_on_same_season_dates_rules_out_seasonality(
    tmp_path, make_context, store, engine
):
    before = make_labels(builtup=(140, 30, 70, 60), water=None, bare=None)
    after = make_labels(builtup=(140, 30, 70, 120), water=None, bare=None)
    a = make_optical_scene(
        tmp_path / "a.tif", labels=before, acquisition_date=MARCH, seed=3
    )
    b = make_optical_scene(
        tmp_path / "b.tif", labels=after, acquisition_date=NEXT_MARCH, seed=4
    )

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )
    test = report.by_kind(ConfounderKind.SEASONALITY)
    assert test is not None
    assert test.verdict is ConfounderVerdict.RULED_OUT
    assert test.requirement is None


def test_a_whole_scene_shift_across_seasons_makes_seasonality_likely(
    tmp_path, make_context, store, engine
):
    """March wheat to May stubble: the whole landscape moves, so nothing changed.

    This is the case a plain difference image gets wrong and reports as a
    catastrophic vegetation loss.
    """
    green = make_labels(water=None, builtup=None, bare=None)
    harvested = np.full_like(green, int(LandCover.BARE))

    a = make_optical_scene(
        tmp_path / "green.tif", labels=green, acquisition_date=MARCH, seed=3
    )
    b = make_optical_scene(
        tmp_path / "bare.tif", labels=harvested, acquisition_date=MAY, seed=4
    )

    _, report, change = run_pair(
        make_context, store, a.path, b.path, targets=["vegetation"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )

    # The change engine must genuinely find the apparent loss, or the confounder
    # engine is being asked to explain away nothing.
    scene_km2 = 256 * 256 * 100 / 1_000_000.0
    assert change.measurement("area_km2_before").value > 0.9 * scene_km2
    assert change.measurement("area_km2_after").value < 0.15 * scene_km2

    test = report.by_kind(ConfounderKind.SEASONALITY)
    assert test is not None
    assert test.verdict is ConfounderVerdict.LIKELY
    assert test.requirement is not None
    assert "same part of the growing season" in test.requirement


def test_extent_decides_when_the_inside_outside_contrast_would_be_circular(
    tmp_path, make_context, store, engine
):
    """Once most of the scene is flagged, "outside" is just the part that held still.

    The contrast then looks large whatever the cause, so it must not be the
    discriminator.
    """
    green = make_labels(water=None, builtup=None, bare=None)
    # Most of the landscape is harvested; a strip of perennial cover holds still,
    # so there is an "outside" and its shift is genuinely near zero. That is the
    # configuration where the contrast reads low while the change is still
    # unmistakably scene-wide.
    harvested = green.copy()
    harvested[40:, :] = int(LandCover.BARE)

    a = make_optical_scene(tmp_path / "a.tif", labels=green, acquisition_date=MARCH)
    b = make_optical_scene(tmp_path / "b.tif", labels=harvested, acquisition_date=MAY)

    outcome, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["vegetation"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )

    extent = outcome.measurement("changed_extent_of_scene")
    share = outcome.measurement("scene_wide_share")
    assert extent is not None and share is not None
    assert extent.value > SCENE_WIDE_EXTENT_LIKELY
    # The contrast is low precisely because the detection is nearly everything.
    assert share.value < 0.5
    assert report.by_kind(ConfounderKind.SEASONALITY).verdict is (
        ConfounderVerdict.LIKELY
    )


def test_a_localised_change_across_seasons_is_plausible_not_ruled_out(
    tmp_path, make_context, store, engine
):
    """Seasonally offset dates leave phenology on the table even when localised."""
    before = make_labels(builtup=(140, 30, 70, 60), water=None, bare=None)
    after = make_labels(builtup=(140, 30, 70, 110), water=None, bare=None)
    a = make_optical_scene(tmp_path / "a.tif", labels=before, acquisition_date=MARCH)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, acquisition_date=MAY)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )
    test = report.by_kind(ConfounderKind.SEASONALITY)
    assert test is not None
    assert test.verdict is ConfounderVerdict.PLAUSIBLE
    assert test.requirement


# -- misregistration -------------------------------------------------------


def test_a_shifted_image_makes_misregistration_likely(
    tmp_path, make_context, store, engine
):
    """Change concentrated on every boundary in the scene is a shifted image."""
    labels = make_labels()
    a = make_optical_scene(tmp_path / "a.tif", labels=labels, seed=3)
    # Same ground, same grid, content rolled by 5 px: every edge becomes change.
    shifted = np.roll(labels, shift=5, axis=1)
    b = make_optical_scene(tmp_path / "b.tif", labels=shifted, seed=3)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.MISREGISTRATION],
    )
    test = report.by_kind(ConfounderKind.MISREGISTRATION)
    assert test is not None
    assert test.verdict in {ConfounderVerdict.LIKELY, ConfounderVerdict.PLAUSIBLE}
    assert test.measured_numeric is not None
    assert test.requirement


def test_a_genuine_new_block_on_an_aligned_pair_rules_out_misregistration(
    tmp_path, make_context, store, engine
):
    """A block appearing where the first date had no feature is not an artifact.

    The two dates are otherwise identical, so alignment measures near zero and the
    detected region's interior is not an edge.
    """
    plain = make_labels(water=None, builtup=None, bare=None)
    after = make_labels(builtup=(40, 40, 150, 150), water=None, bare=None)
    a = make_optical_scene(tmp_path / "a.tif", labels=plain, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, seed=3)

    outcome, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.MISREGISTRATION],
    )
    test = report.by_kind(ConfounderKind.MISREGISTRATION)
    assert test is not None
    assert test.verdict is ConfounderVerdict.RULED_OUT
    enrichment = outcome.measurement("edge_enrichment")
    assert enrichment is not None
    assert enrichment.value < 1.5


def test_a_shift_scores_higher_on_edge_concentration_than_a_real_change(
    tmp_path, make_context, store, engine
):
    """The ordering is the discriminating property, so assert it directly."""
    labels = make_labels()

    plain = make_labels(water=None, builtup=None, bare=None)
    after = make_labels(builtup=(40, 40, 150, 150), water=None, bare=None)
    real, _, _ = run_pair(
        make_context,
        store,
        make_optical_scene(tmp_path / "ra.tif", labels=plain, seed=3).path,
        make_optical_scene(tmp_path / "rb.tif", labels=after, seed=3).path,
        targets=["built-up"],
        confounder_kinds=[ConfounderKind.MISREGISTRATION],
    )
    artifact, _, _ = run_pair(
        make_context,
        store,
        make_optical_scene(tmp_path / "sa.tif", labels=labels, seed=3).path,
        make_optical_scene(
            tmp_path / "sb.tif", labels=np.roll(labels, shift=5, axis=1), seed=3
        ).path,
        targets=["built-up"],
        confounder_kinds=[ConfounderKind.MISREGISTRATION],
    )

    real_value = real.measurement("edge_enrichment")
    artifact_value = artifact.measurement("edge_enrichment")
    assert real_value is not None and artifact_value is not None
    assert artifact_value.value > real_value.value


# -- cloud and shadow ------------------------------------------------------


def test_a_clear_pair_rules_out_cloud_without_needing_a_measurement(
    tmp_path, make_context, store, engine
):
    before = make_labels(builtup=(140, 30, 70, 60))
    after = make_labels(builtup=(140, 30, 70, 120))
    a = make_optical_scene(tmp_path / "a.tif", labels=before, with_scl=True, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, with_scl=True, seed=4)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.CLOUD_SHADOW],
    )
    test = report.by_kind(ConfounderKind.CLOUD_SHADOW)
    assert test is not None
    assert test.verdict is ConfounderVerdict.RULED_OUT
    assert "no pixel was excluded" in test.measured


def test_change_hugging_a_cloud_edge_is_flagged(
    tmp_path, make_context, store, engine
):
    """Thin cloud and shadow penumbra are not sharply bounded.

    Change detected right against the masked region is more likely to be that
    fringe than ground change.
    """
    # A thin band of apparent new built-up immediately below the cloud edge: the
    # shape a shadow penumbra or a thin cirrus margin produces.
    after = make_labels(builtup=(90, 0, 5, 256), water=None, bare=None)
    a = make_optical_scene(
        tmp_path / "a.tif",
        labels=make_labels(cloud=(0, 0, 90, 256), water=None, builtup=None, bare=None),
        with_scl=True,
        seed=3,
    )
    b = make_optical_scene(tmp_path / "b.tif", labels=after, with_scl=True, seed=4)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.CLOUD_SHADOW],
    )
    test = report.by_kind(ConfounderKind.CLOUD_SHADOW)
    assert test is not None
    assert test.verdict in {ConfounderVerdict.LIKELY, ConfounderVerdict.PLAUSIBLE}
    assert test.requirement
    assert "SAR" in test.requirement


# -- radiometry ------------------------------------------------------------


def test_a_uniform_brightness_offset_makes_radiometry_likely(
    tmp_path, make_context, store, engine
):
    """If surfaces that cannot have changed read differently, the sensor moved."""
    labels = make_labels()
    cube = optical_from_labels(labels, seed=3)
    a_path = tmp_path / "a.tif"
    write_raster(
        a_path,
        cube,
        epsg=32643,
        origin=(600000.0, 2000000.0),
        pixel_size=10.0,
        band_names=("blue", "green", "red", "nir", "swir16", "swir22"),
        tags={"ACQUISITION_DATE": MARCH.isoformat()},
    )
    # Every band lifted by 800 DN, about 0.08 reflectance: no ground change at all.
    brighter = np.clip(cube.astype("int32") + 800, 1, 10000).astype("uint16")
    b_path = tmp_path / "b.tif"
    write_raster(
        b_path,
        brighter,
        epsg=32643,
        origin=(600000.0, 2000000.0),
        pixel_size=10.0,
        band_names=("blue", "green", "red", "nir", "swir16", "swir22"),
        tags={"ACQUISITION_DATE": NEXT_MARCH.isoformat()},
    )

    outcome, report, _ = run_pair(
        make_context, store, a_path, b_path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.RADIOMETRY],
    )
    test = report.by_kind(ConfounderKind.RADIOMETRY)
    assert test is not None
    assert test.verdict is ConfounderVerdict.LIKELY
    shift = outcome.measurement("pseudo_invariant_shift")
    assert shift is not None
    assert shift.value > 0.05
    assert test.requirement


def test_an_identically_calibrated_pair_rules_out_radiometry(
    tmp_path, make_context, store, engine
):
    before = make_labels(builtup=(140, 30, 70, 60))
    after = make_labels(builtup=(140, 30, 70, 120))
    a = make_optical_scene(tmp_path / "a.tif", labels=before, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, seed=3)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.RADIOMETRY],
    )
    test = report.by_kind(ConfounderKind.RADIOMETRY)
    assert test is not None
    assert test.verdict is ConfounderVerdict.RULED_OUT
    assert test.requirement is None


# -- sensor mismatch -------------------------------------------------------


def test_a_large_resolution_gap_makes_sensor_mismatch_likely(
    tmp_path, make_context, store, engine
):
    a = make_optical_scene(tmp_path / "fine.tif", pixel_size=10.0, seed=3)
    b = make_optical_scene(tmp_path / "coarse.tif", pixel_size=40.0, seed=4)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.SENSOR_MISMATCH],
    )
    test = report.by_kind(ConfounderKind.SENSOR_MISMATCH)
    assert test is not None
    assert test.verdict is ConfounderVerdict.LIKELY
    assert test.measured_numeric == pytest.approx(4.0, rel=0.01)


def test_a_matched_pair_rules_out_sensor_mismatch(
    tmp_path, make_context, store, engine
):
    a = make_optical_scene(tmp_path / "a.tif", seed=3)
    b = make_optical_scene(tmp_path / "b.tif", seed=4)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.SENSOR_MISMATCH],
    )
    assert report.by_kind(ConfounderKind.SENSOR_MISMATCH).verdict is (
        ConfounderVerdict.RULED_OUT
    )


# -- the shape of every result --------------------------------------------


def test_every_test_declares_its_threshold_and_its_formula(
    tmp_path, make_context, store, engine
):
    before = make_labels(builtup=(140, 30, 70, 60))
    after = make_labels(builtup=(140, 30, 70, 120))
    a = make_optical_scene(tmp_path / "a.tif", labels=before, with_scl=True, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, with_scl=True, seed=4)

    _, report, _ = run_pair(make_context, store, a.path, b.path, targets=["built-up"])

    assert len(report.tests) >= 4
    for test in report.tests:
        assert test.threshold
        assert test.explanation
        assert test.label
        assert test.question
        if test.verdict is not ConfounderVerdict.NOT_TESTED:
            assert test.formula
            assert test.method


def test_anything_not_ruled_out_names_the_data_that_would_settle_it(
    tmp_path, make_context, store, engine
):
    """A refusal is only useful if it says what would change the answer."""
    green = make_labels(water=None, builtup=None, bare=None)
    harvested = np.full_like(green, int(LandCover.BARE))
    a = make_optical_scene(tmp_path / "a.tif", labels=green, acquisition_date=MARCH)
    b = make_optical_scene(tmp_path / "b.tif", labels=harvested, acquisition_date=MAY)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["vegetation"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )
    for test in report.surviving:
        if test.verdict is ConfounderVerdict.NOT_TESTED:
            continue
        assert test.requirement, test.kind
    assert report.requirements


def test_an_untestable_confounder_is_named_not_assumed_away(
    tmp_path, make_context, store, engine
):
    before = make_labels(builtup=(140, 30, 70, 60))
    after = make_labels(builtup=(140, 30, 70, 120))
    a = make_optical_scene(tmp_path / "a.tif", labels=before, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, seed=4)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"],
        confounder_kinds=[ConfounderKind.SAR_SPECIFIC],
    )
    test = report.by_kind(ConfounderKind.SAR_SPECIFIC)
    assert test is not None
    assert test.verdict is ConfounderVerdict.NOT_TESTED
    assert test.survived
    assert ConfounderKind.SAR_SPECIFIC in report.untested_kinds


def test_the_engine_waits_for_a_finding_rather_than_declaring_itself_unusable(
    tmp_path, make_context, engine
):
    """"Has not run yet" and "cannot run here" are different answers.

    Availability is about the input; readiness is about where the run has got to.
    Conflating them made the planner refuse to schedule any tool that reads
    another's output.
    """
    a = make_optical_scene(tmp_path / "a.tif", seed=3)
    b = make_optical_scene(tmp_path / "b.tif", seed=4)
    context = make_context({ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path})

    allowed, _ = engine.can_run(context)
    assert allowed

    ready, reason = engine.upstream_ready(context)
    assert not ready
    assert "has to run first" in reason


def test_a_run_with_no_finding_is_skipped_with_that_reason(
    tmp_path, make_context, engine
):
    a = make_optical_scene(tmp_path / "a.tif", seed=3)
    b = make_optical_scene(tmp_path / "b.tif", seed=4)
    context = make_context({ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path})

    outcome = engine.run(context)
    assert not outcome.ok
    assert "has to run first" in (outcome.skipped_reason or "")


def test_the_engine_is_plannable_before_its_dependency_has_run(
    tmp_path, make_context, engine
):
    from app.core.contract import runnable_once_scheduled
    from app.core.registry import build_default_registry

    a = make_optical_scene(tmp_path / "a.tif", seed=3)
    b = make_optical_scene(tmp_path / "b.tif", seed=4)
    context = make_context({ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path})

    assert engine.depends_on() == ("change-cva-engine",)
    assert runnable_once_scheduled(
        "confounder-engine", build_default_registry(), context, planned=set()
    )


def test_a_single_image_cannot_run_the_confounder_engine(
    tmp_path, make_context, engine
):
    scene = make_optical_scene(tmp_path / "single.tif")
    context = make_context({ImageRole.SINGLE: scene.path})
    allowed, reason = engine.can_run(context)
    assert not allowed
    assert "bi_temporal_pair" in reason or "cross_modal_pair" in reason


def test_the_report_counts_what_it_ruled_out(tmp_path, make_context, store, engine):
    before = make_labels(builtup=(140, 30, 70, 60))
    after = make_labels(builtup=(140, 30, 70, 120))
    a = make_optical_scene(tmp_path / "a.tif", labels=before, with_scl=True, seed=3)
    b = make_optical_scene(tmp_path / "b.tif", labels=after, with_scl=True, seed=4)

    outcome, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["built-up"]
    )
    counted = outcome.measurement("confounders_ruled_out")
    assert counted is not None
    assert counted.value == float(len(report.ruled_out))
    assert set(counted.inputs["ruled_out"]) <= {
        test.kind.value for test in report.tests
    }


def test_the_readiness_gate_and_the_seasonality_test_agree_on_the_dates(
    tmp_path, make_context, store, engine
):
    """The calendar offset must come from the gate, not be recomputed."""
    green = make_labels(water=None, builtup=None, bare=None)
    harvested = np.full_like(green, int(LandCover.BARE))
    a = make_optical_scene(tmp_path / "a.tif", labels=green, acquisition_date=MARCH)
    b = make_optical_scene(tmp_path / "b.tif", labels=harvested, acquisition_date=MAY)

    context = make_context(
        {ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path},
        target_classes=["vegetation"],
        with_readiness=True,
    )
    gate = evaluate_readiness(context.session, context.store)

    _, report, _ = run_pair(
        make_context, store, a.path, b.path, targets=["vegetation"],
        confounder_kinds=[ConfounderKind.SEASONALITY],
    )
    test = report.by_kind(ConfounderKind.SEASONALITY)
    assert test is not None
    assert test.inputs["month_of_year_delta"] == gate.month_of_year_delta
    assert test.inputs["day_delta"] == gate.day_delta
