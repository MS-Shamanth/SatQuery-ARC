"""SAR engine and fusion tests.

The fusion tests are the important ones. Their job is to confirm that agreement
and disagreement are measured rather than asserted, that a sensor which could not
see is not scored as disagreeing, and that raw overlap never gets to stand in for
chance-corrected agreement on a scene that is mostly dry land.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.models.schemas import ImageRole
from app.tools.base import ToolError
from app.tools.fusion import (
    FAIR_AGREEMENT_IOU,
    STRONG_AGREEMENT_IOU,
    OpticalSarFusion,
    kappa_label,
)
from app.tools.indices import SpectralIndexEngine
from app.tools.sar import (
    CONVENTIONAL_WATER_DB,
    PLAUSIBLE_WATER_DB,
    SarBackscatterEngine,
    speckle_filtered_db,
    water_threshold_db,
)
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    make_sar_scene,
)

PIXEL_AREA_KM2 = 100.0 / 1_000_000.0
WATER_RECT = (20, 150, 80, 90)


@pytest.fixture
def sar_engine():
    return SarBackscatterEngine()


@pytest.fixture
def fusion_engine():
    return OpticalSarFusion()


# -- decibels and speckle -------------------------------------------------


def test_linear_power_is_converted_but_decibels_are_left_alone():
    """Taking the logarithm of a decibel value silently halves the dynamic range."""
    linear = np.ma.masked_array(np.full((40, 40), 0.01))
    decibels = np.ma.masked_array(np.full((40, 40), -20.0))

    converted, _ = speckle_filtered_db(linear, window=3)
    untouched, _ = speckle_filtered_db(decibels, window=3)

    assert float(converted.mean()) == pytest.approx(-20.0, abs=0.5)
    assert float(untouched.mean()) == pytest.approx(-20.0, abs=0.5)


def test_the_speckle_filter_reduces_variance_without_moving_the_level():
    rng = np.random.default_rng(7)
    # Multiplicative speckle on a constant surface, as SAR intensity actually is.
    power = 0.01 * rng.gamma(shape=4.0, scale=0.25, size=(120, 120))
    raw = np.ma.masked_array(power)

    filtered, recipe = speckle_filtered_db(raw, window=5)
    unfiltered, _ = speckle_filtered_db(raw, window=1)

    assert filtered.std() < unfiltered.std()
    assert float(filtered.mean()) == pytest.approx(float(unfiltered.mean()), abs=1.5)
    assert "median" in recipe


def test_the_recipe_states_that_filtering_happens_in_the_log_domain():
    filtered, recipe = speckle_filtered_db(
        np.ma.masked_array(np.full((40, 40), 0.01)), window=5
    )
    assert filtered is not None
    assert "logarithmic" in recipe


# -- the water threshold --------------------------------------------------


def test_an_implausible_automatic_threshold_is_refused(sar_engine):
    """Otsu always returns something, including on a scene with no water."""
    rng = np.random.default_rng(3)
    # Built-up only: bright everywhere, no water population at all.
    land = rng.normal(-4.0, 1.0, size=20000)

    threshold, _, note = water_threshold_db(land, "otsu")
    low, high = PLAUSIBLE_WATER_DB
    assert threshold == pytest.approx(CONVENTIONAL_WATER_DB)
    assert ("outside the plausible" in note) or ("separates the scene" in note)
    assert low <= threshold <= high


def test_a_clear_water_land_split_is_accepted(sar_engine):
    rng = np.random.default_rng(5)
    values = np.concatenate(
        [rng.normal(-22.0, 1.0, size=6000), rng.normal(-8.0, 1.0, size=14000)]
    )
    threshold, separability, note = water_threshold_db(values, "otsu")
    assert -20.0 < threshold < -12.0
    assert separability > 0.5
    assert "Otsu" in note


def test_the_fixed_threshold_cites_its_source():
    threshold, _, note = water_threshold_db(np.zeros(100), "fixed")
    assert threshold == CONVENTIONAL_WATER_DB
    assert "dB" in note


# -- radar water detection ------------------------------------------------


def test_radar_water_is_measured_against_a_known_area(
    tmp_path, make_context, sar_engine
):
    labels = make_labels(water=WATER_RECT)
    scene = make_sar_scene(tmp_path / "s.tif", labels=labels)
    context = make_context({ImageRole.SINGLE: scene.path})

    outcome = sar_engine.run(context)
    assert outcome.ok, outcome.skipped_reason

    truth = WATER_RECT[2] * WATER_RECT[3] * PIXEL_AREA_KM2
    measured = outcome.measurement("sar_water_area_km2")
    assert measured is not None
    assert measured.value == pytest.approx(truth, rel=0.2)


def test_a_scene_with_no_water_reports_none(tmp_path, make_context, sar_engine):
    labels = make_labels(water=None, builtup=(40, 40, 100, 100), bare=None)
    scene = make_sar_scene(tmp_path / "s.tif", labels=labels)
    context = make_context({ImageRole.SINGLE: scene.path})

    outcome = sar_engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert outcome.measurement("sar_water_area_km2").value < 0.2


def test_the_polarisation_substitution_is_reported_not_silent(
    tmp_path, make_context, sar_engine
):
    scene = make_sar_scene(tmp_path / "s.tif", labels=make_labels(water=WATER_RECT))
    context = make_context(
        {ImageRole.SINGLE: scene.path}, parameters={"polarisation": "hh"}
    )
    outcome = sar_engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert outcome.parameters["polarisation"] in {"vv", "vh"}
    assert any("not present in this product" in note for note in outcome.notes)


def test_the_engine_says_what_it_cannot_separate(tmp_path, make_context, sar_engine):
    """Radar shadow is dark too, and this engine has no terrain model."""
    scene = make_sar_scene(tmp_path / "s.tif", labels=make_labels(water=WATER_RECT))
    context = make_context({ImageRole.SINGLE: scene.path})
    outcome = sar_engine.run(context)
    assert any("Radar shadow" in note for note in outcome.notes)


def test_an_optical_only_session_cannot_run_the_radar_engine(
    tmp_path, make_context, sar_engine
):
    scene = make_optical_scene(tmp_path / "o.tif")
    context = make_context({ImageRole.SINGLE: scene.path})
    allowed, reason = sar_engine.can_run(context)
    assert not allowed
    assert "sar" in reason


def test_every_radar_measurement_carries_its_provenance(
    tmp_path, make_context, sar_engine
):
    scene = make_sar_scene(tmp_path / "s.tif", labels=make_labels(water=WATER_RECT))
    context = make_context({ImageRole.SINGLE: scene.path})
    outcome = sar_engine.run(context)

    assert outcome.measurements
    for measurement in outcome.measurements:
        assert measurement.formula
        assert measurement.source_tool == "sar-backscatter-engine"


# -- fusion ---------------------------------------------------------------


def cross_modal(tmp_path, make_context, labels_optical, labels_sar, **kwargs):
    """A co-registered optical and radar pair on one grid."""
    optical = make_optical_scene(
        tmp_path / "opt.tif", labels=labels_optical, seed=3, **kwargs
    )
    sar = make_sar_scene(tmp_path / "sar.tif", labels=labels_sar, seed=11)
    return make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path},
        target_classes=["water"],
    )


def fuse(context):
    index = SpectralIndexEngine().run(
        replace(context, parameters={"indices": ["MNDWI"]})
    )
    assert index.ok, index.skipped_reason
    radar = SarBackscatterEngine().run(replace(context, parameters={}))
    assert radar.ok, radar.skipped_reason

    probe = replace(
        context,
        parameters={},
        upstream={
            "spectral-index-engine": index,
            "sar-backscatter-engine": radar,
        },
    )
    outcome = OpticalSarFusion().run(probe)
    return outcome, index, radar


def test_two_sensors_seeing_the_same_water_agree_strongly(tmp_path, make_context):
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)

    outcome, _, _ = fuse(context)
    assert outcome.ok, outcome.skipped_reason

    iou = outcome.measurement("agreement_iou")
    kappa = outcome.measurement("agreement_kappa")
    assert iou is not None and kappa is not None
    assert iou.value > STRONG_AGREEMENT_IOU
    assert kappa.value > 0.6
    assert outcome.measurement("agreed_water_km2").value > 0.5


def test_the_disagreement_map_separates_who_saw_what(tmp_path, make_context):
    """Radar sees water the optical instrument does not, and it is drawn separately."""
    optical_labels = make_labels(water=WATER_RECT)
    # The radar scene has an additional water body the optical scene shows as bare.
    sar_labels = make_labels(water=WATER_RECT)
    sar_labels[150:210, 20:80] = int(LandCover.WATER)

    context = cross_modal(tmp_path, make_context, optical_labels, sar_labels)
    outcome, _, _ = fuse(context)
    assert outcome.ok, outcome.skipped_reason

    agree = outcome.mask("fusion_agree_water")
    radar_only = outcome.mask("fusion_disagree_sar_only")
    optical_only = outcome.mask("fusion_disagree_optical_only")
    assert agree is not None and radar_only is not None and optical_only is not None

    assert agree.pixel_count > 0
    assert radar_only.pixel_count > 0
    # The extra radar water must land in the radar-only layer, not in agreement.
    assert radar_only.array[150:210, 20:80].any()
    assert not agree.array[150:210, 20:80].any()
    assert outcome.measurement("sar_only_km2").value > 0.2


def test_the_three_disagreement_layers_are_mutually_exclusive(tmp_path, make_context):
    labels = make_labels(water=WATER_RECT)
    sar_labels = labels.copy()
    sar_labels[150:200, 20:70] = int(LandCover.WATER)
    context = cross_modal(tmp_path, make_context, labels, sar_labels)

    outcome, _, _ = fuse(context)
    agree = outcome.mask("fusion_agree_water").array
    radar_only = outcome.mask("fusion_disagree_sar_only").array
    optical_only = outcome.mask("fusion_disagree_optical_only").array

    assert not (agree & radar_only).any()
    assert not (agree & optical_only).any()
    assert not (radar_only & optical_only).any()


def test_chance_corrected_agreement_is_reported_alongside_raw_overlap(
    tmp_path, make_context
):
    """Both sensors calling dry land dry is not agreement worth crediting."""
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)
    outcome, _, _ = fuse(context)

    iou = outcome.measurement("agreement_iou")
    kappa = outcome.measurement("agreement_kappa")
    assert iou is not None and kappa is not None
    assert "by accident" in kappa.method
    assert "chance" in kappa.label
    assert kappa.inputs["interpretation"] == kappa_label(kappa.value)
    # Raw overlap already excludes the shared negatives, so the two should be close
    # on a clean pair; the point is that both are shown.
    assert abs(iou.value - kappa.value) < 0.4


def test_a_sensor_that_could_not_see_is_not_scored_as_disagreeing(
    tmp_path, make_context
):
    """Cloud is not an optical error; it is an optical absence."""
    clear = make_labels(water=WATER_RECT)
    clouded = make_labels(water=WATER_RECT, cloud=(0, 0, 90, 256))

    with_cloud = cross_modal(
        tmp_path / "cloudy", make_context, clouded, clear, with_scl=True
    )
    cloudy_outcome, _, _ = fuse(with_cloud)
    assert cloudy_outcome.ok, cloudy_outcome.skipped_reason

    # The cloud band carries no optical claim, so nothing there may appear in the
    # optical-only layer.
    optical_only = cloudy_outcome.mask("fusion_disagree_optical_only")
    assert optical_only is not None
    assert not optical_only.array[:90, :].any()

    blind = cloudy_outcome.mask("fusion_sar_under_cloud")
    if blind is not None:
        assert cloudy_outcome.notes


def test_cloud_is_offered_as_the_explanation_when_it_fits(tmp_path, make_context):
    """The disagreement being cloud is a better outcome than a sensor conflict."""
    # Radar sees water under the cloud band; the optical instrument sees nothing
    # there because it is blocked.
    clouded = make_labels(water=WATER_RECT, cloud=(0, 0, 90, 256))
    sar_labels = make_labels(water=WATER_RECT)
    sar_labels[20:80, 20:230] = int(LandCover.WATER)

    context = cross_modal(
        tmp_path, make_context, clouded, sar_labels, with_scl=True
    )
    outcome, _, _ = fuse(context)
    assert outcome.ok, outcome.skipped_reason

    share = outcome.measurement("sar_only_under_cloud_share")
    if share is not None:
        assert 0.0 <= share.value <= 1.0
        assert any("optical cloud" in note for note in outcome.notes)


def test_fusion_refuses_masks_on_different_grids(tmp_path, make_context):
    """Comparing arrays of different shapes would compare resampling, not sensors."""
    optical = make_optical_scene(
        tmp_path / "opt.tif", labels=make_labels(water=WATER_RECT), pixel_size=10.0
    )
    sar = make_sar_scene(
        tmp_path / "sar.tif", labels=make_labels(width=128, height=128), width=128,
        height=128, pixel_size=20.0,
    )
    context = make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path},
        target_classes=["water"],
    )

    index = SpectralIndexEngine().run(
        replace(context, parameters={"indices": ["MNDWI"]})
    )
    radar = SarBackscatterEngine().run(replace(context, parameters={}))
    probe = replace(
        context,
        parameters={},
        upstream={
            "spectral-index-engine": index,
            "sar-backscatter-engine": radar,
        },
    )
    outcome = OpticalSarFusion().run(probe)
    assert not outcome.ok
    assert "different grids" in (outcome.skipped_reason or "")


def test_fusion_waits_for_both_sensors_rather_than_declaring_itself_unusable(
    tmp_path, make_context, fusion_engine
):
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)

    allowed, _ = fusion_engine.can_run(context)
    assert allowed

    ready, reason = fusion_engine.upstream_ready(context)
    assert not ready
    assert "has to run first" in reason


def test_fusion_needs_a_cross_modal_pair(tmp_path, make_context, fusion_engine):
    scene = make_optical_scene(tmp_path / "o.tif")
    context = make_context({ImageRole.SINGLE: scene.path})
    allowed, reason = fusion_engine.can_run(context)
    assert not allowed
    assert "cross_modal_pair" in reason


def test_the_agreement_reading_is_written_for_a_reader(tmp_path, make_context):
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)
    outcome, _, _ = fuse(context)

    assert any("unrelated physics" in note for note in outcome.notes)
    assert FAIR_AGREEMENT_IOU < STRONG_AGREEMENT_IOU


def test_fusion_measurements_name_both_source_masks(tmp_path, make_context):
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)
    outcome, _, _ = fuse(context)

    iou = outcome.measurement("agreement_iou")
    assert iou is not None
    assert iou.inputs["radar_mask"] == "sar_water"
    assert "mndwi" in iou.inputs["optical_mask"].lower()
    for measurement in outcome.measurements:
        assert measurement.source_tool == "optical-sar-fusion"
        assert measurement.formula


def test_an_empty_optical_result_is_not_an_error(tmp_path, make_context):
    """No water found is a result, and the comparison still has to work."""
    dry = make_labels(water=None, builtup=(40, 40, 60, 60), bare=None)
    context = cross_modal(tmp_path, make_context, dry, dry)
    outcome, _, _ = fuse(context)

    assert outcome.ok, outcome.skipped_reason
    assert outcome.measurement("agreed_water_km2").value < 0.2


def test_a_missing_optical_mask_is_reported_rather_than_guessed(
    tmp_path, make_context, fusion_engine
):
    labels = make_labels(water=WATER_RECT)
    context = cross_modal(tmp_path, make_context, labels, labels)
    radar = SarBackscatterEngine().run(replace(context, parameters={}))

    probe = replace(
        context, parameters={}, upstream={"sar-backscatter-engine": radar}
    )
    ready, reason = fusion_engine.upstream_ready(probe)
    assert not ready
    assert "optical water mask" in reason


def test_the_radar_engine_refuses_a_product_with_no_usable_polarisation(
    tmp_path, make_context, sar_engine
):
    scene = make_sar_scene(tmp_path / "s.tif", labels=make_labels(water=WATER_RECT))
    context = make_context({ImageRole.SINGLE: scene.path})
    meta = context.metadata(ImageRole.SINGLE)

    class Blank:
        resolved_roles: list = []

        @staticmethod
        def band_index(_role):
            return None

    with pytest.raises(ToolError, match="No usable polarisation"):
        sar_engine._band_for(Blank(), "vv")
    assert meta is not None


def test_an_unmatched_mask_request_is_worked_around_and_reported(
    make_context, tmp_path, tool_registry
) -> None:
    """A parameter that cannot be honoured must not disable the comparison.

    The index engine names a mask after both the index and the image role, as in
    ``ndwi_mask.optical``. A plan that asked for ``ndwi`` exactly matched nothing,
    so the fusion engine skipped itself and the entire optical-versus-radar
    comparison silently did not happen on a scene that exists to show it. Now the
    name is matched loosely, then fallen back on, and the substitution is stated.
    """
    labels = make_labels(96, 96, water=(10, 10, 40, 40))
    optical = make_optical_scene(
        tmp_path / "o.tif", width=96, height=96, labels=labels, with_scl=True
    )
    sar = make_sar_scene(tmp_path / "s.tif", width=96, height=96, labels=labels)
    context = make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path},
        with_readiness=True,
    )
    context.target_classes = ["water"]
    # Deliberately the bare index name rather than the mask key.
    context.parameters = {"optical_mask": "ndwi"}

    index_outcome = SpectralIndexEngine().run(context)
    assert index_outcome.ok
    context.upstream[SpectralIndexEngine.name] = index_outcome
    sar_outcome = SarBackscatterEngine().run(context)
    assert sar_outcome.ok
    context.upstream[SarBackscatterEngine.name] = sar_outcome

    fused = OpticalSarFusion().run(context)

    assert fused.ok, fused.skipped_reason
    assert fused.mask("fusion_agree_water") is not None
    used = fused.parameters["optical_mask"]
    assert "ndwi" in used.lower()
    assert used != "ndwi", "the mask key is more specific than the index name"
    assert any("asked for the optical mask" in note for note in fused.notes)


def test_a_mask_request_that_matches_nothing_at_all_still_finds_water(
    make_context, tmp_path, tool_registry
) -> None:
    """The fallback is to the water mask, not to nothing."""
    labels = make_labels(96, 96, water=(10, 10, 40, 40))
    optical = make_optical_scene(
        tmp_path / "o2.tif", width=96, height=96, labels=labels, with_scl=True
    )
    sar = make_sar_scene(tmp_path / "s2.tif", width=96, height=96, labels=labels)
    context = make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path},
        with_readiness=True,
    )
    context.target_classes = ["water"]
    context.parameters = {"optical_mask": "no_such_mask_anywhere"}

    index_outcome = SpectralIndexEngine().run(context)
    context.upstream[SpectralIndexEngine.name] = index_outcome
    sar_outcome = SarBackscatterEngine().run(context)
    context.upstream[SarBackscatterEngine.name] = sar_outcome

    fused = OpticalSarFusion().run(context)
    assert fused.ok, fused.skipped_reason
    assert any("no tool in this run produced" in note for note in fused.notes)
