"""Change engine tests.

Bi-temporal change detection is easy to get confidently wrong. These tests are
built around the specific ways it lies: measuring resampling instead of change,
counting cloud as change, and collapsing gain and loss into a net that hides both.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas import ImageRole
from app.tools.base import ToolError
from app.tools.change import (
    ChangeCvaEngine,
    build_common_grid,
    resolve_change_index,
)
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    make_sar_scene,
)

PIXEL_AREA_KM2 = 100.0 / 1_000_000.0


@pytest.fixture
def engine():
    return ChangeCvaEngine()


def pair(tmp_path, labels_a, labels_b, **kwargs):
    """Two dated scenes on an identical grid."""
    a = make_optical_scene(tmp_path / "a.tif", labels=labels_a, seed=3, **kwargs)
    b = make_optical_scene(tmp_path / "b.tif", labels=labels_b, seed=4, **kwargs)
    return a, b


def context_for(make_context, a, b, **kwargs):
    return make_context(
        {ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path}, **kwargs
    )


# -- the common grid ------------------------------------------------------


def test_both_dates_are_read_onto_one_grid(tmp_path, make_context):
    a, b = pair(tmp_path, make_labels(), make_labels())
    context = context_for(make_context, a, b)
    grid = build_common_grid(context, ImageRole.DATE_A, ImageRole.DATE_B)
    assert grid.shape == (256, 256)
    assert grid.pixel_m == pytest.approx(10.0)


def test_the_finer_resolution_is_kept_not_the_coarser(tmp_path, make_context):
    """Degrading to the coarser sensor would throw away real detail."""
    a = make_optical_scene(tmp_path / "fine.tif", pixel_size=10.0)
    b = make_optical_scene(tmp_path / "coarse.tif", pixel_size=20.0)
    context = context_for(make_context, a, b)
    grid = build_common_grid(context, ImageRole.DATE_A, ImageRole.DATE_B)
    assert grid.pixel_m == pytest.approx(10.0)
    assert "upsampled" in grid.note


def test_mismatched_projections_are_refused_rather_than_compared(
    tmp_path, make_context
):
    a = make_optical_scene(tmp_path / "a.tif", epsg=32643)
    b = make_optical_scene(tmp_path / "b.tif", epsg=32644)
    context = context_for(make_context, a, b)
    with pytest.raises(ToolError, match="different projections"):
        build_common_grid(context, ImageRole.DATE_A, ImageRole.DATE_B)


def test_non_overlapping_dates_are_refused(tmp_path, make_context):
    a = make_optical_scene(tmp_path / "a.tif", origin=(600000.0, 2000000.0))
    b = make_optical_scene(tmp_path / "b.tif", origin=(900000.0, 2000000.0))
    context = context_for(make_context, a, b)
    with pytest.raises(ToolError, match="do not overlap"):
        build_common_grid(context, ImageRole.DATE_A, ImageRole.DATE_B)


def test_only_the_overlap_is_measured(tmp_path, make_context):
    """Half a scene of nodata is not half a scene of change."""
    a = make_optical_scene(tmp_path / "a.tif", origin=(600000.0, 2000000.0))
    b = make_optical_scene(tmp_path / "b.tif", origin=(600000.0 + 1280.0, 2000000.0))
    context = context_for(make_context, a, b)
    grid = build_common_grid(context, ImageRole.DATE_A, ImageRole.DATE_B)
    assert grid.width == 128
    assert grid.height == 256


# -- index selection -----------------------------------------------------


def test_the_target_class_chooses_the_index(tmp_path, make_context):
    a, b = pair(tmp_path, make_labels(), make_labels())
    context = context_for(make_context, a, b)
    name, why = resolve_change_index(["built-up"], context)
    assert name == "NDBI"
    assert "built-up" in why


def test_an_unrecognised_target_falls_back_to_ndvi_and_says_so(
    tmp_path, make_context
):
    a, b = pair(tmp_path, make_labels(), make_labels())
    context = context_for(make_context, a, b)
    name, why = resolve_change_index(["sentiment"], context)
    assert name == "NDVI"
    assert "no recognised target class" in why


def test_water_falls_back_to_ndwi_when_swir_is_absent(tmp_path, make_context):
    bands = ("blue", "green", "red", "nir", "swir16", "swir22")
    a = make_optical_scene(tmp_path / "a.tif", band_names=bands[:4] + ("x", "y"))
    b = make_optical_scene(tmp_path / "b.tif", band_names=bands[:4] + ("x", "y"))
    context = context_for(make_context, a, b)
    name, why = resolve_change_index(["water"], context)
    assert name == "NDWI"
    assert "no SWIR" in why


# -- measured change against ground truth --------------------------------


def test_built_up_growth_is_measured_against_known_areas(tmp_path, make_context, engine):
    """A rectangle of built-up is added; the measured gain must match its area."""
    before = make_labels(builtup=(140, 30, 70, 60), water=None, bare=None)
    after = make_labels(builtup=(140, 30, 70, 120), water=None, bare=None)
    a, b = pair(tmp_path, before, after)
    context = context_for(
        make_context, a, b, target_classes=["built-up"], with_readiness=True
    )

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason

    truth_before = 70 * 60 * PIXEL_AREA_KM2
    truth_after = 70 * 120 * PIXEL_AREA_KM2

    measured_before = outcome.measurement("area_km2_before").value
    measured_after = outcome.measurement("area_km2_after").value
    # NDBI separates built-up from vegetation cleanly in the fixture signatures,
    # so the tolerance only has to absorb the noise the fixture adds.
    assert measured_before == pytest.approx(truth_before, rel=0.15)
    assert measured_after == pytest.approx(truth_after, rel=0.15)

    gain = outcome.measurement("gain_km2").value
    assert gain == pytest.approx(truth_after - truth_before, rel=0.2)
    assert outcome.measurement("percentage_change").value > 50.0


def test_gain_and_loss_are_reported_separately_not_only_as_net(
    tmp_path, make_context, engine
):
    """Equal gain and loss nets to zero, and calling that unchanged is false."""
    before = make_labels(builtup=(20, 20, 60, 60), water=None, bare=None)
    after = make_labels(builtup=(150, 150, 60, 60), water=None, bare=None)
    a, b = pair(tmp_path, before, after)
    context = context_for(make_context, a, b, target_classes=["built-up"])

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason

    gain = outcome.measurement("gain_km2").value
    loss = outcome.measurement("loss_km2").value
    net = outcome.measurement("net_change_km2").value

    assert gain > 0.2
    assert loss > 0.2
    assert net == pytest.approx(gain - loss, abs=1e-9)
    assert abs(net) < 0.1 * (gain + loss)
    # The net measurement must carry the warning about reading it alone.
    assert "never instead of them" in outcome.measurement("net_change_km2").method


def test_an_unchanged_pair_reports_no_meaningful_change(
    tmp_path, make_context, engine
):
    labels = make_labels()
    a, b = pair(tmp_path, labels, labels)
    context = context_for(make_context, a, b, target_classes=["built-up"])

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert abs(outcome.measurement("area_change_km2").value) < 0.05
    assert abs(outcome.measurement("delta_mean").value) < 0.05


# -- observability -------------------------------------------------------


def test_cloud_on_either_date_is_excluded_from_every_figure(
    tmp_path, make_context, engine
):
    """Cloud on one date and clear on the other is not change."""
    labels = make_labels()
    clear = make_optical_scene(
        tmp_path / "clear.tif", labels=labels, with_scl=True, seed=3
    )
    cloudy = make_optical_scene(
        tmp_path / "cloudy.tif",
        labels=make_labels(cloud=(0, 0, 90, 256)),
        with_scl=True,
        seed=4,
    )
    context = context_for(
        make_context, clear, cloudy, target_classes=["built-up"]
    )

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason

    observable = outcome.measurement("observable_fraction")
    assert observable.value < 0.7
    assert observable.inputs["cloud_fraction_either_date"] > 0.3
    assert any("cloudy on at least one date" in note for note in outcome.notes)

    # The excluded band must not appear as change in any mask.
    gain = outcome.mask("change_gain")
    assert gain is not None
    assert not gain.array[:90, :].any()


def test_cloud_is_kept_when_exclusion_is_turned_off(tmp_path, make_context, engine):
    labels = make_labels()
    clear = make_optical_scene(tmp_path / "c.tif", labels=labels, with_scl=True)
    cloudy = make_optical_scene(
        tmp_path / "d.tif", labels=make_labels(cloud=(0, 0, 60, 256)), with_scl=True
    )
    context = context_for(
        make_context,
        clear,
        cloudy,
        parameters={"exclude_cloud": False},
        target_classes=["built-up"],
    )
    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert (
        outcome.measurement("observable_fraction").inputs[
            "cloud_fraction_either_date"
        ]
        == 0.0
    )


def test_a_mostly_unobservable_pair_is_refused_rather_than_measured(
    tmp_path, make_context, engine
):
    clear = make_optical_scene(tmp_path / "c.tif", with_scl=True)
    cloudy = make_optical_scene(
        tmp_path / "d.tif", labels=make_labels(cloud=(0, 0, 250, 256)), with_scl=True
    )
    context = context_for(make_context, clear, cloudy, target_classes=["built-up"])
    outcome = engine.run(context)
    assert not outcome.ok
    assert "observable on both dates" in (outcome.skipped_reason or "")


# -- thresholds ----------------------------------------------------------


def test_one_threshold_serves_both_dates(tmp_path, make_context, engine):
    """A threshold that moves between dates makes the boundary look like change."""
    labels = make_labels()
    a, b = pair(tmp_path, labels, labels)
    context = context_for(make_context, a, b, target_classes=["built-up"])
    outcome = engine.run(context)

    before = outcome.mask("change_gain")
    assert before is not None
    assert before.threshold is not None
    loss = outcome.mask("change_loss")
    assert loss is not None
    assert loss.threshold == before.threshold


def test_otsu_pools_both_dates_before_choosing(tmp_path, make_context, engine):
    labels = make_labels()
    a, b = pair(tmp_path, labels, labels)
    context = context_for(
        make_context,
        a,
        b,
        parameters={"class_threshold_method": "otsu"},
        target_classes=["built-up"],
    )
    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert any("pooled" in note for note in outcome.notes)


def test_a_change_below_the_noise_step_is_not_reported_as_change(
    tmp_path, make_context, engine
):
    """An identical pair with a high step must yield essentially nothing.

    Not exactly nothing: NDVI over water is the ratio of two near-zero bands, so
    its value there is unstable and a handful of water pixels swing past even a
    0.9 step purely from sensor noise. A few pixels in 65,536 is the honest
    outcome; anything more would mean the threshold was not doing its job.
    """
    labels = make_labels()
    a, b = pair(tmp_path, labels, labels)
    context = context_for(
        make_context, a, b, parameters={"delta_threshold": 0.9},
        target_classes=["vegetation"],
    )
    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    increase = outcome.mask("delta_ndvi_increase")
    decrease = outcome.mask("delta_ndvi_decrease")
    assert increase is not None and decrease is not None
    noise = increase.pixel_count + decrease.pixel_count
    assert noise / increase.array.size < 0.0005, noise


# -- change vector analysis ----------------------------------------------


def test_the_change_vector_finds_change_the_index_alone_would_miss(
    tmp_path, make_context, engine
):
    """Bare ground replacing built-up barely moves NDBI but moves reflectance."""
    before = make_labels(builtup=(40, 40, 120, 120), water=None, bare=None)
    after = make_labels(builtup=None, water=None, bare=(40, 40, 120, 120))
    a, b = pair(tmp_path, before, after)
    context = context_for(make_context, a, b, target_classes=["built-up"])

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    magnitude = outcome.mask("change_magnitude")
    assert magnitude is not None
    assert magnitude.pixel_count > 0
    assert outcome.measurement("cva_mean_magnitude").value > 0.0


def test_the_change_vector_is_skipped_with_a_reason_when_bands_are_missing(
    tmp_path, make_context, engine
):
    bands = ("nir", "red", "a", "b", "c", "d")
    a = make_optical_scene(tmp_path / "a.tif", band_names=bands)
    b = make_optical_scene(tmp_path / "b.tif", band_names=bands)
    context = context_for(make_context, a, b, target_classes=["vegetation"])

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert outcome.mask("change_magnitude") is None
    assert any("change vector" in note.lower() for note in outcome.notes)


# -- provenance ----------------------------------------------------------


def test_every_measurement_carries_a_formula_and_its_inputs(
    tmp_path, make_context, engine
):
    a, b = pair(tmp_path, make_labels(), make_labels(builtup=(140, 30, 70, 120)))
    context = context_for(make_context, a, b, target_classes=["built-up"])
    outcome = engine.run(context)

    assert outcome.measurements
    for measurement in outcome.measurements:
        assert measurement.formula
        assert measurement.source_tool == "change-cva-engine"
        assert measurement.unit


def test_a_percentage_with_no_denominator_is_omitted_not_invented(
    tmp_path, make_context, engine
):
    """Nothing on the first date means the percentage is undefined, not zero."""
    before = make_labels(water=None, builtup=None, bare=None)
    after = make_labels(water=(20, 20, 80, 80), builtup=None, bare=None)
    a, b = pair(tmp_path, before, after)
    context = context_for(make_context, a, b, target_classes=["water"])

    outcome = engine.run(context)
    assert outcome.ok, outcome.skipped_reason
    assert outcome.measurement("area_km2_before").value == pytest.approx(0.0, abs=1e-6)
    assert outcome.measurement("percentage_change") is None
    assert any("no denominator" in note for note in outcome.notes)
    # The absolute areas are still reported, so the finding is not lost.
    assert outcome.measurement("area_km2_after").value > 0.5


# -- requirements --------------------------------------------------------


def test_a_single_image_cannot_run_the_change_engine(tmp_path, make_context, engine):
    scene = make_optical_scene(tmp_path / "single.tif")
    context = make_context({ImageRole.SINGLE: scene.path})
    allowed, reason = engine.can_run(context)
    assert not allowed
    assert "bi_temporal_pair" in reason


def test_a_sar_pair_is_declined_with_a_reason(tmp_path, make_context, engine):
    a = make_sar_scene(tmp_path / "a.tif")
    b = make_sar_scene(tmp_path / "b.tif")
    context = context_for(make_context, a, b)
    allowed, reason = engine.can_run(context)
    assert not allowed
    assert "optical" in reason


def test_masks_and_measurements_agree_on_area(tmp_path, make_context, engine):
    """The drawn layer and the quoted number must describe the same pixels."""
    a, b = pair(tmp_path, make_labels(), make_labels(builtup=(140, 30, 70, 120)))
    context = context_for(make_context, a, b, target_classes=["built-up"])
    outcome = engine.run(context)

    gain_mask = outcome.mask("change_gain")
    assert gain_mask is not None
    assert gain_mask.area_km2() == pytest.approx(
        outcome.measurement("gain_km2").value, abs=1e-9
    )


def test_unobservable_pixels_are_excluded_from_the_mask_denominator(
    tmp_path, make_context, engine
):
    clear = make_optical_scene(tmp_path / "c.tif", with_scl=True, seed=3)
    cloudy = make_optical_scene(
        tmp_path / "d.tif", labels=make_labels(cloud=(0, 0, 80, 256)), with_scl=True,
        seed=4,
    )
    context = context_for(make_context, clear, cloudy, target_classes=["built-up"])
    outcome = engine.run(context)

    gain = outcome.mask("change_gain")
    assert gain is not None
    summary = gain.summary()
    assert summary.total_pixels < 256 * 256
    assert summary.total_pixels == int(np.count_nonzero(~gain.invalid))
