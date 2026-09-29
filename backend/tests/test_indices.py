"""Spectral index tests, checked against exactly known class areas.

The central test is the known-answer one: a scene is generated from a label map
whose water rectangle is an exact pixel count, and NDWI thresholding must recover
that area. If the index maths, the reflectance scaling, the threshold, or the area
computation drifts, that test fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas import BandRole, ImageRole, ToolImplementation
from app.tools.base import ToolError
from app.tools.indices import (
    INDEX_DEFINITIONS,
    TARGET_TO_INDEX,
    WEAK_SEPARABILITY,
    SpectralIndexEngine,
    compute_index,
    normalised_difference,
    otsu_threshold,
)
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    make_sar_scene,
)


# ---------------------------------------------------------------------------
# The maths
# ---------------------------------------------------------------------------


def test_normalised_difference_matches_the_definition() -> None:
    a = np.ma.masked_array([0.4, 0.1, 0.5], mask=False)
    b = np.ma.masked_array([0.1, 0.4, 0.5], mask=False)

    result = normalised_difference(a, b)

    assert result[0] == pytest.approx(0.3 / 0.5)
    assert result[1] == pytest.approx(-0.3 / 0.5)
    assert result[2] == pytest.approx(0.0)


def test_zero_denominator_is_masked_not_treated_as_zero() -> None:
    """Both bands reading zero is nodata, not an index value of zero."""
    a = np.ma.masked_array([0.0, 0.3], mask=False)
    b = np.ma.masked_array([0.0, 0.1], mask=False)

    result = normalised_difference(a, b)

    assert np.ma.is_masked(result[0])
    assert result[1] == pytest.approx(0.5)


def test_index_definitions_use_the_conventional_band_pairs() -> None:
    assert INDEX_DEFINITIONS["NDVI"].required_roles() == (BandRole.NIR, BandRole.RED)
    assert INDEX_DEFINITIONS["NDWI"].required_roles() == (BandRole.GREEN, BandRole.NIR)
    assert INDEX_DEFINITIONS["MNDWI"].required_roles() == (
        BandRole.GREEN,
        BandRole.SWIR16,
    )
    assert INDEX_DEFINITIONS["NDBI"].required_roles() == (BandRole.SWIR16, BandRole.NIR)
    assert INDEX_DEFINITIONS["NDMI"].required_roles() == (BandRole.NIR, BandRole.SWIR16)
    # Each definition states its published source.
    for definition in INDEX_DEFINITIONS.values():
        assert definition.reference
        assert definition.formula.startswith(definition.key)


def test_target_vocabulary_prefers_mndwi_for_water() -> None:
    """NDWI marks urban shadow as water; MNDWI is the published remedy."""
    assert TARGET_TO_INDEX["water body"] == "MNDWI"
    assert TARGET_TO_INDEX["reservoir"] == "MNDWI"
    assert TARGET_TO_INDEX["flooded"] == "MNDWI"
    assert TARGET_TO_INDEX["urban"] == "NDBI"
    assert TARGET_TO_INDEX["cropland"] == "NDVI"


def test_mndwi_pushes_built_up_further_from_the_water_threshold_than_ndwi(
    make_context, tmp_path
) -> None:
    """The reason water grounding prefers MNDWI.

    Substituting SWIR for NIR lowers the index over built-up surfaces, widening
    the margin between a city and open water. Observed on the real Patna scene,
    where NDWI speckled the urban area with false water.
    """
    labels = make_labels(
        200, 200, water=(0, 0, 60, 200), builtup=(60, 0, 60, 200), bare=None
    )
    scene = make_optical_scene(
        tmp_path / "city_water.tif", width=200, height=200, labels=labels
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    ndwi = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDWI"])
    mndwi = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["MNDWI"])

    builtup = slice(60, 120)
    ndwi_builtup = float(np.nanmean(np.ma.filled(ndwi.values, np.nan)[builtup]))
    mndwi_builtup = float(np.nanmean(np.ma.filled(mndwi.values, np.nan)[builtup]))
    mndwi_water = float(np.nanmean(np.ma.filled(mndwi.values, np.nan)[:60]))

    assert mndwi_builtup < ndwi_builtup < 0
    # Water stays clearly positive, so the margin widens rather than shifting.
    assert mndwi_water > 0.5
    assert mndwi_water - mndwi_builtup > ndwi_water_margin(ndwi)


def ndwi_water_margin(ndwi) -> float:
    values = np.ma.filled(ndwi.values, np.nan)
    return float(np.nanmean(values[:60]) - np.nanmean(values[60:120]))


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------


def test_otsu_separates_two_clear_populations_with_high_separability() -> None:
    rng = np.random.default_rng(1)
    values = np.concatenate(
        [rng.normal(-0.7, 0.05, 5000), rng.normal(0.5, 0.05, 5000)]
    )

    threshold, separability = otsu_threshold(values)

    assert -0.7 < threshold < 0.5
    assert separability > 0.9


def test_otsu_reports_zero_separability_on_a_single_population() -> None:
    """Otsu always returns a number; separability is what says it means little.

    A Gaussian cut at its optimum yields a raw between-class variance ratio of
    2/pi, about 0.64, purely from being halved. Rescaling against that floor is
    what makes 0.0 mean "this threshold separates nothing".
    """
    rng = np.random.default_rng(2)
    values = rng.normal(0.2, 0.05, 10000)

    threshold, separability = otsu_threshold(values)

    assert np.isfinite(threshold)
    assert separability == pytest.approx(0.0, abs=0.05)
    assert separability < WEAK_SEPARABILITY


def test_separability_scales_with_how_far_apart_two_populations_are() -> None:
    rng = np.random.default_rng(9)
    scores = []
    for gap in (0.05, 0.3, 1.2):
        values = np.concatenate(
            [rng.normal(0.0, 0.1, 4000), rng.normal(gap, 0.1, 4000)]
        )
        scores.append(otsu_threshold(values)[1])

    assert scores[0] < scores[1] < scores[2]
    assert scores[2] > 0.9


def test_otsu_refuses_a_constant_or_tiny_sample() -> None:
    """A uniform index must be refused, not thresholded at its own value.

    numpy reports a residual variance around 1e-33 for a constant array rather
    than exactly zero, so an equality check against zero would let this through
    and return a meaningless threshold.
    """
    with pytest.raises(ToolError, match="constant"):
        otsu_threshold(np.full(500, 0.3))
    with pytest.raises(ToolError, match="constant"):
        otsu_threshold(np.zeros(500))
    with pytest.raises(ToolError, match="Too few"):
        otsu_threshold(np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Known-answer: measured area must match the generated geometry
# ---------------------------------------------------------------------------


def test_ndwi_recovers_an_exactly_known_water_area(make_context, tmp_path) -> None:
    """The headline guarantee: the reported area is the real area.

    The fixture places water in a 120 x 150 pixel rectangle. At 10 m that is
    18 000 pixels and exactly 1.8 km2.
    """
    labels = make_labels(
        400, 400, water=(40, 60, 120, 150), builtup=None, bare=None
    )
    scene = make_optical_scene(
        tmp_path / "water.tif", width=400, height=400, labels=labels
    )
    expected_pixels = 120 * 150
    assert scene.pixels(LandCover.WATER) == expected_pixels
    assert scene.area_km2(LandCover.WATER) == pytest.approx(1.8)

    context = make_context({ImageRole.SINGLE: scene.path})
    result = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDWI"])

    # The threshold must land between the water and vegetation populations.
    assert -0.7 < result.threshold < 0.49
    assert result.separability > 0.85

    detected = int(np.count_nonzero(result.mask))
    assert detected == pytest.approx(expected_pixels, rel=0.02)

    area_km2 = detected * 100 / 1e6
    assert area_km2 == pytest.approx(1.8, rel=0.02)

    # And it found the right pixels, not merely the right count.
    truth = scene.labels == int(LandCover.WATER)
    overlap = np.count_nonzero(result.mask & truth) / np.count_nonzero(truth)
    assert overlap > 0.98


def test_ndbi_recovers_a_known_built_up_area(make_context, tmp_path) -> None:
    labels = make_labels(
        300, 300, water=None, builtup=(50, 50, 100, 120), bare=None
    )
    scene = make_optical_scene(
        tmp_path / "builtup.tif", width=300, height=300, labels=labels
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    result = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDBI"])

    expected = 100 * 120
    assert int(np.count_nonzero(result.mask)) == pytest.approx(expected, rel=0.03)
    truth = scene.labels == int(LandCover.BUILTUP)
    overlap = np.count_nonzero(result.mask & truth) / np.count_nonzero(truth)
    assert overlap > 0.95


def test_ndvi_is_positive_over_vegetation_and_negative_over_water(
    make_context, tmp_path
) -> None:
    labels = make_labels(200, 200, water=(0, 0, 100, 200), builtup=None, bare=None)
    scene = make_optical_scene(
        tmp_path / "half.tif", width=200, height=200, labels=labels
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    result = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDVI"])
    values = np.ma.filled(result.values, np.nan)

    assert float(np.nanmean(values[:100])) < 0      # water
    assert float(np.nanmean(values[100:])) > 0.7    # vegetation


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------


def test_missing_swir_names_the_band_it_needs(make_context, tmp_path) -> None:
    """Availability gating is a feature: NDBI without SWIR is not NDBI."""
    labels = make_labels(100, 100)
    scene = make_optical_scene(
        tmp_path / "rgbn.tif",
        width=100,
        height=100,
        labels=labels,
        band_names=("blue", "green", "red", "nir"),
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    # NDVI and NDWI remain possible.
    assert compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDVI"]).mask.any()

    with pytest.raises(ToolError) as excinfo:
        compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDBI"])
    assert "swir16" in str(excinfo.value)


def test_engine_skips_a_sar_only_session_with_a_reason(make_context, tmp_path) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=100, height=100)
    context = make_context({ImageRole.SINGLE: scene.path})

    outcome = SpectralIndexEngine().run(context)

    assert outcome.ok is False
    assert outcome.skipped_reason is not None
    assert "optical" in outcome.skipped_reason
    assert outcome.measurements == []


def test_engine_rejects_an_unknown_index_name(make_context, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "s.tif", width=80, height=80)
    context = make_context(
        {ImageRole.SINGLE: scene.path}, parameters={"indices": ["NDXX"]}
    )

    outcome = SpectralIndexEngine().run(context)

    assert outcome.ok is False
    assert "NDXX" in (outcome.skipped_reason or "")


# ---------------------------------------------------------------------------
# Cloud handling
# ---------------------------------------------------------------------------


def test_cloud_pixels_are_excluded_from_the_mask_and_the_denominator(
    make_context, tmp_path
) -> None:
    """Cloud is unobservable, so it belongs in neither the numerator nor the total."""
    labels = make_labels(
        200, 200, water=(120, 0, 80, 200), builtup=None, bare=None,
        cloud=(0, 0, 50, 200),  # 25 per cent of the scene
    )
    scene = make_optical_scene(
        tmp_path / "cloudy.tif", width=200, height=200, labels=labels, with_scl=True
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    result = compute_index(context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDWI"])

    cloud_rows = result.invalid[:50]
    assert cloud_rows.all(), "every cloud pixel must be marked unobservable"
    assert not result.mask[:50].any(), "no cloud pixel may be classified as water"

    # Water is 80 x 200 = 16 000 px of the 30 000 observable pixels.
    observable = int(np.count_nonzero(~result.invalid))
    assert observable == pytest.approx(30_000, rel=0.01)
    detected = int(np.count_nonzero(result.mask))
    assert detected == pytest.approx(16_000, rel=0.03)


def test_cloud_exclusion_can_be_switched_off(make_context, tmp_path) -> None:
    labels = make_labels(
        150, 150, water=None, builtup=None, bare=None, cloud=(0, 0, 40, 150)
    )
    scene = make_optical_scene(
        tmp_path / "c.tif", width=150, height=150, labels=labels, with_scl=True
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    excluded = compute_index(
        context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDVI"], exclude_cloud=True
    )
    kept = compute_index(
        context, ImageRole.SINGLE, INDEX_DEFINITIONS["NDVI"], exclude_cloud=False
    )

    assert int(np.count_nonzero(excluded.invalid)) > int(np.count_nonzero(kept.invalid))


# ---------------------------------------------------------------------------
# The engine end to end
# ---------------------------------------------------------------------------


def test_engine_reports_every_index_with_traceable_measurements(
    make_context, tmp_path
) -> None:
    labels = make_labels(
        300, 300, water=(20, 20, 100, 100), builtup=(180, 180, 80, 80), bare=None
    )
    scene = make_optical_scene(
        tmp_path / "mixed.tif", width=300, height=300, labels=labels
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    outcome = SpectralIndexEngine().run(context)

    assert outcome.ok is True
    assert outcome.tool == "spectral-index-engine"
    assert outcome.implementation is ToolImplementation.DETERMINISTIC
    assert outcome.duration_ms > 0

    # One mask per index.
    keys = {layer.key for layer in outcome.masks}
    assert keys == {
        "ndvi_mask.single",
        "ndwi_mask.single",
        "mndwi_mask.single",
        "ndbi_mask.single",
        "ndmi_mask.single",
    }

    # Water is a 100 x 100 rectangle (1.0 km2) and built-up an 80 x 80 one
    # (0.64 km2). NDWI ranks water highest, built-up in the middle, and
    # vegetation lowest, so Otsu's two-class split separates vegetation from
    # everything else and the mask covers water plus built-up: 1.64 km2. That is
    # Otsu behaving correctly on a three-population scene, not a maths error, and
    # the engine is required to flag the discrepancy.
    area = outcome.measurement("ndwi.single_area_km2")
    assert area is not None
    assert area.value == pytest.approx(1.64, rel=0.05)
    assert area.unit == "km2"
    assert "px" in area.formula and "m2" in area.formula
    assert area.inputs["pixel_count"] > 0
    assert area.inputs["pixel_area_m2"] == pytest.approx(100.0)

    # Every measurement is attributed and carries its formula.
    for measurement in outcome.measurements:
        assert measurement.source_tool == "spectral-index-engine"
        assert measurement.source_version == "1.0.0"
        assert measurement.formula, measurement.key
        assert measurement.unit

    # The threshold and its quality are both reported.
    threshold = outcome.measurement("ndwi.single_threshold")
    separability = outcome.measurement("ndwi.single_separability")
    assert threshold is not None and separability is not None
    assert "Otsu" in (threshold.method or "")
    assert 0.0 <= separability.value <= 1.0
    assert threshold.inputs["conventional_threshold"] == pytest.approx(0.0)

    # The engine must say that its data-driven threshold disagrees with the
    # published one, and by how much.
    assert any(
        "conventional threshold" in note and "NDWI" in note
        for note in outcome.notes
    ), outcome.notes


def test_conventional_threshold_isolates_water_where_otsu_over_selects(
    make_context, tmp_path
) -> None:
    """The published NDWI threshold of 0 is the right tool for a 3-class scene.

    Water sits above 0, built-up and vegetation below it, so the fixed threshold
    recovers the 1.0 km2 water rectangle that Otsu merges with built-up.
    """
    labels = make_labels(
        300, 300, water=(20, 20, 100, 100), builtup=(180, 180, 80, 80), bare=None
    )
    scene = make_optical_scene(
        tmp_path / "mixed.tif", width=300, height=300, labels=labels
    )
    context = make_context(
        {ImageRole.SINGLE: scene.path},
        parameters={
            "indices": ["NDWI"],
            "threshold_method": "fixed",
            "fixed_threshold": 0.0,
        },
    )

    outcome = SpectralIndexEngine().run(context)
    area = outcome.measurement("ndwi.single_area_km2")

    assert area is not None
    assert area.value == pytest.approx(1.0, rel=0.03)

    layer = outcome.mask("ndwi_mask.single")
    assert layer is not None
    truth = scene.labels == int(LandCover.WATER)
    overlap = np.count_nonzero(layer.array & truth) / np.count_nonzero(truth)
    assert overlap > 0.98


def test_engine_can_be_restricted_to_one_index(make_context, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "s.tif", width=120, height=120)
    context = make_context(
        {ImageRole.SINGLE: scene.path}, parameters={"indices": ["NDWI"]}
    )

    outcome = SpectralIndexEngine().run(context)

    assert [layer.key for layer in outcome.masks] == ["ndwi_mask.single"]
    assert outcome.parameters["indices"] == ["NDWI"]


def test_engine_honours_a_fixed_threshold(make_context, tmp_path) -> None:
    """A conventional threshold must be usable in place of the data-driven one."""
    scene = make_optical_scene(tmp_path / "s.tif", width=150, height=150)
    context = make_context(
        {ImageRole.SINGLE: scene.path},
        parameters={
            "indices": ["NDWI"],
            "threshold_method": "fixed",
            "fixed_threshold": 0.0,
        },
    )

    outcome = SpectralIndexEngine().run(context)
    threshold = outcome.measurement("ndwi.single_threshold")

    assert threshold is not None
    assert threshold.value == pytest.approx(0.0)
    assert "fixed" in (threshold.method or "")
    layer = outcome.mask("ndwi_mask.single")
    assert layer is not None and layer.threshold == pytest.approx(0.0)


def test_engine_notes_a_weakly_separated_threshold(make_context, tmp_path) -> None:
    """A uniform scene has no two populations to separate, and must say so."""
    labels = make_labels(150, 150, water=None, builtup=None, bare=None)
    scene = make_optical_scene(
        tmp_path / "uniform.tif", width=150, height=150, labels=labels
    )
    context = make_context(
        {ImageRole.SINGLE: scene.path}, parameters={"indices": ["NDWI"]}
    )

    outcome = SpectralIndexEngine().run(context)

    separability = outcome.measurement("ndwi.single_separability")
    assert separability is not None
    assert separability.value < WEAK_SEPARABILITY
    assert any("weakly supported" in note for note in outcome.notes)


def test_engine_runs_on_both_dates_of_a_bitemporal_pair(make_context, tmp_path) -> None:
    labels = make_labels(150, 150)
    a = make_optical_scene(tmp_path / "a.tif", width=150, height=150, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=150, height=150, labels=labels, seed=31
    )
    context = make_context(
        {ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path},
        parameters={"indices": ["NDVI"]},
    )

    outcome = SpectralIndexEngine().run(context)

    keys = {layer.key for layer in outcome.masks}
    assert keys == {"ndvi_mask.date_a", "ndvi_mask.date_b"}
    assert outcome.measurement("ndvi.date_a_mean") is not None
    assert outcome.measurement("ndvi.date_b_mean") is not None


def test_engine_uses_only_the_optical_half_of_a_cross_modal_pair(
    make_context, tmp_path
) -> None:
    labels = make_labels(150, 150)
    optical = make_optical_scene(
        tmp_path / "o.tif", width=150, height=150, labels=labels
    )
    sar = make_sar_scene(tmp_path / "s.tif", width=150, height=150, labels=labels)
    context = make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path},
        parameters={"indices": ["NDWI"]},
    )

    outcome = SpectralIndexEngine().run(context)

    # Spectral indices are defined on reflectance, not backscatter.
    assert [layer.key for layer in outcome.masks] == ["ndwi_mask.optical"]
