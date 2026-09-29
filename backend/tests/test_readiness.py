"""Readiness gate tests.

Each deliberate defect gets a fixture that must trigger exactly the right check,
and a clean fixture that must not trigger it. That pairing is what stops the gate
from degenerating into something that either always passes or always complains.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.readiness import (
    COREG_WARN_PX,
    NODATA_FAIL,
    SEASONAL_MONTH_DELTA,
    estimate_cloud_fraction,
    estimate_coregistration,
    evaluate_readiness,
)
from app.core.raster_io import read_metadata
from app.models.schemas import CheckStatus, ImageRole, ReadinessVerdict
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    make_sar_scene,
    optical_from_labels,
    write_raster,
)

MARCH = datetime(2020, 3, 14, tzinfo=timezone.utc)
MARCH_LATER = datetime(2024, 3, 18, tzinfo=timezone.utc)
OCTOBER = datetime(2024, 10, 22, tzinfo=timezone.utc)


def _check(report, check_id: str):
    for check in report.checks:
        if check.id == check_id:
            return check
    raise AssertionError(
        f"No check {check_id!r}. Present: {sorted(c.id for c in report.checks)}"
    )


def _ingest(store, session_id: str, role: ImageRole, path, name: str | None = None):
    return store.ingest(session_id, role, path, name or path.name, move=False)


def _bitemporal(store, tmp_path, *, a_kwargs=None, b_kwargs=None, labels=None):
    """Create a session holding a bi-temporal pair."""
    labels = labels if labels is not None else make_labels(220, 220)
    a = make_optical_scene(
        tmp_path / "a.tif", width=220, height=220, labels=labels,
        acquisition_date=MARCH, **(a_kwargs or {}),
    )
    b_labels = (b_kwargs or {}).pop("labels", labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=220, height=220, labels=b_labels,
        acquisition_date=MARCH_LATER, seed=23, **(b_kwargs or {}),
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)
    return store.load(record.session_id), a, b


# ---------------------------------------------------------------------------
# Clean inputs
# ---------------------------------------------------------------------------


def test_clean_single_image_is_ready(store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "clean.tif", width=200, height=200)
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    assert report.verdict is ReadinessVerdict.READY, report.refusal_reasons
    assert report.refusal_reasons == []
    assert _check(report, "crs.single").status is CheckStatus.PASS
    assert _check(report, "crs.single").measured == "EPSG:32643"
    assert _check(report, "resolution.single").measured_numeric == pytest.approx(10.0)
    assert _check(report, "bands.single").measured == "6 of 6"
    assert report.computed_ms > 0


def test_pair_checks_are_marked_not_applicable_for_a_single_image(
    store, tmp_path
) -> None:
    """The gate still reports pair checks, so its coverage is visible."""
    scene = make_optical_scene(tmp_path / "clean.tif", width=120, height=120)
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    for check_id in (
        "pair.crs_match",
        "pair.overlap",
        "pair.coregistration",
        "pair.temporal",
        "pair.modality_complement",
    ):
        assert _check(report, check_id).status is CheckStatus.NOT_APPLICABLE


def test_aligned_bitemporal_pair_passes_every_gate(store, tmp_path) -> None:
    record, _a, _b = _bitemporal(store, tmp_path)
    report = evaluate_readiness(record, store)

    assert report.verdict is ReadinessVerdict.READY, report.refusal_reasons
    assert _check(report, "pair.crs_match").status is CheckStatus.PASS
    assert _check(report, "pair.overlap").status is CheckStatus.PASS
    assert _check(report, "pair.overlap").measured_numeric == pytest.approx(1.0)
    assert _check(report, "pair.resolution_match").measured_numeric == pytest.approx(1.0)

    coreg = _check(report, "pair.coregistration")
    assert coreg.status is CheckStatus.PASS
    assert coreg.measured_numeric is not None
    assert coreg.measured_numeric < 1.0

    # Same month of year four years apart: no seasonal risk.
    assert _check(report, "pair.temporal").status is CheckStatus.PASS
    assert report.seasonal_risk is False
    assert report.day_delta is not None and report.day_delta > 1000
    assert report.month_of_year_delta == 0

    # Geometry is carried forward for later stages.
    assert report.common_epsg == 32643
    assert report.overlap_bounds_native is not None
    assert report.overlap_bounds_wgs84 is not None
    assert report.overlap_fraction == pytest.approx(1.0)


def test_cross_modal_pair_with_both_modalities_passes(store, tmp_path) -> None:
    labels = make_labels(200, 200)
    optical = make_optical_scene(
        tmp_path / "o.tif", width=200, height=200, labels=labels
    )
    sar = make_sar_scene(tmp_path / "s.tif", width=200, height=200, labels=labels)
    record = store.create()
    _ingest(store, record.session_id, ImageRole.OPTICAL, optical.path)
    _ingest(store, record.session_id, ImageRole.SAR, sar.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    complement = _check(report, "pair.modality_complement")
    assert complement.status is CheckStatus.PASS
    assert complement.measured == "optical + SAR"
    # Radar is unaffected by cloud, so that check does not apply to it.
    assert _check(report, "cloud.sar").status is CheckStatus.NOT_APPLICABLE
    assert _check(report, "cloud.sar").method == "not-applicable-sar"
    # Cross-modal registration is estimated on gradients, not raw intensity.
    assert "gradient" in (_check(report, "pair.coregistration").method or "")
    assert report.verdict is not ReadinessVerdict.REFUSED, report.refusal_reasons


# ---------------------------------------------------------------------------
# Deliberate defects: each must trigger its own check and refuse
# ---------------------------------------------------------------------------


def test_mismatched_projections_are_refused(store, tmp_path) -> None:
    labels = make_labels(150, 150)
    a = make_optical_scene(
        tmp_path / "utm43.tif", width=150, height=150, labels=labels, epsg=32643
    )
    b = make_optical_scene(
        tmp_path / "utm44.tif",
        width=150,
        height=150,
        labels=labels,
        epsg=32644,
        origin=(300000.0, 2000000.0),
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    assert report.verdict is ReadinessVerdict.REFUSED
    crs_match = _check(report, "pair.crs_match")
    assert crs_match.status is CheckStatus.FAIL
    assert "32643" in crs_match.measured and "32644" in crs_match.measured
    # Downstream geometry checks cannot run until projections agree.
    assert _check(report, "pair.overlap").status is CheckStatus.NOT_APPLICABLE
    assert _check(report, "pair.coregistration").status is CheckStatus.NOT_APPLICABLE
    assert any("same projection" in r.what for r in report.requirements)


def test_non_overlapping_footprints_are_refused(store, tmp_path) -> None:
    labels = make_labels(150, 150)
    a = make_optical_scene(tmp_path / "a.tif", width=150, height=150, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif",
        width=150,
        height=150,
        labels=labels,
        # 500 km east: same projection, no shared ground.
        origin=(1_100_000.0, 2_000_000.0),
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    overlap = _check(report, "pair.overlap")
    assert report.verdict is ReadinessVerdict.REFUSED
    assert overlap.status is CheckStatus.FAIL
    assert overlap.measured_numeric == pytest.approx(0.0)
    assert any("same ground" in r.what for r in report.requirements)


def test_incompatible_resolutions_are_refused(store, tmp_path) -> None:
    labels = make_labels(150, 150)
    a = make_optical_scene(
        tmp_path / "fine.tif", width=150, height=150, labels=labels, pixel_size=10.0
    )
    b = make_optical_scene(
        tmp_path / "coarse.tif", width=150, height=150, labels=labels, pixel_size=60.0
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    match = _check(report, "pair.resolution_match")
    assert match.status is CheckStatus.FAIL
    assert match.measured_numeric == pytest.approx(6.0)
    assert report.verdict is ReadinessVerdict.REFUSED


def test_a_shifted_pair_is_detected_and_refused(store, tmp_path) -> None:
    """Content moved but georeferencing unchanged: a registration error.

    The gate must measure the offset rather than silently producing a change map
    made of edge artefacts.
    """
    labels = make_labels(240, 240)
    shifted = np.roll(np.roll(labels, 6, axis=0), 4, axis=1)

    a = make_optical_scene(tmp_path / "a.tif", width=240, height=240, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=240, height=240, labels=shifted, seed=23
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    coreg = _check(report, "pair.coregistration")
    assert coreg.status is CheckStatus.FAIL, coreg.measured
    assert coreg.measured_numeric is not None
    assert coreg.measured_numeric > COREG_WARN_PX
    # A 6-row, 4-column shift is about 7.2 pixels of displacement.
    assert coreg.measured_numeric == pytest.approx(7.2, abs=1.0)
    assert report.verdict is ReadinessVerdict.REFUSED
    assert any("sub-pixel" in r.what for r in report.requirements)


def test_coregistration_measures_the_shift_direction(tmp_path) -> None:
    labels = make_labels(200, 200)
    shifted = np.roll(labels, 5, axis=1)  # five columns east
    a = make_optical_scene(tmp_path / "a.tif", width=200, height=200, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=200, height=200, labels=shifted, seed=23
    )

    result = estimate_coregistration(
        a.path, read_metadata(a.path), b.path, read_metadata(b.path)
    )

    assert result is not None
    assert abs(result.dx_px) == pytest.approx(5.0, abs=0.6)
    assert abs(result.dy_px) == pytest.approx(0.0, abs=0.6)
    assert result.shift_m == pytest.approx(50.0, abs=6.0)


def test_a_large_but_genuine_shift_is_still_reported_as_a_failure(
    store, tmp_path
) -> None:
    """The plausibility bound must not excuse a real misregistration.

    A 30-pixel shift is past the bound used to spot estimator failures, so the
    validation step has to distinguish "the peak is noise" from "the offset is
    genuinely large". Content that matches after shifting is a real offset.
    """
    labels = make_labels(300, 300)
    shifted = np.roll(labels, 30, axis=1)

    a = make_optical_scene(tmp_path / "a.tif", width=300, height=300, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=300, height=300, labels=shifted, seed=23
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    coreg = _check(report, "pair.coregistration")
    assert coreg.status is CheckStatus.FAIL, coreg.measured
    assert coreg.measured_numeric == pytest.approx(30.0, abs=2.0)
    assert report.verdict is ReadinessVerdict.REFUSED


def test_an_unvalidated_correlation_peak_is_discarded(tmp_path) -> None:
    """Uncorrelated content must not yield a confident offset."""
    rng = np.random.default_rng(3)
    labels_a = rng.integers(1, 5, (260, 260)).astype("uint8")
    labels_b = rng.integers(1, 5, (260, 260)).astype("uint8")

    a = make_optical_scene(tmp_path / "noise_a.tif", width=260, height=260, labels=labels_a)
    b = make_sar_scene(tmp_path / "noise_b.tif", width=260, height=260, labels=labels_b)

    result = estimate_coregistration(
        a.path, read_metadata(a.path), b.path, read_metadata(b.path)
    )

    assert result is not None
    assert result.reliable is False
    assert "mutual information" in result.reliability_note
    # Both were written on the same grid, so that fact is available as a fallback.
    assert result.identical_grid is True


def test_identical_grid_falls_back_to_georeferencing_when_correlation_fails(
    store, tmp_path
) -> None:
    """A cross-modal pair cut onto one grid is aligned by construction.

    Optical brightness and radar backscatter need not correlate. When the
    content-based estimate cannot be validated, the shared grid is the better
    authority, and the check must say which authority it used.
    """
    rng = np.random.default_rng(11)
    labels_a = rng.integers(1, 5, (260, 260)).astype("uint8")
    labels_b = rng.integers(1, 5, (260, 260)).astype("uint8")

    optical = make_optical_scene(
        tmp_path / "o.tif", width=260, height=260, labels=labels_a
    )
    sar = make_sar_scene(tmp_path / "s.tif", width=260, height=260, labels=labels_b)
    record = store.create()
    _ingest(store, record.session_id, ImageRole.OPTICAL, optical.path)
    _ingest(store, record.session_id, ImageRole.SAR, sar.path)

    report = evaluate_readiness(store.load(record.session_id), store)
    coreg = _check(report, "pair.coregistration")

    assert coreg.status is CheckStatus.PASS
    assert coreg.measured == "aligned by georeferencing"
    assert "identical grid" in coreg.message
    # The rejected estimate is disclosed rather than quietly dropped.
    assert coreg.method is not None and "estimate rejected" in coreg.method
    assert report.verdict is not ReadinessVerdict.REFUSED


def test_unverifiable_alignment_on_different_grids_warns(store, tmp_path) -> None:
    """Without a shared grid there is no fallback authority, so warn."""
    rng = np.random.default_rng(5)
    labels_a = rng.integers(1, 5, (240, 240)).astype("uint8")
    labels_b = rng.integers(1, 5, (240, 240)).astype("uint8")

    optical = make_optical_scene(
        tmp_path / "o.tif", width=240, height=240, labels=labels_a
    )
    # Same CRS and resolution, but the grid origin differs by half a scene.
    sar = make_sar_scene(
        tmp_path / "s.tif",
        width=240,
        height=240,
        labels=labels_b,
        origin=(600600.0, 1999400.0),
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.OPTICAL, optical.path)
    _ingest(store, record.session_id, ImageRole.SAR, sar.path)

    report = evaluate_readiness(store.load(record.session_id), store)
    coreg = _check(report, "pair.coregistration")

    assert coreg.status is CheckStatus.WARN
    assert coreg.measured == "not verified"
    assert "could not be confirmed" in coreg.message


def test_mostly_empty_scene_is_refused(store, tmp_path) -> None:
    labels = make_labels(160, 160, water=None, builtup=None, bare=None)
    cube = optical_from_labels(labels)
    cube[:, :, :130] = 0  # blank out about 81 per cent of the columns
    path = write_raster(
        tmp_path / "gappy.tif",
        cube,
        band_names=("blue", "green", "red", "nir", "swir16", "swir22"),
        nodata=0,
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, path)

    report = evaluate_readiness(store.load(record.session_id), store)

    nodata = _check(report, "nodata.single")
    assert nodata.status is CheckStatus.FAIL
    assert nodata.measured_numeric is not None
    assert nodata.measured_numeric > NODATA_FAIL
    assert report.verdict is ReadinessVerdict.REFUSED


def test_benchmark_png_is_refused_for_geospatial_work_with_clear_requirements(
    store, tmp_path
) -> None:
    rgb = np.random.default_rng(5).integers(20, 240, (3, 64, 64), dtype="uint8")
    path = write_raster(tmp_path / "bench.png", rgb, epsg=None, driver="PNG")
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, path)

    report = evaluate_readiness(store.load(record.session_id), store)

    assert report.verdict is ReadinessVerdict.REFUSED
    assert _check(report, "crs.single").status is CheckStatus.FAIL
    assert _check(report, "crs.single").measured == "absent"
    assert _check(report, "resolution.single").status is CheckStatus.FAIL
    # The refusal explains what to supply instead.
    assert any("georeferenced GeoTIFF" in r.what for r in report.requirements)
    assert any("CRS" in reason for reason in report.refusal_reasons)


def test_two_optical_images_cannot_form_a_cross_modal_pair(store, tmp_path) -> None:
    labels = make_labels(140, 140)
    a = make_optical_scene(tmp_path / "a.tif", width=140, height=140, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=140, height=140, labels=labels, seed=31
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.OPTICAL, a.path)
    _ingest(store, record.session_id, ImageRole.SAR, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    complement = _check(report, "pair.modality_complement")
    assert complement.status is CheckStatus.FAIL
    assert complement.measured == "optical"
    assert report.verdict is ReadinessVerdict.REFUSED
    assert any("one optical" in r.what.lower() for r in report.requirements)


def test_incomplete_configuration_is_refused(store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=100, height=100)
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    assert report.verdict is ReadinessVerdict.REFUSED
    assert any("not fully populated" in reason for reason in report.refusal_reasons)


def test_empty_session_is_refused_without_crashing(store) -> None:
    record = store.create()
    report = evaluate_readiness(record, store)

    assert report.verdict is ReadinessVerdict.REFUSED
    assert report.checks == []
    assert "No imagery" in report.refusal_reasons[0]


# ---------------------------------------------------------------------------
# Seasonality: the input to the kill-shot demo
# ---------------------------------------------------------------------------


def test_seasonal_pair_warns_and_flags_seasonal_risk(store, tmp_path) -> None:
    labels = make_labels(180, 180)
    a = make_optical_scene(
        tmp_path / "march.tif", width=180, height=180, labels=labels,
        acquisition_date=MARCH,
    )
    b = make_optical_scene(
        tmp_path / "october.tif", width=180, height=180, labels=labels, seed=23,
        acquisition_date=OCTOBER,
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    temporal = _check(report, "pair.temporal")
    assert temporal.status is CheckStatus.WARN
    assert report.seasonal_risk is True
    assert report.month_of_year_delta is not None
    assert report.month_of_year_delta >= SEASONAL_MONTH_DELTA
    assert "Phenology" in temporal.message
    # Seasonality is a warning, not a refusal: the analysis proceeds and the
    # confounder test decides what the change means.
    assert report.verdict is ReadinessVerdict.READY_WITH_WARNINGS


def test_missing_dates_warn_that_seasonality_cannot_be_ruled_out(
    store, tmp_path
) -> None:
    labels = make_labels(120, 120)
    a = make_optical_scene(
        tmp_path / "undated_a.tif", width=120, height=120, labels=labels,
        acquisition_date=None,
    )
    b = make_optical_scene(
        tmp_path / "undated_b.tif", width=120, height=120, labels=labels, seed=23,
        acquisition_date=None,
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    temporal = _check(report, "pair.temporal")
    assert temporal.status is CheckStatus.WARN
    assert temporal.measured == "unknown"
    assert report.seasonal_risk is True
    assert report.verdict is ReadinessVerdict.READY_WITH_WARNINGS


def test_identical_dates_warn_that_there_is_no_interval(store, tmp_path) -> None:
    labels = make_labels(120, 120)
    a = make_optical_scene(
        tmp_path / "a.tif", width=120, height=120, labels=labels, acquisition_date=MARCH
    )
    b = make_optical_scene(
        tmp_path / "b.tif", width=120, height=120, labels=labels, seed=23,
        acquisition_date=MARCH,
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.DATE_A, a.path)
    _ingest(store, record.session_id, ImageRole.DATE_B, b.path)

    report = evaluate_readiness(store.load(record.session_id), store)
    assert _check(report, "pair.temporal").status is CheckStatus.WARN
    assert report.day_delta == 0


# ---------------------------------------------------------------------------
# Cloud: real classification versus proxy
# ---------------------------------------------------------------------------


def test_cloud_fraction_uses_the_scene_classification_layer_when_present(
    tmp_path,
) -> None:
    # A 60 x 200 cloud band over a 200 x 200 scene is exactly 30 per cent.
    labels = make_labels(200, 200, water=None, builtup=None, bare=None,
                        cloud=(0, 0, 60, 200))
    scene = make_optical_scene(
        tmp_path / "cloudy_scl.tif", width=200, height=200, labels=labels,
        with_scl=True,
    )
    meta = read_metadata(scene.path)

    estimate = estimate_cloud_fraction(scene.path, meta)

    assert estimate.method == "sentinel-2-scl"
    assert estimate.fraction == pytest.approx(0.30, abs=0.02)
    assert "high probability" in estimate.detail


def test_cloud_fraction_falls_back_to_a_labelled_brightness_proxy(tmp_path) -> None:
    labels = make_labels(200, 200, water=None, builtup=None, bare=None,
                        cloud=(0, 0, 60, 200))
    scene = make_optical_scene(
        tmp_path / "cloudy_plain.tif", width=200, height=200, labels=labels,
        with_scl=False,
    )
    meta = read_metadata(scene.path)

    estimate = estimate_cloud_fraction(scene.path, meta)

    assert estimate.method == "brightness-heuristic"
    assert estimate.fraction == pytest.approx(0.30, abs=0.05)
    # The proxy must announce that it is a proxy.
    assert "Less reliable" in estimate.detail


def test_heavy_cloud_is_refused(store, tmp_path) -> None:
    labels = make_labels(200, 200, water=None, builtup=None, bare=None,
                        cloud=(0, 0, 150, 200))  # 75 per cent
    scene = make_optical_scene(
        tmp_path / "socked_in.tif", width=200, height=200, labels=labels, with_scl=True
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    cloud = _check(report, "cloud.single")
    assert cloud.status is CheckStatus.FAIL
    assert cloud.measured_numeric == pytest.approx(0.75, abs=0.03)
    assert report.verdict is ReadinessVerdict.REFUSED
    assert any("SAR acquisition" in r.why or "SAR" in r.what for r in report.requirements)


def test_moderate_cloud_warns_but_proceeds(store, tmp_path) -> None:
    labels = make_labels(200, 200, water=None, builtup=None, bare=None,
                        cloud=(0, 0, 70, 200))  # 35 per cent
    scene = make_optical_scene(
        tmp_path / "hazy.tif", width=200, height=200, labels=labels, with_scl=True
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)

    assert _check(report, "cloud.single").status is CheckStatus.WARN
    assert report.verdict is ReadinessVerdict.READY_WITH_WARNINGS
    assert report.passed is True


def test_clear_scene_passes_the_cloud_check(store, tmp_path) -> None:
    scene = make_optical_scene(
        tmp_path / "clear.tif", width=200, height=200, with_scl=True
    )
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, scene.path)

    report = evaluate_readiness(store.load(record.session_id), store)
    cloud = _check(report, "cloud.single")
    assert cloud.status is CheckStatus.PASS
    assert cloud.measured_numeric == pytest.approx(0.0, abs=0.001)


def test_scl_band_is_recognised_as_a_band_role(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "scl.tif", width=64, height=64, with_scl=True)
    meta = read_metadata(scene.path)
    from app.models.schemas import BandRole

    assert meta.band_index(BandRole.SCL) == 7
    assert meta.band_count == 7


# ---------------------------------------------------------------------------
# Traceability: every check must carry its number and its threshold
# ---------------------------------------------------------------------------


def test_every_applicable_check_reports_a_measurement_and_a_threshold(
    store, tmp_path
) -> None:
    record, _a, _b = _bitemporal(store, tmp_path)
    report = evaluate_readiness(record, store)

    applicable = [
        check
        for check in report.checks
        if check.status is not CheckStatus.NOT_APPLICABLE
    ]
    assert len(applicable) >= 14
    for check in applicable:
        assert check.measured, f"{check.id} reports no measured value"
        assert check.threshold, f"{check.id} states no threshold"
        assert check.message, f"{check.id} has no explanation"


def test_readiness_route_returns_a_report_not_an_error(api, store, tmp_path) -> None:
    """A refusal is a 200 with a report: the client needs the reasons."""
    rgb = np.random.default_rng(5).integers(20, 240, (3, 48, 48), dtype="uint8")
    path = write_raster(tmp_path / "bench.png", rgb, epsg=None, driver="PNG")
    record = store.create()
    _ingest(store, record.session_id, ImageRole.SINGLE, path)

    response = api.get(f"/api/sessions/{record.session_id}/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "refused"
    assert body["refusal_reasons"]
    assert body["requirements"]
    assert any(c["id"] == "crs.single" and c["status"] == "fail" for c in body["checks"])


def test_readiness_route_404s_for_an_unknown_session(api) -> None:
    assert api.get(f"/api/sessions/{'0' * 32}/readiness").status_code == 404
