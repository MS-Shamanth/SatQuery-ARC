"""Raster ingest tests, checked against synthetic scenes with known properties."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.raster_io import (
    RasterIngestError,
    build_thumbnail,
    extract_acquisition_date,
    normalise_label,
    read_metadata,
    resolve_band_role,
)
from app.models.schemas import BandRole, DateSource, FormatClass, Modality
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
    make_sar_scene,
    optical_from_labels,
    write_raster,
)


# ---------------------------------------------------------------------------
# Band role resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("B04", BandRole.RED),
        ("b04", BandRole.RED),
        ("red", BandRole.RED),
        ("B8A", BandRole.NIR08),
        ("B_8A", BandRole.NIR08),
        ("swir16", BandRole.SWIR16),
        ("B11", BandRole.SWIR16),
        ("SCL", BandRole.SCL),
        ("Scene Classification Map", BandRole.SCL),
        ("VV", BandRole.VV),
        ("Sigma0_VV_db", BandRole.VV),
        ("Gamma0_VH", BandRole.VH),
    ],
)
def test_band_labels_resolve_to_roles(label: str, expected: BandRole) -> None:
    role, source = resolve_band_role(label, 1, 6)
    assert role is expected
    assert source == "description"


def test_normalise_label_strips_separators_and_case() -> None:
    assert normalise_label("Sigma0_VV db") == "sigma0vvdb"
    assert normalise_label(None) == ""


def test_three_band_file_falls_back_to_rgb_convention() -> None:
    roles = [resolve_band_role(None, i, 3)[0] for i in (1, 2, 3)]
    assert roles == [BandRole.RED, BandRole.GREEN, BandRole.BLUE]


def test_unusual_band_count_is_left_unknown_rather_than_guessed() -> None:
    """A 7-band file has no dominant convention, so guessing would be wrong."""
    role, source = resolve_band_role(None, 5, 7)
    assert role is BandRole.UNKNOWN
    assert source == "unknown"


# ---------------------------------------------------------------------------
# Core metadata extraction
# ---------------------------------------------------------------------------


def test_optical_scene_metadata_matches_what_was_written(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif", width=200, height=160)
    meta = read_metadata(scene.path)

    assert meta.driver == "GTiff"
    assert meta.format_class is FormatClass.GEOSPATIAL
    assert (meta.width, meta.height) == (200, 160)
    assert meta.band_count == 6
    assert meta.dtype == "uint16"
    assert meta.geo.epsg == 32643
    assert meta.geo.is_projected is True

    # 10 m pixels in a metre-based projection must report exactly 10 m.
    assert meta.geo.gsd_x_m == pytest.approx(10.0)
    assert meta.geo.gsd_y_m == pytest.approx(10.0)
    assert meta.geo.gsd_m == pytest.approx(10.0)
    assert meta.geo.gsd_method == "projected-linear-unit"

    # Footprint: 200 x 10 m by 160 x 10 m = 2000 m x 1600 m = 3.2 km2.
    assert meta.geo.area_km2 == pytest.approx(3.2)


def test_all_six_optical_band_roles_resolve_from_descriptions(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif")
    meta = read_metadata(scene.path)

    assert meta.resolved_roles == [
        BandRole.BLUE,
        BandRole.GREEN,
        BandRole.RED,
        BandRole.NIR,
        BandRole.SWIR16,
        BandRole.SWIR22,
    ]
    assert meta.has_roles(BandRole.NIR, BandRole.RED, BandRole.SWIR16)
    assert meta.missing_roles(BandRole.COASTAL, BandRole.RED) == [BandRole.COASTAL]
    assert meta.band_index(BandRole.NIR) == 4
    assert meta.band_index(BandRole.CIRRUS) is None


def test_band_wavelengths_are_attached_from_the_role(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif")
    meta = read_metadata(scene.path)
    by_role = {band.role: band for band in meta.bands}
    assert by_role[BandRole.RED].wavelength_nm == pytest.approx(665.0)
    assert by_role[BandRole.NIR].wavelength_nm == pytest.approx(842.0)


def test_optical_modality_is_inferred_with_reasons(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif")
    meta = read_metadata(scene.path)
    assert meta.modality.modality is Modality.OPTICAL
    assert meta.modality.confidence >= 0.9
    assert meta.modality.reasons


def test_sar_modality_is_inferred_from_polarisation_bands(tmp_path) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif")
    meta = read_metadata(scene.path)

    assert meta.modality.modality is Modality.SAR
    assert meta.modality.confidence >= 0.95
    assert "VV" in " ".join(meta.modality.reasons)
    assert meta.band_index(BandRole.VV) == 1
    assert meta.band_index(BandRole.VH) == 2
    assert meta.dtype == "float32"


def test_band_statistics_are_computed_from_pixel_values(tmp_path) -> None:
    labels = make_labels(128, 128, water=(0, 0, 64, 128), builtup=None, bare=None)
    scene = make_optical_scene(tmp_path / "half_water.tif", width=128, height=128, labels=labels)
    meta = read_metadata(scene.path)

    nir = next(band for band in meta.bands if band.role is BandRole.NIR)
    assert nir.stats is not None
    # Half water (NIR ~120), half vegetation (NIR ~3800): the mean must land
    # between them, nowhere near either extreme.
    assert 1500 < nir.stats.mean < 2500
    assert nir.stats.minimum < 400
    assert nir.stats.maximum > 3000
    assert nir.stats.valid_pixels > 0
    assert nir.stats.sample_fraction == pytest.approx(1.0)


def test_large_raster_statistics_report_their_decimation(tmp_path) -> None:
    """Stats on big rasters are sampled; the fraction must say so."""
    labels = make_labels(1200, 1200)
    scene = make_optical_scene(
        tmp_path / "big.tif", width=1200, height=1200, labels=labels
    )
    meta = read_metadata(scene.path)
    band = meta.bands[0]
    assert band.stats is not None
    assert band.stats.sample_fraction < 1.0


# ---------------------------------------------------------------------------
# Ground sample distance across CRS types
# ---------------------------------------------------------------------------


def test_geographic_crs_gsd_is_measured_geodesically(tmp_path) -> None:
    """Degrees are not a length, so a geographic CRS needs geodesic conversion.

    At 20 degrees north, 0.0001 degrees of longitude is roughly 10.4 m and the
    same step in latitude is roughly 11.1 m, so the two axes must differ.
    """
    labels = make_labels(64, 64)
    cube = optical_from_labels(labels)
    path = write_raster(
        tmp_path / "geographic.tif",
        cube,
        epsg=4326,
        origin=(72.5, 20.0),
        pixel_size=0.0001,
        band_names=("blue", "green", "red", "nir", "swir16", "swir22"),
    )
    meta = read_metadata(path)

    assert meta.geo.is_geographic is True
    assert meta.geo.gsd_method == "geodesic-at-centre"
    assert meta.geo.pixel_size_native_x == pytest.approx(0.0001)
    assert meta.geo.gsd_x_m == pytest.approx(10.4, abs=0.6)
    assert meta.geo.gsd_y_m == pytest.approx(11.06, abs=0.6)
    # Longitude degrees are shorter than latitude degrees away from the equator.
    assert meta.geo.gsd_x_m < meta.geo.gsd_y_m


def test_wgs84_bounds_and_centroid_are_derived_for_a_utm_scene(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif", width=128, height=128)
    meta = read_metadata(scene.path)

    assert meta.geo.bounds_wgs84 is not None
    west, south, east, north = meta.geo.bounds_wgs84
    assert west < east and south < north
    # UTM 43N with this origin sits over western India.
    assert 70 < west < 80
    assert 15 < south < 25

    assert meta.geo.centroid_wgs84 is not None
    lon, lat = meta.geo.centroid_wgs84
    assert west <= lon <= east
    assert south <= lat <= north


# ---------------------------------------------------------------------------
# Missing georeferencing and nodata
# ---------------------------------------------------------------------------


def test_png_without_crs_is_classified_as_benchmark_and_noted(tmp_path) -> None:
    """PNG is permitted for benchmark datasets, so this is a note, not a crash."""
    rgb = np.random.default_rng(3).integers(0, 255, (3, 48, 48), dtype="uint8")
    path = write_raster(tmp_path / "bench.png", rgb, epsg=None, driver="PNG")
    meta = read_metadata(path)

    assert meta.format_class is FormatClass.BENCHMARK_RASTER
    assert meta.geo.epsg is None
    assert meta.geo.gsd_m is None
    assert meta.geo.area_km2 is None
    assert any("no CRS" in note for note in meta.ingest_notes)
    # Positional convention still applies, so indices remain possible.
    assert meta.resolved_roles == [BandRole.RED, BandRole.GREEN, BandRole.BLUE]


def test_nodata_fraction_is_measured(tmp_path) -> None:
    labels = make_labels(100, 100, water=None, builtup=None, bare=None)
    cube = optical_from_labels(labels).astype("uint16")
    # Blank out a known quarter of the scene.
    cube[:, :50, :50] = 0
    path = write_raster(
        tmp_path / "gappy.tif",
        cube,
        band_names=("blue", "green", "red", "nir", "swir16", "swir22"),
        nodata=0,
    )
    meta = read_metadata(path)

    assert meta.nodata_value == 0
    assert meta.nodata_fraction == pytest.approx(0.25, abs=0.01)


def test_acquisition_date_and_day_of_year_come_from_the_tag(tmp_path) -> None:
    scene = make_optical_scene(
        tmp_path / "dated.tif",
        acquisition_date=datetime(2024, 11, 5, tzinfo=timezone.utc),
    )
    meta = read_metadata(scene.path)

    assert meta.acquisition_date is not None
    assert meta.acquisition_date.year == 2024
    assert meta.acquisition_date.month == 11
    assert meta.date_source is DateSource.GEOTIFF_TAG
    assert meta.day_of_year == 310


def test_acquisition_date_falls_back_to_the_filename(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "scene_20210718_utm.tif", acquisition_date=None)
    meta = read_metadata(scene.path)

    assert meta.date_source is DateSource.FILENAME
    assert meta.acquisition_date is not None
    assert (meta.acquisition_date.year, meta.acquisition_date.month) == (2021, 7)


@pytest.mark.parametrize(
    ("tags", "expected_source"),
    [
        ({"TIFFTAG_DATETIME": "2022:06:01 10:30:00"}, DateSource.GEOTIFF_TAG),
        ({"SENSING_TIME": "2022-06-01T10:30:00Z"}, DateSource.GEOTIFF_TAG),
        ({"SATQUERY_ACQUISITION_DATE": "2022-06-01"}, DateSource.MANIFEST),
        ({}, DateSource.UNKNOWN),
    ],
)
def test_date_tag_formats_are_parsed(tags: dict, expected_source: DateSource) -> None:
    parsed, source = extract_acquisition_date(tags, "nodate.tif")
    assert source is expected_source
    if expected_source is not DateSource.UNKNOWN:
        assert parsed is not None and parsed.year == 2022


def test_unreadable_file_raises_a_clear_ingest_error(tmp_path) -> None:
    bogus = tmp_path / "not_a_raster.tif"
    bogus.write_bytes(b"this is definitely not a GeoTIFF")

    with pytest.raises(RasterIngestError) as excinfo:
        read_metadata(bogus)
    assert "not_a_raster.tif" in str(excinfo.value)


def test_missing_file_raises_ingest_error(tmp_path) -> None:
    with pytest.raises(RasterIngestError):
        read_metadata(tmp_path / "absent.tif")


def test_tiling_is_flagged_only_for_large_rasters(tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "small.tif", width=128, height=128)
    assert read_metadata(scene.path).tiling_recommended is False


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def test_optical_thumbnail_uses_a_true_colour_recipe(tmp_path) -> None:
    from PIL import Image

    scene = make_optical_scene(tmp_path / "optical.tif", width=300, height=200)
    meta = read_metadata(scene.path)
    out = tmp_path / "thumbs" / "single.png"

    recipe = build_thumbnail(scene.path, meta, out)

    assert out.exists()
    assert recipe is not None and "True colour" in recipe
    with Image.open(out) as image:
        assert image.mode == "RGB"
        assert max(image.size) <= 640
        # Aspect ratio is preserved.
        assert image.size[0] > image.size[1]


def test_sar_thumbnail_reports_its_false_colour_composite(tmp_path) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=128, height=128)
    meta = read_metadata(scene.path)
    out = tmp_path / "thumbs" / "sar.png"

    recipe = build_thumbnail(scene.path, meta, out)

    assert out.exists()
    assert recipe is not None
    # SAR is not a photograph, so the UI must be able to say what is displayed.
    assert "VV" in recipe and "dB" in recipe


def test_thumbnail_of_water_scene_is_not_uniform(tmp_path) -> None:
    """A stretched preview of a scene with distinct classes must show contrast."""
    from PIL import Image

    labels = make_labels(128, 128, water=(0, 0, 64, 128), builtup=None, bare=None)
    scene = make_optical_scene(
        tmp_path / "contrast.tif", width=128, height=128, labels=labels
    )
    meta = read_metadata(scene.path)
    out = tmp_path / "t.png"
    build_thumbnail(scene.path, meta, out)

    with Image.open(out) as image:
        array = np.asarray(image)
    assert array.std() > 20


# ---------------------------------------------------------------------------
# Fixture self-checks: the ground truth must actually be what it claims
# ---------------------------------------------------------------------------


def test_fixture_reports_exact_class_areas() -> None:
    labels = make_labels(200, 200, water=(0, 0, 50, 100), builtup=None, bare=None)
    from tests.raster_fixtures import SyntheticScene
    from pathlib import Path

    scene = SyntheticScene(
        path=Path("unused"), labels=labels, pixel_size_m=10.0, epsg=32643
    )
    # 50 x 100 pixels at 10 m = 5000 x 100 m2 = 0.5 km2.
    assert scene.pixels(LandCover.WATER) == 5000
    assert scene.area_km2(LandCover.WATER) == pytest.approx(0.5)
    assert scene.total_area_km2() == pytest.approx(4.0)


def test_fixture_spectral_signatures_produce_expected_index_signs() -> None:
    """Guards the fixture itself: NDVI/NDWI/NDBI must behave per class."""
    labels = np.array(
        [[LandCover.WATER, LandCover.VEGETATION, LandCover.BUILTUP]], dtype="uint8"
    )
    cube = optical_from_labels(labels, noise_sigma=0.0).astype("float64")
    blue, green, red, nir, swir16, _ = cube

    ndvi = (nir - red) / (nir + red)
    ndwi = (green - nir) / (green + nir)
    ndbi = (swir16 - nir) / (swir16 + nir)

    assert ndvi[0, 0] < 0 < ndvi[0, 1]      # water negative, vegetation positive
    assert ndwi[0, 0] > 0 > ndwi[0, 1]      # water positive, vegetation negative
    assert ndbi[0, 2] > 0 > ndbi[0, 1]      # built-up positive, vegetation negative
    assert blue[0, 2] > blue[0, 1]          # built-up brighter in blue than veg


def test_fixture_sar_signatures_order_classes_correctly() -> None:
    from tests.raster_fixtures import sar_from_labels

    labels = np.full((64, 64), int(LandCover.WATER), dtype="uint8")
    labels[:, 32:] = int(LandCover.BUILTUP)
    vv = sar_from_labels(labels, looks=30)[0]

    water_mean = float(vv[:, :32].mean())
    builtup_mean = float(vv[:, 32:].mean())
    # Built-up must be far brighter than specular water.
    assert builtup_mean - water_mean > 12.0
    assert water_mean < -15.0
