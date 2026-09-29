"""GIS measurement tests against exactly known geometry.

Every assertion here compares a computed value to a number derived by hand from
the fixture, not to a previously recorded output. If the area maths ever drifts,
these fail.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.tools.base import MaskLayer, ToolError
from app.tools.gis import (
    adjacency_fraction,
    centroid_distance_m,
    cohens_kappa,
    connected_components,
    coverage_fraction,
    intersection_over_union,
    mask_area_km2,
    mask_centroid_lonlat,
    mask_to_geojson,
    measurements_for_mask,
    pixel_area_m2,
)

# 10 m pixels in UTM 43N, matching the synthetic fixtures.
TRANSFORM = Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 2000000.0)
UTM43N = CRS.from_epsg(32643)


def _layer(mask: np.ndarray, **kwargs) -> MaskLayer:
    return MaskLayer(
        key="probe",
        label="Probe",
        array=mask,
        transform=TRANSFORM,
        crs=UTM43N,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


def test_pixel_area_is_the_product_of_the_pixel_sides() -> None:
    assert pixel_area_m2(TRANSFORM) == pytest.approx(100.0)
    assert pixel_area_m2(Affine(20.0, 0, 0, 0, -20.0, 0)) == pytest.approx(400.0)


def test_area_of_a_known_rectangle_is_exact() -> None:
    """A 50 x 100 block of 10 m pixels is 5000 px and exactly 0.5 km2."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[10:60, 20:120] = True

    count, area = mask_area_km2(mask, TRANSFORM)

    assert count == 5000
    assert area == pytest.approx(0.5)


def test_empty_mask_has_zero_area() -> None:
    count, area = mask_area_km2(np.zeros((32, 32), dtype=bool), TRANSFORM)
    assert count == 0
    assert area == 0.0


def test_full_mask_area_equals_the_scene_footprint() -> None:
    mask = np.ones((100, 150), dtype=bool)
    count, area = mask_area_km2(mask, TRANSFORM)
    assert count == 15_000
    # 150 x 10 m by 100 x 10 m = 1500 m x 1000 m = 1.5 km2.
    assert area == pytest.approx(1.5)


def test_coverage_excludes_unobservable_pixels_from_the_denominator() -> None:
    """Reporting a share against cloud-covered pixels would understate it."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :3] = True  # 30 of 100 pixels
    assert coverage_fraction(mask) == pytest.approx(0.30)

    invalid = np.zeros((10, 10), dtype=bool)
    invalid[:, 8:] = True  # 20 pixels unobservable, none of them in the mask
    # 30 set pixels out of 80 observable, not 100.
    assert coverage_fraction(mask, invalid) == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def test_two_separate_blocks_are_counted_as_two_components() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True   # 400 px = 0.04 km2
    mask[60:90, 60:80] = True   # 600 px = 0.06 km2

    components = connected_components(mask, TRANSFORM, UTM43N, min_area_m2=0.0)

    assert len(components) == 2
    # Sorted largest first.
    assert components[0].pixel_count == 600
    assert components[1].pixel_count == 400
    assert components[0].area_km2 == pytest.approx(0.06)
    assert components[1].area_km2 == pytest.approx(0.04)


def test_diagonally_touching_blocks_are_one_component_under_8_connectivity() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    mask[10:15, 10:15] = True  # touches the first only at a corner

    assert len(connected_components(mask, TRANSFORM, min_area_m2=0.0)) == 1
    assert len(
        connected_components(mask, TRANSFORM, min_area_m2=0.0, connectivity=1)
    ) == 2


def test_components_below_the_minimum_area_are_dropped() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True  # 400 px = 40 000 m2
    mask[80, 80] = True        # 1 px = 100 m2, speckle

    # 2000 m2 is 20 pixels at this resolution.
    components = connected_components(mask, TRANSFORM, min_area_m2=2000.0)
    assert len(components) == 1
    assert components[0].pixel_count == 400


def test_component_centroid_and_bbox_are_georeferenced() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:50] = True

    component = connected_components(mask, TRANSFORM, UTM43N, min_area_m2=0.0)[0]

    # Centre of rows 20-39 is 29.5, of cols 30-49 is 39.5. Adding the half-pixel
    # offset and applying the transform gives 600000 + 40 x 10 = 600400.
    assert component.centroid_xy[0] == pytest.approx(600400.0)
    assert component.centroid_xy[1] == pytest.approx(2000000.0 - 300.0)
    left, bottom, right, top = component.bbox_xy
    assert left == pytest.approx(600300.0)
    assert right == pytest.approx(600500.0)
    assert top == pytest.approx(2000000.0 - 200.0)
    assert bottom == pytest.approx(2000000.0 - 400.0)
    # And it converts to plausible lon/lat over western India.
    assert component.centroid_lonlat is not None
    lon, lat = component.centroid_lonlat
    assert 70 < lon < 80 and 15 < lat < 22


def test_no_components_for_an_empty_mask() -> None:
    assert connected_components(np.zeros((10, 10), dtype=bool), TRANSFORM) == []


def test_mask_centroid_returns_none_without_a_crs() -> None:
    mask = np.ones((10, 10), dtype=bool)
    assert mask_centroid_lonlat(mask, TRANSFORM, None) is None
    assert mask_centroid_lonlat(np.zeros((10, 10), dtype=bool), TRANSFORM, UTM43N) is None
    assert mask_centroid_lonlat(mask, TRANSFORM, UTM43N) is not None


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------


def test_mask_vectorises_to_wgs84_geojson_with_measured_areas() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True  # 400 px = 40 000 m2

    collection = mask_to_geojson(mask, TRANSFORM, UTM43N, min_area_m2=0.0)

    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 1
    feature = collection["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["area_m2"] == pytest.approx(40_000.0, rel=0.01)
    assert feature["properties"]["rank"] == 1

    # GeoJSON must be in lon/lat, not metres.
    lon, lat = feature["geometry"]["coordinates"][0][0]
    assert 70 < lon < 80
    assert 15 < lat < 22


def test_vector_features_are_ranked_by_area() -> None:
    mask = np.zeros((120, 120), dtype=bool)
    mask[5:15, 5:15] = True     # 100 px
    mask[40:70, 40:70] = True   # 900 px

    features = mask_to_geojson(mask, TRANSFORM, UTM43N, min_area_m2=0.0)["features"]

    assert [f["properties"]["rank"] for f in features] == [1, 2]
    assert features[0]["properties"]["area_m2"] > features[1]["properties"]["area_m2"]


def test_empty_mask_vectorises_to_no_features() -> None:
    collection = mask_to_geojson(np.zeros((20, 20), dtype=bool), TRANSFORM, UTM43N)
    assert collection["features"] == []


def test_vector_output_is_capped() -> None:
    # A checkerboard yields many tiny polygons; the cap must hold.
    mask = np.indices((60, 60)).sum(axis=0) % 2 == 0
    collection = mask_to_geojson(
        mask, TRANSFORM, UTM43N, min_area_m2=0.0, max_features=25
    )
    assert len(collection["features"]) <= 25


# ---------------------------------------------------------------------------
# Relationships between masks
# ---------------------------------------------------------------------------


def test_iou_of_known_overlap() -> None:
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:, :6] = True   # 60 px
    b[:, 4:] = True   # 60 px
    # Intersection is columns 4-5 = 20 px; union is all 100 px.
    assert intersection_over_union(a, b) == pytest.approx(0.2)


def test_iou_extremes() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    assert intersection_over_union(mask, mask.copy()) == pytest.approx(1.0)
    assert intersection_over_union(mask, ~mask) == pytest.approx(0.0)
    # Two empty masks agree completely rather than being undefined.
    empty = np.zeros((8, 8), dtype=bool)
    assert intersection_over_union(empty, empty.copy()) == pytest.approx(1.0)


def test_iou_rejects_mismatched_shapes() -> None:
    with pytest.raises(ToolError, match="different shapes"):
        intersection_over_union(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


def test_kappa_is_one_for_identical_and_near_zero_for_chance() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10] = True
    assert cohens_kappa(mask, mask.copy()) == pytest.approx(1.0)

    # Perfect disagreement on a balanced split.
    assert cohens_kappa(mask, ~mask) == pytest.approx(-1.0)


def test_kappa_discounts_agreement_that_is_mostly_chance() -> None:
    """Two sources that both call 95 per cent of a scene land agree by accident."""
    rng = np.random.default_rng(4)
    a = rng.random((200, 200)) < 0.95
    b = rng.random((200, 200)) < 0.95

    raw_agreement = float((a == b).mean())
    kappa = cohens_kappa(a, b)

    assert raw_agreement > 0.85  # looks like strong agreement
    assert abs(kappa) < 0.10     # but is worth almost nothing


def test_adjacency_measures_proximity_not_overlap() -> None:
    subject = np.zeros((20, 20), dtype=bool)
    neighbour = np.zeros((20, 20), dtype=bool)
    subject[:, 5] = True     # a column
    neighbour[:, 7] = True   # two columns away

    assert adjacency_fraction(subject, neighbour, within_pixels=1) == pytest.approx(0.0)
    assert adjacency_fraction(subject, neighbour, within_pixels=2) == pytest.approx(1.0)
    assert adjacency_fraction(np.zeros((20, 20), bool), neighbour) == 0.0


def test_centroid_distance_between_two_blocks() -> None:
    a = np.zeros((100, 100), dtype=bool)
    b = np.zeros((100, 100), dtype=bool)
    a[0:10, 0:10] = True     # centroid at row 4.5, col 4.5
    b[0:10, 50:60] = True    # centroid at row 4.5, col 54.5

    # 50 columns apart at 10 m per pixel.
    assert centroid_distance_m(a, b, TRANSFORM) == pytest.approx(500.0)
    assert centroid_distance_m(a, np.zeros((100, 100), bool), TRANSFORM) is None


# ---------------------------------------------------------------------------
# Measurement records
# ---------------------------------------------------------------------------


def test_mask_measurements_carry_formula_and_inputs() -> None:
    """Every measurement must be reconstructable from what it reports."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 20:120] = True  # clipped to width: 10:60 x 20:100 = 50 x 80 = 4000 px
    layer = _layer(mask, threshold=0.153, threshold_method="Otsu")

    measurements = measurements_for_mask(
        "spectral-index-engine", "1.0.0", layer,
        label="NDWI positive", key_prefix="ndwi.single",
        min_component_area_m2=0.0,
    )
    by_key = {item.key: item for item in measurements}

    area = by_key["ndwi.single_area_km2"]
    assert area.value == pytest.approx(4000 * 100 / 1e6)  # 0.4 km2
    assert "4000 px" in area.formula
    assert area.inputs["pixel_count"] == 4000
    assert area.inputs["pixel_area_m2"] == pytest.approx(100.0)
    assert area.inputs["threshold"] == pytest.approx(0.153)
    assert area.source_tool == "spectral-index-engine"
    assert area.source_version == "1.0.0"

    assert by_key["ndwi.single_pixels"].value == 4000
    assert by_key["ndwi.single_fraction"].value == pytest.approx(0.4)
    assert by_key["ndwi.single_clusters"].value == 1
    assert by_key["ndwi.single_largest_cluster_km2"].value == pytest.approx(0.4)


def test_measurements_omit_area_without_a_crs() -> None:
    """No CRS means no area, rather than an area in meaningless units."""
    layer = MaskLayer(
        key="probe", label="Probe",
        array=np.ones((10, 10), dtype=bool),
        transform=TRANSFORM, crs=None,
    )
    keys = {
        item.key
        for item in measurements_for_mask(
            "t", "1", layer, label="Probe", key_prefix="p", min_component_area_m2=0.0
        )
    }
    assert "p_area_km2" not in keys
    assert "p_pixels" in keys


def test_mask_layer_reports_its_own_geometry() -> None:
    mask = np.zeros((50, 80), dtype=bool)
    mask[:25] = True  # half the rows: 25 x 80 = 2000 px
    layer = _layer(mask)

    assert layer.pixel_count == 2000
    assert layer.valid_pixels == 4000
    assert layer.area_km2() == pytest.approx(0.2)
    left, bottom, right, top = layer.bounds()
    assert left == pytest.approx(600000.0)
    assert right == pytest.approx(600800.0)
    assert top == pytest.approx(2000000.0)
    assert bottom == pytest.approx(1999500.0)

    summary = layer.summary()
    assert summary.pixel_count == 2000
    assert summary.epsg == 32643
    assert summary.coverage_fraction == pytest.approx(0.5)
    assert summary.bounds_wgs84 is not None


def test_mask_layer_excludes_invalid_pixels_from_its_denominator() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3] = True
    invalid = np.zeros((10, 10), dtype=bool)
    invalid[8:] = True

    layer = _layer(mask, invalid=invalid)

    assert layer.valid_pixels == 80
    assert layer.summary().coverage_fraction == pytest.approx(30 / 80)
