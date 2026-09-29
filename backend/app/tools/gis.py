"""Deterministic geospatial measurement.

Every area, count, centroid, and distance in the product comes from here. The
functions are plain and auditable on purpose: an area is a pixel count times a
pixel area, and the formula string returned alongside it says exactly that.

Nothing in this module estimates or infers. If a value cannot be computed from
the geometry provided, it returns ``None`` rather than a plausible substitute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Measurement,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome

logger = logging.getLogger(__name__)

# Clusters below this are dropped from polygon output and component counts. At
# 10 m resolution this is 20 pixels, which is the scale below which a Sentinel-2
# "object" is mostly mixed pixels.
MIN_COMPONENT_AREA_M2 = 2000.0


# ---------------------------------------------------------------------------
# Area and counting
# ---------------------------------------------------------------------------


def pixel_area_m2(transform: Affine) -> float:
    """Ground area of one pixel, in square metres."""
    return abs(transform.a) * abs(transform.e)


def mask_area_km2(mask: np.ndarray, transform: Affine) -> tuple[int, float]:
    """Pixel count and area for a boolean mask.

    Returns ``(pixel_count, area_km2)``. This is the single place area is
    derived, so every reported area in the system shares one definition.
    """
    count = int(np.count_nonzero(mask))
    return count, count * pixel_area_m2(transform) / 1_000_000.0


def coverage_fraction(mask: np.ndarray, invalid: np.ndarray | None = None) -> float:
    """Share of usable pixels the mask covers.

    Cloud and nodata are excluded from the denominator: reporting a water
    fraction against a total that includes unobservable pixels understates it.
    """
    valid = mask.size if invalid is None else mask.size - int(np.count_nonzero(invalid))
    if valid <= 0:
        return 0.0
    if invalid is None:
        return int(np.count_nonzero(mask)) / valid
    return int(np.count_nonzero(mask & ~invalid)) / valid


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


@dataclass
class Component:
    """One connected region of a mask."""

    label: int
    pixel_count: int
    area_km2: float
    centroid_rowcol: tuple[float, float]
    centroid_xy: tuple[float, float]
    centroid_lonlat: tuple[float, float] | None
    bbox_rowcol: tuple[int, int, int, int]
    bbox_xy: tuple[float, float, float, float]


def connected_components(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS | None = None,
    *,
    min_area_m2: float = MIN_COMPONENT_AREA_M2,
    connectivity: int = 2,
) -> list[Component]:
    """Label contiguous regions and measure each one.

    Sorted largest first, so the dominant feature is always ``[0]``.
    """
    from skimage.measure import label as sk_label
    from skimage.measure import regionprops

    if not mask.any():
        return []

    labelled = sk_label(mask.astype(bool), connectivity=connectivity)
    area_per_pixel = pixel_area_m2(transform)
    min_pixels = max(1, int(round(min_area_m2 / area_per_pixel))) if area_per_pixel else 1

    components: list[Component] = []
    for region in regionprops(labelled):
        if region.area < min_pixels:
            continue
        row, col = region.centroid
        x, y = transform * (col + 0.5, row + 0.5)
        min_row, min_col, max_row, max_col = region.bbox
        left, top = transform * (min_col, min_row)
        right, bottom = transform * (max_col, max_row)

        lonlat: tuple[float, float] | None = None
        if crs is not None:
            try:
                from rasterio.warp import transform as warp_transform

                xs, ys = warp_transform(crs, "EPSG:4326", [x], [y])
                lonlat = (float(xs[0]), float(ys[0]))
            except Exception:  # noqa: BLE001
                lonlat = None

        components.append(
            Component(
                label=int(region.label),
                pixel_count=int(region.area),
                area_km2=int(region.area) * area_per_pixel / 1_000_000.0,
                centroid_rowcol=(float(row), float(col)),
                centroid_xy=(float(x), float(y)),
                centroid_lonlat=lonlat,
                bbox_rowcol=(int(min_row), int(min_col), int(max_row), int(max_col)),
                bbox_xy=(
                    float(min(left, right)),
                    float(min(top, bottom)),
                    float(max(left, right)),
                    float(max(top, bottom)),
                ),
            )
        )

    components.sort(key=lambda component: component.pixel_count, reverse=True)
    return components


def mask_centroid_lonlat(
    mask: np.ndarray, transform: Affine, crs: CRS | None
) -> list[float] | None:
    """Centroid of every set pixel, in WGS84 degrees."""
    if crs is None or not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    x, y = transform * (cols.mean() + 0.5, rows.mean() + 0.5)
    try:
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform(crs, "EPSG:4326", [x], [y])
        return [float(xs[0]), float(ys[0])]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------


def mask_to_geojson(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS | None,
    *,
    min_area_m2: float = MIN_COMPONENT_AREA_M2,
    simplify_tolerance_m: float | None = None,
    properties: dict[str, Any] | None = None,
    max_features: int = 400,
) -> dict[str, Any]:
    """Convert a mask to a WGS84 GeoJSON FeatureCollection.

    Output is always in EPSG:4326 because that is what GeoJSON specifies and what
    a web map consumes. Each feature carries its own measured area.
    """
    from rasterio.features import shapes as raster_shapes

    features: list[dict[str, Any]] = []
    if not mask.any():
        return {"type": "FeatureCollection", "features": features}

    area_per_pixel = pixel_area_m2(transform)
    binary = mask.astype("uint8")

    try:
        from shapely.geometry import mapping, shape
        from shapely.ops import transform as shapely_transform

        reprojector = None
        if crs is not None:
            from pyproj import Transformer

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            reprojector = transformer.transform

        collected: list[tuple[float, dict[str, Any]]] = []
        for geometry, value in raster_shapes(binary, mask=binary.astype(bool),
                                             transform=transform):
            if value != 1:
                continue
            polygon = shape(geometry)
            area_m2 = polygon.area if area_per_pixel else 0.0
            if area_m2 < min_area_m2:
                continue
            if simplify_tolerance_m:
                polygon = polygon.simplify(simplify_tolerance_m, preserve_topology=True)
            if reprojector is not None:
                polygon = shapely_transform(reprojector, polygon)
            collected.append((area_m2, mapping(polygon)))

        collected.sort(key=lambda item: item[0], reverse=True)
        for index, (area_m2, geometry) in enumerate(collected[:max_features]):
            features.append(
                {
                    "type": "Feature",
                    "id": index,
                    "geometry": geometry,
                    "properties": {
                        "area_m2": round(area_m2, 1),
                        "area_km2": round(area_m2 / 1_000_000.0, 6),
                        "rank": index + 1,
                        **(properties or {}),
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - vector output is not load-bearing
        logger.warning("vectorisation failed: %s", exc)

    return {"type": "FeatureCollection", "features": features}


def component_bboxes_lonlat(
    components: list[Component], crs: CRS | None
) -> list[list[float]]:
    """Bounding boxes in WGS84, for grounding output."""
    if crs is None:
        return []
    try:
        from rasterio.warp import transform_bounds
    except Exception:  # noqa: BLE001
        return []

    boxes: list[list[float]] = []
    for component in components:
        try:
            boxes.append(
                list(transform_bounds(crs, "EPSG:4326", *component.bbox_xy))
            )
        except Exception:  # noqa: BLE001
            continue
    return boxes


# ---------------------------------------------------------------------------
# Relationships between masks
# ---------------------------------------------------------------------------


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks. 1.0 is identical, 0.0 is disjoint."""
    if a.shape != b.shape:
        raise ToolError(
            f"Cannot compare masks of different shapes: {a.shape} and {b.shape}."
        )
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0  # both empty: they agree completely
    return int(np.count_nonzero(a & b)) / union


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two binary masks, corrected for chance.

    Two sources that both mark 95 per cent of a scene as land agree strongly by
    accident. Kappa removes that, which is why it is reported next to raw
    agreement in the debate stage.
    """
    if a.shape != b.shape:
        raise ToolError("Cannot compare masks of different shapes.")
    total = a.size
    if total == 0:
        return 0.0

    both = int(np.count_nonzero(a & b))
    neither = int(np.count_nonzero(~a & ~b))
    observed = (both + neither) / total

    p_a = int(np.count_nonzero(a)) / total
    p_b = int(np.count_nonzero(b)) / total
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def adjacency_fraction(
    subject: np.ndarray, neighbour: np.ndarray, *, within_pixels: int = 2
) -> float:
    """Fraction of ``subject`` pixels lying within N pixels of ``neighbour``.

    Used for questions of the form "how much of the built-up area is next to
    water", and by the misregistration test to measure how much detected change
    hugs an edge.
    """
    if subject.shape != neighbour.shape:
        raise ToolError("Cannot relate masks of different shapes.")
    if not subject.any():
        return 0.0

    from scipy.ndimage import binary_dilation

    structure = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(
        neighbour.astype(bool), structure=structure, iterations=max(1, within_pixels)
    )
    return int(np.count_nonzero(subject & dilated)) / int(np.count_nonzero(subject))


def centroid_distance_m(
    a: np.ndarray, b: np.ndarray, transform: Affine
) -> float | None:
    """Distance between the centroids of two masks, in metres."""
    if not a.any() or not b.any():
        return None
    rows_a, cols_a = np.nonzero(a)
    rows_b, cols_b = np.nonzero(b)
    xa, ya = transform * (cols_a.mean() + 0.5, rows_a.mean() + 0.5)
    xb, yb = transform * (cols_b.mean() + 0.5, rows_b.mean() + 0.5)
    return float(np.hypot(xb - xa, yb - ya))


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


class GisMeasureEngine(BaseTool):
    """Turns masks into measurements, and measures the scene footprint.

    Run on its own it reports the geometry of the input. Its real job is the
    functions above, which every other tool uses so that no two tools can
    disagree about what an area is.
    """

    name = "gis-measure-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Deterministic geometry: areas from pixel counts, connected components, "
        "centroids, bounding boxes, IoU, Cohen's kappa, and adjacency."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            requires_crs=True,
            description="Any georeferenced raster.",
            configurations=[
                InputConfiguration.SINGLE,
                InputConfiguration.CROSS_MODAL_PAIR,
                InputConfiguration.BI_TEMPORAL_PAIR,
            ],
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "min_component_area_m2": {
                "type": "number",
                "default": MIN_COMPONENT_AREA_M2,
                "minimum": 0.0,
                "description": "Clusters smaller than this are ignored.",
            }
        }

    def produces(self) -> list[str]:
        return ["scene_area_km2", "scene_valid_area_km2", "pixel_area_m2"]

    def execute(self, context: ToolContext) -> ToolOutcome:
        outcome = self.outcome(parameters={
            "min_component_area_m2": context.param(
                "min_component_area_m2", MIN_COMPONENT_AREA_M2
            )
        })

        for role in context.roles:
            meta = context.metadata(role)
            if meta.geo.gsd_x_m is None or meta.geo.gsd_y_m is None:
                outcome.notes.append(
                    f"'{role.value}' has no resolvable pixel size; skipped."
                )
                continue

            area_per_pixel = meta.geo.gsd_x_m * meta.geo.gsd_y_m
            total_pixels = meta.width * meta.height
            scene_area = total_pixels * area_per_pixel / 1_000_000.0
            valid_pixels = int(round(total_pixels * (1.0 - meta.nodata_fraction)))

            outcome.measurements.append(
                self.measurement(
                    key=f"pixel_area_m2.{role.value}",
                    label="Pixel ground area",
                    value=area_per_pixel,
                    unit="m2",
                    formula=(
                        f"pixel_area = {meta.geo.gsd_x_m:.4f} m x "
                        f"{meta.geo.gsd_y_m:.4f} m"
                    ),
                    inputs={
                        "gsd_x_m": meta.geo.gsd_x_m,
                        "gsd_y_m": meta.geo.gsd_y_m,
                        "gsd_method": meta.geo.gsd_method,
                    },
                    method=meta.geo.gsd_method,
                    applies_to=[role],
                )
            )
            outcome.measurements.append(
                self.measurement(
                    key=f"scene_area_km2.{role.value}",
                    label="Scene footprint",
                    value=scene_area,
                    unit="km2",
                    formula=(
                        f"area = {meta.width} px x {meta.height} px x "
                        f"{area_per_pixel:.2f} m2 / 1e6"
                    ),
                    inputs={
                        "width_px": meta.width,
                        "height_px": meta.height,
                        "pixel_area_m2": area_per_pixel,
                    },
                    applies_to=[role],
                )
            )
            outcome.measurements.append(
                self.measurement(
                    key=f"scene_valid_area_km2.{role.value}",
                    label="Observable area",
                    value=valid_pixels * area_per_pixel / 1_000_000.0,
                    unit="km2",
                    formula=(
                        f"area = {total_pixels} px x "
                        f"(1 - {meta.nodata_fraction:.4f} nodata) x "
                        f"{area_per_pixel:.2f} m2 / 1e6"
                    ),
                    inputs={
                        "total_pixels": total_pixels,
                        "nodata_fraction": meta.nodata_fraction,
                        "pixel_area_m2": area_per_pixel,
                    },
                    applies_to=[role],
                )
            )

        if not outcome.measurements:
            raise ToolError("No input carried a resolvable pixel size.")
        return outcome


def measurements_for_mask(
    tool_name: str,
    tool_version: str,
    layer: MaskLayer,
    *,
    label: str,
    key_prefix: str,
    min_component_area_m2: float = MIN_COMPONENT_AREA_M2,
    include_components: bool = True,
) -> list[Measurement]:
    """Standard measurement set for a mask: area, share, and cluster count.

    Shared so that every mask in the system is described the same way and the
    numbers are directly comparable between tools.
    """
    results: list[Measurement] = []
    area_per_pixel = layer.pixel_area_m2()
    count = layer.pixel_count
    roles = list(layer.applies_to)

    def make(
        key: str,
        label_: str,
        value: float,
        unit: str,
        formula: str,
        inputs: dict[str, Any],
        precision: int = 2,
    ) -> Measurement:
        return Measurement(
            key=key,
            label=label_,
            value=float(value),
            unit=unit,
            formula=formula,
            inputs=inputs,
            source_tool=tool_name,
            source_version=tool_version,
            method="pixel counting on the computed mask",
            applies_to=roles,
            precision=precision,
        )

    threshold_text = (
        f"{layer.threshold:.4f}" if layer.threshold is not None else "mask"
    )

    if area_per_pixel is not None:
        results.append(
            make(
                f"{key_prefix}_area_km2",
                f"{label} area",
                count * area_per_pixel / 1_000_000.0,
                "km2",
                f"area = {count} px x {area_per_pixel:.2f} m2 / 1e6",
                {
                    "pixel_count": count,
                    "pixel_area_m2": area_per_pixel,
                    "threshold": layer.threshold,
                    "threshold_method": layer.threshold_method,
                },
                precision=3,
            )
        )

    results.append(
        make(
            f"{key_prefix}_pixels",
            f"{label} pixel count",
            count,
            "px",
            f"count of pixels where condition holds ({threshold_text})",
            {"threshold": layer.threshold, "total_pixels": layer.array.size},
            precision=0,
        )
    )

    valid = layer.valid_pixels
    results.append(
        make(
            f"{key_prefix}_fraction",
            f"{label} share of observable area",
            (count / valid) if valid else 0.0,
            "fraction",
            f"fraction = {count} px / {valid} observable px",
            {"pixel_count": count, "observable_pixels": valid},
            precision=4,
        )
    )

    if include_components and count:
        components = connected_components(
            layer.array,
            layer.transform,
            layer.crs,
            min_area_m2=min_component_area_m2,
        )
        results.append(
            make(
                f"{key_prefix}_clusters",
                f"{label} cluster count",
                len(components),
                "count",
                (
                    "connected components (8-connectivity) with area at or above "
                    f"{min_component_area_m2:.0f} m2"
                ),
                {
                    "min_component_area_m2": min_component_area_m2,
                    "connectivity": 8,
                },
                precision=0,
            )
        )
        if components:
            largest = components[0]
            results.append(
                make(
                    f"{key_prefix}_largest_cluster_km2",
                    f"Largest {label.lower()} cluster",
                    largest.area_km2,
                    "km2",
                    (
                        f"area = {largest.pixel_count} px x "
                        f"{area_per_pixel or 0:.2f} m2 / 1e6"
                    ),
                    {
                        "pixel_count": largest.pixel_count,
                        "pixel_area_m2": area_per_pixel,
                        "centroid_lonlat": list(largest.centroid_lonlat)
                        if largest.centroid_lonlat
                        else None,
                    },
                    precision=3,
                )
            )

    return results


__all__ = [
    "Component",
    "GisMeasureEngine",
    "MIN_COMPONENT_AREA_M2",
    "adjacency_fraction",
    "centroid_distance_m",
    "cohens_kappa",
    "component_bboxes_lonlat",
    "connected_components",
    "coverage_fraction",
    "intersection_over_union",
    "mask_area_km2",
    "mask_centroid_lonlat",
    "mask_to_geojson",
    "measurements_for_mask",
    "pixel_area_m2",
]
