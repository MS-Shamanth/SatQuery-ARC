"""Map layers: reprojected rasters and vector footprints for the web map.

Everything here is geometric bookkeeping in service of one rule: what the map
draws must sit where the measurement says it sits.

Web maps consume EPSG:4326, and the imagery is in UTM. Laying a UTM raster onto
a latitude/longitude image overlay by simply taking its corner coordinates skews
it, and the error grows with distance from the central meridian. So every layer
is properly reprojected before it is written, which means a highlighted region on
the map coincides with the pixels whose area was reported.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from app.core.raster_io import composite_rgb, read_metadata
from app.models.schemas import ImageRole, RasterMetadata

logger = logging.getLogger(__name__)

WGS84 = CRS.from_epsg(4326)

# Reprojected overlays are display assets, so they are capped. A 1600 px edge is
# beyond what a browser shows on a laptop and keeps the files small.
MAX_OVERLAY_EDGE = 1600

# Fill alpha for a highlighted region. Low enough to read the imagery beneath,
# high enough to be unambiguous in a recording.
MASK_ALPHA = 150

# Semantic colours, matching the palette the rest of the interface uses. Keyed by
# a substring of the mask key so a new index inherits a sensible colour.
# Order matters: the first token found in the key wins, so direction words are
# listed before index names. Gain and loss must never render in the same colour.
LAYER_COLOURS: dict[str, tuple[int, int, int]] = {
    "corroborated": (16, 185, 129),
    "gain": (52, 211, 153),
    "increase": (52, 211, 153),
    "loss": (251, 113, 133),
    "decrease": (251, 113, 133),
    "disagree": (244, 63, 94),
    "agree": (52, 211, 153),
    "magnitude": (244, 114, 182),
    "change": (244, 114, 182),
    "mndwi": (56, 189, 248),
    "ndwi": (56, 189, 248),
    "water": (56, 189, 248),
    "flood": (129, 140, 248),
    "ndvi": (132, 204, 22),
    "vegetation": (132, 204, 22),
    "ndbi": (251, 146, 60),
    "built": (251, 146, 60),
    "ndmi": (129, 212, 250),
}
DEFAULT_COLOUR = (34, 211, 238)


def colour_for(key: str) -> tuple[int, int, int]:
    """Semantic colour for a layer, matched on the first token found in its key.

    Direction is checked before subject: a viewer must be able to tell gain from
    loss at a glance without reading the legend, so "change_gain" resolves on
    "gain" rather than on "change".
    """
    lowered = key.lower()
    for token, colour in LAYER_COLOURS.items():
        if token in lowered:
            return colour
    return DEFAULT_COLOUR


@dataclass
class StoredLayer:
    """A layer written to disk, with everything the client needs to place it."""

    key: str
    label: str
    description: str
    kind: str
    png_filename: str
    geojson_filename: str | None
    bounds_wgs84: list[float]
    colour: str
    area_km2: float | None = None
    pixel_count: int | None = None
    applies_to: list[ImageRole] | None = None
    recipe: str | None = None


def _target_grid(
    width: int, height: int, transform: Affine, crs: CRS
) -> tuple[Affine, int, int, list[float]]:
    """The EPSG:4326 grid a source raster reprojects onto, capped in size."""
    left, top = transform.c, transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    bounds = (left, min(top, bottom), right, max(top, bottom))

    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs, WGS84, width, height, *bounds
    )

    long_edge = max(dst_width, dst_height)
    if long_edge > MAX_OVERLAY_EDGE:
        scale = MAX_OVERLAY_EDGE / long_edge
        dst_width = max(1, int(dst_width * scale))
        dst_height = max(1, int(dst_height * scale))
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, WGS84, width, height, *bounds,
            dst_width=dst_width, dst_height=dst_height,
        )

    wgs = transform_bounds(crs, WGS84, *bounds)
    return dst_transform, int(dst_width), int(dst_height), list(wgs)


def _reproject_band(
    source: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    *,
    resampling: Resampling,
    dtype: str,
    fill: float = 0.0,
) -> np.ndarray:
    destination = np.full(dst_shape, fill, dtype=dtype)
    reproject(
        source=source.astype(dtype),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=WGS84,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=None,
    )
    return destination


def write_mask_overlay(
    mask: np.ndarray,
    transform: Affine,
    crs: CRS | None,
    out_path: Path,
    *,
    colour: tuple[int, int, int],
    invalid: np.ndarray | None = None,
) -> list[float] | None:
    """Write a mask as a transparent RGBA PNG in EPSG:4326.

    Returns the geographic bounds the image covers, or None when the raster has
    no CRS and therefore cannot be placed on a map at all.
    """
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = mask.shape

    if crs is None:
        return None

    dst_transform, dst_width, dst_height, bounds = _target_grid(
        width, height, transform, crs
    )
    shape = (dst_height, dst_width)

    # Nearest neighbour: a mask is a decision per pixel, and interpolating it
    # would invent partial membership that no measurement reflects.
    warped = _reproject_band(
        mask.astype("uint8"), transform, crs, dst_transform, shape,
        resampling=Resampling.nearest, dtype="uint8",
    )
    rgba = np.zeros((dst_height, dst_width, 4), dtype="uint8")
    selected = warped > 0
    rgba[selected, 0] = colour[0]
    rgba[selected, 1] = colour[1]
    rgba[selected, 2] = colour[2]
    rgba[selected, 3] = MASK_ALPHA

    # A crisp edge makes the region legible against busy imagery.
    edge = _outline(selected)
    rgba[edge, 0] = min(255, colour[0] + 40)
    rgba[edge, 1] = min(255, colour[1] + 40)
    rgba[edge, 2] = min(255, colour[2] + 40)
    rgba[edge, 3] = 255

    if invalid is not None:
        unusable = _reproject_band(
            invalid.astype("uint8"), transform, crs, dst_transform, shape,
            resampling=Resampling.nearest, dtype="uint8",
        )
        # Pixels that were never observable are drawn as a faint neutral wash so
        # the viewer can tell "not present" apart from "not looked at".
        blank = (unusable > 0) & ~selected
        rgba[blank, 0] = 120
        rgba[blank, 1] = 120
        rgba[blank, 2] = 130
        rgba[blank, 3] = 40

    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG", optimize=True)
    return bounds


def _outline(selected: np.ndarray) -> np.ndarray:
    """Pixels on the boundary of a selection."""
    if not selected.any():
        return np.zeros_like(selected)
    padded = np.pad(selected, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return selected & ~interior


def write_base_overlay(
    path: Path, metadata: RasterMetadata, out_path: Path
) -> tuple[list[float] | None, str]:
    """Write the scene itself as a reprojected RGBA PNG.

    The map keeps working without a tile server because the imagery under
    investigation is itself the base layer. Online basemaps are context, not a
    dependency.
    """
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    long_edge = max(metadata.width, metadata.height)
    decimation = max(1, math.ceil(long_edge / MAX_OVERLAY_EDGE))
    src_height = max(1, metadata.height // decimation)
    src_width = max(1, metadata.width // decimation)

    stack, recipe, valid = composite_rgb(path, metadata, src_height, src_width)

    with rasterio.open(path) as dataset:
        src_crs = dataset.crs
        # The composite was read decimated, so the transform must be scaled to
        # match or the overlay lands in the wrong place.
        src_transform = dataset.transform * Affine.scale(
            dataset.width / src_width, dataset.height / src_height
        )

    if src_crs is None:
        Image.fromarray(stack, mode="RGB").save(out_path, format="PNG", optimize=True)
        return None, recipe

    dst_transform, dst_width, dst_height, bounds = _target_grid(
        src_width, src_height, src_transform, src_crs
    )
    shape = (dst_height, dst_width)

    rgba = np.zeros((dst_height, dst_width, 4), dtype="uint8")
    for band in range(3):
        rgba[:, :, band] = _reproject_band(
            stack[:, :, band], src_transform, src_crs, dst_transform, shape,
            resampling=Resampling.bilinear, dtype="uint8",
        )
    rgba[:, :, 3] = (
        _reproject_band(
            valid.astype("uint8") * 255, src_transform, src_crs, dst_transform, shape,
            resampling=Resampling.nearest, dtype="uint8",
        )
    )

    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG", optimize=True)
    return bounds, recipe


def write_geojson(collection: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collection), encoding="utf-8")


# ---------------------------------------------------------------------------
# Persisting a run's layers
# ---------------------------------------------------------------------------


def persist_outcome_layers(
    directory: Path, tool: str, outcome, *, max_layers: int = 12
) -> list[StoredLayer]:
    """Write every mask a tool produced as a map layer.

    Done at the run level rather than inside each tool, so any tool that emits a
    mask becomes visible on the map without knowing the map exists.
    """
    from app.tools.gis import mask_to_geojson

    stored: list[StoredLayer] = []
    for layer in outcome.masks[:max_layers]:
        safe = _safe_name(f"{tool}__{layer.key}")
        png = directory / f"{safe}.png"
        try:
            bounds = write_mask_overlay(
                layer.array,
                layer.transform,
                layer.crs,
                png,
                colour=colour_for(layer.key),
                invalid=layer.invalid,
            )
        except Exception as exc:  # noqa: BLE001 - a missing overlay is not fatal
            logger.warning("overlay failed for %s: %s", layer.key, exc)
            continue

        if bounds is None:
            logger.info("mask %s has no CRS, so it cannot be mapped", layer.key)
            continue

        geojson_name: str | None = None
        try:
            collection = mask_to_geojson(
                layer.array,
                layer.transform,
                layer.crs,
                properties={"layer": layer.key, "label": layer.label},
            )
            if collection["features"]:
                write_geojson(collection, directory / f"{safe}.geojson")
                geojson_name = f"{safe}.geojson"
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector export failed for %s: %s", layer.key, exc)

        layer.stored_filename = png.name
        red, green, blue = colour_for(layer.key)
        stored.append(
            StoredLayer(
                key=layer.key,
                label=layer.label,
                description=layer.description,
                kind="mask",
                png_filename=png.name,
                geojson_filename=geojson_name,
                bounds_wgs84=bounds,
                colour=f"#{red:02x}{green:02x}{blue:02x}",
                area_km2=layer.area_km2(),
                pixel_count=layer.pixel_count,
                applies_to=list(layer.applies_to),
            )
        )
    return stored


def persist_base_layers(directory: Path, session, store) -> list[StoredLayer]:
    """Write each ingested image as a reprojected base layer."""
    stored: list[StoredLayer] = []
    for role, image in session.images.items():
        path = store.find_image_path(session.session_id, role)
        if path is None:
            continue
        safe = _safe_name(f"base__{role.value}")
        try:
            metadata = image.metadata or read_metadata(path)
            bounds, recipe = write_base_overlay(
                path, metadata, directory / f"{safe}.png"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("base layer failed for %s: %s", role.value, exc)
            continue
        if bounds is None:
            continue
        stored.append(
            StoredLayer(
                key=f"base.{role.value}",
                label=f"{role.value.replace('_', ' ').title()} imagery",
                description=recipe,
                kind="base",
                png_filename=f"{safe}.png",
                geojson_filename=None,
                bounds_wgs84=bounds,
                colour="#0b1220",
                applies_to=[role],
                recipe=recipe,
            )
        )
    return stored


def _safe_name(raw: str) -> str:
    """A filename derived from a key, with nothing that can escape a directory."""
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in raw
    )[:120]


__all__ = [
    "LAYER_COLOURS",
    "MASK_ALPHA",
    "StoredLayer",
    "colour_for",
    "persist_base_layers",
    "persist_outcome_layers",
    "write_base_overlay",
    "write_mask_overlay",
]
