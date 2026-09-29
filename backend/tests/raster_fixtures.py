"""Synthetic raster builders for known-answer tests.

These fixtures exist so that measurements can be checked against ground truth.
A scene is generated from an explicit land-cover label map, which means the
exact water area, built-up area, and change area are known integers before any
code under test runs.

Spectral values are chosen to be physically sensible for Sentinel-2 L2A
reflectance scaled by 10000, so the spectral indices behave the way they do on
real imagery:

    water       NDVI negative, NDWI strongly positive
    vegetation  NDVI strongly positive, NDWI negative
    built-up    NDBI positive, NDVI near zero
    bare soil   NDBI positive, spectrally close to built-up on purpose, because
                that confusion is real and the disagreement engine must show it

SAR values are in decibels with realistic contrast: water is a specular dark
surface, built-up is bright from double-bounce, vegetation sits in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


class LandCover(IntEnum):
    WATER = 1
    VEGETATION = 2
    BUILTUP = 3
    BARE = 4
    # Cloud is modelled as a cover class so the readiness gate's cloud checks can
    # be exercised. It is bright across the visible bands with a near-zero NDVI,
    # which is what both the SCL classifier and the brightness proxy react to.
    CLOUD = 5


# Band order used by the optical builder.
OPTICAL_BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")

# Reflectance x 10000, per land-cover class, in OPTICAL_BANDS order.
OPTICAL_SIGNATURE: dict[LandCover, tuple[int, ...]] = {
    LandCover.WATER: (400, 350, 250, 120, 60, 40),
    LandCover.VEGETATION: (350, 600, 400, 3800, 1500, 700),
    LandCover.BUILTUP: (1400, 1500, 1700, 2000, 2600, 2200),
    LandCover.BARE: (1100, 1300, 1700, 2200, 2900, 2500),
    # Bright in the visible, flat across red/NIR so NDVI lands near zero.
    LandCover.CLOUD: (3000, 3100, 3200, 3300, 2000, 1500),
}

# Backscatter in dB for (VV, VH).
SAR_SIGNATURE: dict[LandCover, tuple[float, float]] = {
    LandCover.WATER: (-22.0, -28.0),
    LandCover.VEGETATION: (-9.0, -15.0),
    LandCover.BUILTUP: (-3.0, -10.0),
    LandCover.BARE: (-12.0, -20.0),
    # Radar sees through cloud, so the surface underneath is what returns.
    LandCover.CLOUD: (-9.0, -15.0),
}

# Sentinel-2 Scene Classification Layer codes for each cover class.
SCL_CODE: dict[LandCover, int] = {
    LandCover.WATER: 6,
    LandCover.VEGETATION: 4,
    LandCover.BUILTUP: 5,
    LandCover.BARE: 5,
    LandCover.CLOUD: 9,
}

# A UTM zone 43N origin, which puts synthetic scenes over western India.
DEFAULT_EPSG = 32643
DEFAULT_ORIGIN = (600000.0, 2000000.0)
DEFAULT_PIXEL_SIZE = 10.0


@dataclass
class SyntheticScene:
    """A written raster plus the ground truth used to build it."""

    path: Path
    labels: np.ndarray
    pixel_size_m: float
    epsg: int
    acquisition_date: datetime | None = None
    counts: dict[LandCover, int] = field(default_factory=dict)

    @property
    def pixel_area_m2(self) -> float:
        return self.pixel_size_m**2

    def pixels(self, cover: LandCover) -> int:
        return int(np.count_nonzero(self.labels == cover))

    def area_km2(self, cover: LandCover) -> float:
        """Exact area of one class, from the label map that generated the file."""
        return self.pixels(cover) * self.pixel_area_m2 / 1_000_000.0

    def total_area_km2(self) -> float:
        return self.labels.size * self.pixel_area_m2 / 1_000_000.0


def make_labels(
    width: int = 256,
    height: int = 256,
    *,
    water: tuple[int, int, int, int] | None = (20, 150, 80, 90),
    builtup: tuple[int, int, int, int] | None = (140, 30, 70, 60),
    bare: tuple[int, int, int, int] | None = (30, 30, 50, 40),
    cloud: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Build a label map. Rectangles are ``(row, col, height, width)``.

    Everything not covered by a rectangle is vegetation, so the class areas are
    exactly the rectangle areas. Later entries paint over earlier ones, and cloud
    paints last because it physically occludes whatever is beneath it.
    """
    labels = np.full((height, width), int(LandCover.VEGETATION), dtype="uint8")
    for rect, cover in (
        (bare, LandCover.BARE),
        (builtup, LandCover.BUILTUP),
        (water, LandCover.WATER),
        (cloud, LandCover.CLOUD),
    ):
        if rect is None:
            continue
        row, col, rows, cols = rect
        labels[row : row + rows, col : col + cols] = int(cover)
    return labels


def optical_from_labels(
    labels: np.ndarray, *, seed: int = 7, noise_sigma: float = 45.0
) -> np.ndarray:
    """Render a 6-band uint16 reflectance cube from a label map."""
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    cube = np.zeros((len(OPTICAL_BANDS), height, width), dtype="float64")

    for cover, signature in OPTICAL_SIGNATURE.items():
        mask = labels == int(cover)
        if not mask.any():
            continue
        for band_index, value in enumerate(signature):
            cube[band_index][mask] = value

    cube += rng.normal(0.0, noise_sigma, cube.shape)
    return np.clip(cube, 1, 10000).astype("uint16")


def sar_from_labels(
    labels: np.ndarray, *, seed: int = 11, looks: int = 6
) -> np.ndarray:
    """Render a 2-band float32 dB backscatter pair (VV, VH) from a label map.

    Speckle is applied as multiplicative Gamma noise in the linear power domain,
    which is the standard statistical model for multi-look SAR intensity, then
    converted to decibels.
    """
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    out = np.zeros((2, height, width), dtype="float64")

    for pol_index in range(2):
        linear = np.zeros((height, width), dtype="float64")
        for cover, signature in SAR_SIGNATURE.items():
            mask = labels == int(cover)
            if not mask.any():
                continue
            linear[mask] = 10.0 ** (signature[pol_index] / 10.0)
        speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(height, width))
        out[pol_index] = 10.0 * np.log10(np.maximum(linear * speckle, 1e-12))

    return out.astype("float32")


def write_raster(
    path: Path,
    array: np.ndarray,
    *,
    epsg: int | None = DEFAULT_EPSG,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    band_names: tuple[str, ...] | None = None,
    tags: dict[str, str] | None = None,
    nodata: float | None = None,
    driver: str = "GTiff",
) -> Path:
    """Write a 2D or 3D array to a georeferenced raster."""
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count, height, width = array.shape

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile: dict[str, object] = {
        "driver": driver,
        "width": width,
        "height": height,
        "count": count,
        "dtype": array.dtype.name,
        "transform": from_origin(origin[0], origin[1], pixel_size, pixel_size),
    }
    if epsg is not None:
        profile["crs"] = rasterio.CRS.from_epsg(epsg)
    if nodata is not None:
        profile["nodata"] = nodata
    if driver == "GTiff":
        profile.update({"compress": "deflate", "tiled": False})

    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array)
        if band_names:
            for index, name in enumerate(band_names[:count], start=1):
                dataset.set_band_description(index, name)
        if tags:
            dataset.update_tags(**{k: str(v) for k, v in tags.items()})

    return path


def make_optical_scene(
    path: Path,
    *,
    width: int = 256,
    height: int = 256,
    labels: np.ndarray | None = None,
    epsg: int | None = DEFAULT_EPSG,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    acquisition_date: datetime | None = datetime(2020, 3, 14, tzinfo=timezone.utc),
    platform: str = "sentinel-2a",
    seed: int = 7,
    nodata: float | None = None,
    extra_tags: dict[str, str] | None = None,
    band_names: tuple[str, ...] | None = OPTICAL_BANDS,
    with_scl: bool = False,
) -> SyntheticScene:
    """Write a 6-band optical scene with named bands and a date tag.

    With ``with_scl`` a seventh band is appended carrying Sentinel-2 Scene
    Classification Layer codes, which lets the readiness gate use a real
    classification for cloud rather than the brightness proxy.
    """
    if labels is None:
        labels = make_labels(width, height)
    cube = optical_from_labels(labels, seed=seed)

    if with_scl:
        scl = np.zeros(labels.shape, dtype="uint16")
        for cover, code in SCL_CODE.items():
            scl[labels == int(cover)] = code
        cube = np.concatenate([cube, scl[np.newaxis, ...]], axis=0)
        band_names = tuple(band_names or ()) + ("scl",)

    tags: dict[str, str] = {"PLATFORM": platform, "INSTRUMENT": "MSI"}
    if acquisition_date is not None:
        tags["ACQUISITION_DATE"] = acquisition_date.isoformat()
    if extra_tags:
        tags.update(extra_tags)

    write_raster(
        path,
        cube,
        epsg=epsg,
        origin=origin,
        pixel_size=pixel_size,
        band_names=band_names,
        tags=tags,
        nodata=nodata,
    )
    return SyntheticScene(
        path=Path(path),
        labels=labels,
        pixel_size_m=pixel_size,
        epsg=epsg or 0,
        acquisition_date=acquisition_date,
    )


def make_sar_scene(
    path: Path,
    *,
    width: int = 256,
    height: int = 256,
    labels: np.ndarray | None = None,
    epsg: int | None = DEFAULT_EPSG,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    acquisition_date: datetime | None = datetime(2020, 3, 16, tzinfo=timezone.utc),
    platform: str = "sentinel-1a",
    seed: int = 11,
    looks: int = 6,
) -> SyntheticScene:
    """Write a 2-band VV/VH decibel scene."""
    if labels is None:
        labels = make_labels(width, height)
    cube = sar_from_labels(labels, seed=seed, looks=looks)

    tags: dict[str, str] = {
        "PLATFORM": platform,
        "INSTRUMENT": "C-SAR",
        "PRODUCT_TYPE": "GRD",
        "POLARISATIONS": "VV VH",
    }
    if acquisition_date is not None:
        tags["ACQUISITION_DATE"] = acquisition_date.isoformat()

    write_raster(
        path,
        cube,
        epsg=epsg,
        origin=origin,
        pixel_size=pixel_size,
        band_names=("VV", "VH"),
        tags=tags,
    )
    return SyntheticScene(
        path=Path(path),
        labels=labels,
        pixel_size_m=pixel_size,
        epsg=epsg or 0,
        acquisition_date=acquisition_date,
    )
