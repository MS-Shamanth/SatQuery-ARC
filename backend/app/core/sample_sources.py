"""Remote imagery acquisition.

Two providers, both reachable without paid access:

* Sentinel-2 L2A from the Earth Search STAC API. Unauthenticated, hosted as
  Cloud-Optimized GeoTIFFs, so a small window can be read with HTTP range
  requests instead of downloading a 10980 x 10980 tile.
* Sentinel-1 RTC from the Microsoft Planetary Computer, signed with an anonymous
  SAS token. RTC rather than GRD on purpose: the RTC product is radiometrically
  terrain corrected and gridded into UTM with a real affine transform, whereas
  the GRD product carries only ground-control points and no CRS, which the
  readiness gate would correctly refuse.

Cross-modal pairs are built by reading the SAR onto the optical scene's exact
grid, so the pair is genuinely co-registered rather than merely nearby.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

COPERNICUS_LICENCE = (
    "Copernicus Sentinel data, free and open under the Copernicus programme"
)
COPERNICUS_ATTRIBUTION = "Contains modified Copernicus Sentinel data"

# GDAL settings for reading remote COGs. Disabling the directory listing keeps
# each open to a single range request.
GDAL_REMOTE_ENV: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "33554432",
}

# Sentinel-2 asset keys to pull, in band order. SCL last: it is a classification,
# not a reflectance band, and must be resampled with nearest neighbour.
S2_ASSETS: tuple[str, ...] = (
    "blue", "green", "red", "nir", "swir16", "swir22", "scl",
)
S2_CATEGORICAL: frozenset[str] = frozenset({"scl"})
S2_REFERENCE_ASSET = "red"  # a 10 m band, used to define the target grid

S1_ASSETS: tuple[str, ...] = ("vv", "vh")
S1_NODATA = -32768.0


class SampleFetchError(RuntimeError):
    """Raised when imagery could not be retrieved from a provider."""


@dataclass
class TargetGrid:
    """The grid a scene is cut onto, shared by every band and both modalities."""

    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return (left, min(bottom, top), right, max(bottom, top))

    @property
    def epsg(self) -> int | None:
        try:
            return rasterio.crs.CRS.from_user_input(self.crs).to_epsg()
        except Exception:  # noqa: BLE001
            return None


@dataclass
class FetchedScene:
    """An in-memory scene ready to be written to disk."""

    cube: np.ndarray
    band_names: list[str]
    grid: TargetGrid
    nodata: float | None
    tags: dict[str, str] = field(default_factory=dict)
    item_id: str | None = None
    item_url: str | None = None
    acquisition_date: datetime | None = None
    cloud_cover: float | None = None
    platform: str | None = None
    collection: str | None = None


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------


def rfc3339_range(date_range: str) -> str:
    """Normalise ``YYYY-MM-DD/YYYY-MM-DD`` to an RFC3339 interval.

    Earth Search rejects bare calendar dates with
    "datetime value is invalid, does not match RFC3339 format", so the interval
    is expanded to cover the full first and last day.
    """
    parts = date_range.split("/")
    if len(parts) != 2:
        return date_range

    def expand(value: str, end_of_day: bool) -> str:
        value = value.strip()
        if "T" in value or value in ("", ".."):
            return value
        return f"{value}T23:59:59Z" if end_of_day else f"{value}T00:00:00Z"

    return f"{expand(parts[0], False)}/{expand(parts[1], True)}"


def search_sentinel2(
    client: httpx.Client,
    bbox: Sequence[float],
    date_range: str,
    max_cloud: float = 12.0,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find low-cloud Sentinel-2 L2A items, least cloudy first."""
    response = client.post(
        f"{EARTH_SEARCH}/search",
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": list(bbox),
            "datetime": rfc3339_range(date_range),
            "limit": limit,
            "query": {"eo:cloud_cover": {"lt": max_cloud}},
        },
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    features.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100.0))
    return features


def search_sentinel1_rtc(
    client: httpx.Client,
    bbox: Sequence[float],
    date_range: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find Sentinel-1 RTC items carrying both VV and VH."""
    response = client.post(
        f"{PC_STAC}/search",
        json={
            "collections": ["sentinel-1-rtc"],
            "bbox": list(bbox),
            "datetime": rfc3339_range(date_range),
            "limit": limit,
        },
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    return [
        feature
        for feature in features
        if {"vv", "vh"} <= set(feature.get("assets", {}))
    ]


def item_covers_point(item: dict[str, Any], lon: float, lat: float) -> bool:
    """True when the item's data footprint contains the point.

    A bbox search returns items whose footprint merely intersects the box. That
    is not enough: Sentinel-2 granules frequently fill only part of their tile,
    so a point inside the tile can still be nodata.
    """
    geometry = item.get("geometry")
    if not geometry:
        return True
    try:
        from shapely.geometry import Point, shape

        return shape(geometry).contains(Point(lon, lat))
    except Exception:  # noqa: BLE001 - fall back to the empirical probe
        return True


def coverage_fraction(url: str, grid: TargetGrid, sample_px: int = 64) -> float:
    """Fraction of the target window that holds real data.

    Read at low resolution against the COG's overviews, so this costs one small
    range request and catches a granule that does not actually cover the area.
    """
    with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(url) as dataset:
        nodata = dataset.nodata if dataset.nodata is not None else 0
        window = from_bounds(*grid.bounds, transform=dataset.transform)
        patch = dataset.read(
            1,
            window=window,
            out_shape=(sample_px, sample_px),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=nodata,
        )
    return float(np.count_nonzero(patch != nodata) / patch.size)


def sign_planetary_computer(href: str, client: httpx.Client) -> str:
    """Attach an anonymous SAS token to a Planetary Computer blob URL."""
    rest = href.split("://", 1)[1]
    host, path = rest.split("/", 1)
    account = host.split(".")[0]
    container = path.split("/", 1)[0]
    response = client.get(f"{PC_SAS}/{account}/{container}")
    if response.status_code != 200:
        raise SampleFetchError(
            f"Planetary Computer refused a token for {account}/{container}: "
            f"HTTP {response.status_code}"
        )
    return f"{href}?{response.json()['token']}"


# ---------------------------------------------------------------------------
# Windowed reads
# ---------------------------------------------------------------------------


def grid_centred_on(
    reference_url: str,
    lon: float,
    lat: float,
    size_px: int,
) -> TargetGrid:
    """Build a square target grid centred on a coordinate.

    The reference asset supplies the CRS and pixel size, so the cut-out lands on
    the source grid exactly and no resampling is needed for bands at that
    resolution.
    """
    with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(reference_url) as dataset:
        crs = dataset.crs
        pixel_x = abs(dataset.transform.a)
        pixel_y = abs(dataset.transform.e)
        xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
        centre_x, centre_y = xs[0], ys[0]

        # Snap the centre to the source grid so pixel edges align.
        col, row = ~dataset.transform * (centre_x, centre_y)
        col, row = round(col), round(row)
        snapped_x, snapped_y = dataset.transform * (col, row)

        left = snapped_x - (size_px / 2) * pixel_x
        top = snapped_y + (size_px / 2) * pixel_y
        transform = Affine(pixel_x, 0.0, left, 0.0, -pixel_y, top)
        return TargetGrid(
            crs=crs.to_string(), transform=transform, width=size_px, height=size_px
        )


def read_onto_grid(
    url: str,
    grid: TargetGrid,
    *,
    band: int = 1,
    categorical: bool = False,
    src_nodata: float | None = None,
) -> np.ndarray:
    """Read a remote raster onto the target grid.

    Same-CRS reads use a plain windowed read. A differing CRS is handled by a
    WarpedVRT, which is what makes an optical/SAR pair land on identical pixels.
    Categorical layers use nearest neighbour: interpolating class codes would
    invent classes that do not exist.
    """
    resampling = Resampling.nearest if categorical else Resampling.bilinear

    with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(url) as dataset:
        same_crs = dataset.crs is not None and dataset.crs.to_string() == grid.crs

        if same_crs:
            window = from_bounds(*grid.bounds, transform=dataset.transform)
            return dataset.read(
                band,
                window=window,
                out_shape=(grid.height, grid.width),
                resampling=resampling,
                boundless=True,
                fill_value=src_nodata if src_nodata is not None else 0,
            )

        with WarpedVRT(
            dataset,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=src_nodata,
            nodata=src_nodata,
        ) as vrt:
            return vrt.read(band)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def fetch_sentinel2_scene(
    client: httpx.Client,
    item: dict[str, Any],
    lon: float,
    lat: float,
    size_px: int,
    grid: TargetGrid | None = None,
) -> FetchedScene:
    """Cut a multi-band Sentinel-2 window out of a STAC item."""
    assets = item.get("assets", {})
    missing = [key for key in S2_ASSETS if key not in assets]
    if missing:
        raise SampleFetchError(
            f"Item {item['id']} lacks assets {missing}; cannot build the scene."
        )

    if grid is None:
        grid = grid_centred_on(assets[S2_REFERENCE_ASSET]["href"], lon, lat, size_px)

    planes: list[np.ndarray] = []
    for key in S2_ASSETS:
        plane = read_onto_grid(
            assets[key]["href"],
            grid,
            categorical=key in S2_CATEGORICAL,
            src_nodata=0,
        )
        planes.append(plane.astype("uint16", copy=False))
        logger.info("  read S2 %-7s %s", key, plane.shape)

    properties = item.get("properties", {})
    acquired = properties.get("datetime")
    acquisition = (
        datetime.fromisoformat(acquired.replace("Z", "+00:00")) if acquired else None
    )

    return FetchedScene(
        cube=np.stack(planes),
        band_names=list(S2_ASSETS),
        grid=grid,
        nodata=0,
        collection="sentinel-2-l2a",
        item_id=item["id"],
        item_url=f"{EARTH_SEARCH}/collections/sentinel-2-l2a/items/{item['id']}",
        acquisition_date=acquisition,
        cloud_cover=properties.get("eo:cloud_cover"),
        platform=properties.get("platform"),
        tags={
            "PLATFORM": str(properties.get("platform", "sentinel-2")),
            "INSTRUMENT": "MSI",
            "PRODUCT": "S2_MSI_L2A",
            "STAC_ITEM_ID": item["id"],
            "REFLECTANCE_SCALE": "10000",
        },
    )


def fetch_sentinel1_rtc_scene(
    client: httpx.Client,
    item: dict[str, Any],
    grid: TargetGrid,
) -> FetchedScene:
    """Cut a VV/VH Sentinel-1 RTC window onto an existing target grid.

    Reading onto the optical grid is what guarantees the cross-modal pair shares
    a CRS, a pixel size, and a footprint.
    """
    assets = item.get("assets", {})
    planes: list[np.ndarray] = []
    for key in S1_ASSETS:
        signed = sign_planetary_computer(assets[key]["href"], client)
        plane = read_onto_grid(signed, grid, src_nodata=S1_NODATA)
        planes.append(plane.astype("float32", copy=False))
        logger.info("  read S1 %-7s %s", key, plane.shape)

    cube = np.stack(planes)
    valid = cube[cube > 0]
    if valid.size < cube.size * 0.25:
        raise SampleFetchError(
            "The Sentinel-1 window is mostly outside the radar swath "
            f"({valid.size / cube.size:.0%} valid)."
        )

    properties = item.get("properties", {})
    acquired = properties.get("datetime")
    acquisition = (
        datetime.fromisoformat(acquired.replace("Z", "+00:00")) if acquired else None
    )

    return FetchedScene(
        cube=cube,
        band_names=["VV", "VH"],
        grid=grid,
        nodata=S1_NODATA,
        collection="sentinel-1-rtc",
        item_id=item["id"],
        item_url=f"{PC_STAC}/collections/sentinel-1-rtc/items/{item['id']}",
        acquisition_date=acquisition,
        platform=properties.get("platform"),
        tags={
            "PLATFORM": str(properties.get("platform", "sentinel-1")),
            "INSTRUMENT": "C-SAR",
            "PRODUCT": "S1_IW_RTC",
            "POLARISATIONS": "VV VH",
            "STAC_ITEM_ID": item["id"],
            "BACKSCATTER_UNITS": "gamma0 linear power",
        },
    )


def write_scene(scene: FetchedScene, path: Path) -> None:
    """Write a fetched scene to a compressed, tiled GeoTIFF with band names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = scene.cube.shape

    tags = dict(scene.tags)
    if scene.acquisition_date is not None:
        tags["ACQUISITION_DATE"] = scene.acquisition_date.isoformat()
    if scene.item_url:
        tags["STAC_URL"] = scene.item_url
    tags["LICENCE"] = COPERNICUS_LICENCE
    tags["ATTRIBUTION"] = COPERNICUS_ATTRIBUTION

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": scene.cube.dtype.name,
        "crs": scene.grid.crs,
        "transform": scene.grid.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if scene.nodata is not None:
        profile["nodata"] = scene.nodata

    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(scene.cube)
        for index, name in enumerate(scene.band_names, start=1):
            dataset.set_band_description(index, name)
        dataset.update_tags(**tags)
        dataset.build_overviews([2, 4, 8], Resampling.average)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
