"""Raster ingest: read a GeoTIFF and extract everything measurable from it.

This module is deliberately conservative. Where a value cannot be derived from
the file it is left as ``None`` and a note is recorded, rather than being filled
with a plausible default. Downstream stages depend on being able to tell the
difference between "measured 10 m" and "we do not know the resolution".

Ground sample distance gets special care: a projected CRS reports pixel size in
its own linear unit, while a geographic CRS reports degrees, which are not a
length. The two cases are converted differently and the method used is recorded
alongside the value.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from app.models.schemas import (
    BAND_ALIASES,
    BAND_WAVELENGTH_NM,
    OPTICAL_ROLES,
    SAR_ROLES,
    BandInfo,
    BandRole,
    BandStats,
    DateSource,
    FormatClass,
    GeoReference,
    Modality,
    ModalityInference,
    RasterMetadata,
)

logger = logging.getLogger(__name__)

GEOD = Geod(ellps="WGS84")

# Statistics are computed on a decimated read so that ingest stays responsive on
# full Sentinel-2 tiles. The sampling fraction is reported with the numbers.
STATS_TARGET_PIXELS = 512 * 512

# Above this size the pipeline switches to windowed processing.
TILING_THRESHOLD_PIXELS = 4096 * 4096

THUMBNAIL_MAX_EDGE = 640

GEOSPATIAL_DRIVERS = {"GTiff", "COG", "VRT", "HFA", "NetCDF", "JP2OpenJPEG"}

DATE_TAG_KEYS = (
    "ACQUISITION_DATETIME",
    "ACQUISITION_DATE",
    "SENSING_TIME",
    "DATETIME",
    "TIFFTAG_DATETIME",
    "DATE",
    "START_TIME",
    "SATQUERY_ACQUISITION_DATE",
)

_FILENAME_DATE = re.compile(r"(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")


class RasterIngestError(ValueError):
    """Raised when a file cannot be opened or interpreted as a raster."""


# ---------------------------------------------------------------------------
# Band role resolution
# ---------------------------------------------------------------------------


def normalise_label(text: str | None) -> str:
    """Collapse a band label to an alias-table key.

    ``"Sigma0_VV_db"`` and ``"sigma0 vv db"`` both become ``"sigma0vvdb"``.
    """
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_band_role(
    label: str | None,
    index: int,
    band_count: int,
    band_tags: dict[str, str] | None = None,
) -> tuple[BandRole, str]:
    """Resolve one band's semantic role.

    Resolution order is most-trustworthy first: an explicit band description,
    then a per-band tag, then a positional convention based on band count.
    The source is returned so the UI can show how confident the mapping is.
    """
    direct = BAND_ALIASES.get(normalise_label(label))
    if direct:
        return direct, "description"

    for key in ("BANDNAME", "band_name", "NAME", "role", "COMMON_NAME", "common_name"):
        candidate = (band_tags or {}).get(key)
        alias = BAND_ALIASES.get(normalise_label(candidate))
        if alias:
            return alias, "tag"

    # Positional conventions. Only applied when there is a widely used one, so a
    # 5-band file does not get silently mislabelled.
    if band_count == 1:
        return BandRole.GRAY, "convention"
    if band_count == 3:
        return [BandRole.RED, BandRole.GREEN, BandRole.BLUE][index - 1], "convention"
    if band_count == 4:
        # RGB+NIR is the dominant 4-band convention (NAIP, Cartosat-2S MX).
        return [BandRole.RED, BandRole.GREEN, BandRole.BLUE, BandRole.NIR][
            index - 1
        ], "convention"

    return BandRole.UNKNOWN, "unknown"


def infer_modality(
    bands: list[BandInfo],
    tags: dict[str, str],
    dtype: str,
    band_count: int,
    sample_minimum: float | None,
) -> ModalityInference:
    """Decide whether a raster is optical or SAR, with stated reasons.

    Polarisation band names are near-conclusive. Failing that, file tags,
    then band count and dtype heuristics. The reasons list is surfaced in the
    execution trace so the decision is auditable rather than magic.
    """
    reasons: list[str] = []
    roles = {band.role for band in bands}

    sar_roles = roles & SAR_ROLES
    if sar_roles:
        names = ", ".join(sorted(role.value.upper() for role in sar_roles))
        reasons.append(f"Polarisation bands present ({names})")
        return ModalityInference(modality=Modality.SAR, confidence=0.98, reasons=reasons)

    optical_roles = roles & OPTICAL_ROLES
    if len(optical_roles) >= 3:
        names = ", ".join(sorted(role.value for role in optical_roles))
        reasons.append(f"{len(optical_roles)} named optical bands resolved ({names})")
        return ModalityInference(
            modality=Modality.OPTICAL, confidence=0.95, reasons=reasons
        )

    blob = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
    sar_hints = ("sentinel-1", "sentinel1", "s1a", "s1b", "risat", "sar", "grd",
                 "backscatter", "sigma0", "gamma0", "polarisation", "polarization")
    optical_hints = ("sentinel-2", "sentinel2", "msi", "cartosat", "landsat",
                     "resourcesat", "liss", "oli", "reflectance", "toa", "boa")

    hit_sar = [hint for hint in sar_hints if hint in blob]
    if hit_sar:
        reasons.append(f"File tags mention {', '.join(hit_sar[:3])}")
        return ModalityInference(modality=Modality.SAR, confidence=0.90, reasons=reasons)

    hit_optical = [hint for hint in optical_hints if hint in blob]
    if hit_optical:
        reasons.append(f"File tags mention {', '.join(hit_optical[:3])}")
        return ModalityInference(
            modality=Modality.OPTICAL, confidence=0.90, reasons=reasons
        )

    if len(optical_roles) >= 1:
        reasons.append(
            f"Optical band role resolved ({sorted(r.value for r in optical_roles)[0]})"
        )
        return ModalityInference(
            modality=Modality.OPTICAL, confidence=0.75, reasons=reasons
        )

    if band_count == 3 and dtype in ("uint8", "int8"):
        reasons.append("Three 8-bit bands, consistent with an RGB composite")
        return ModalityInference(
            modality=Modality.OPTICAL, confidence=0.75, reasons=reasons
        )

    if band_count >= 4:
        reasons.append(f"{band_count} bands, more than SAR products typically carry")
        return ModalityInference(
            modality=Modality.OPTICAL, confidence=0.65, reasons=reasons
        )

    if band_count <= 2 and dtype.startswith("float"):
        detail = "single floating-point channel"
        if sample_minimum is not None and sample_minimum < 0:
            detail += f", minimum {sample_minimum:.2f} suggests decibel scaling"
            confidence = 0.75
        else:
            confidence = 0.55
        reasons.append(f"{detail}, consistent with SAR backscatter")
        return ModalityInference(
            modality=Modality.SAR, confidence=confidence, reasons=reasons
        )

    reasons.append(
        f"No band names, tags, or dtype signature matched ({band_count} bands, {dtype})"
    )
    return ModalityInference(modality=Modality.UNKNOWN, confidence=0.30, reasons=reasons)


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------


def _decimation_for(width: int, height: int, target_pixels: int) -> int:
    total = max(1, width * height)
    if total <= target_pixels:
        return 1
    return max(1, math.ceil(math.sqrt(total / target_pixels)))


def compute_gsd(
    crs: Any,
    pixel_x: float,
    pixel_y: float,
    centre_lon: float | None,
    centre_lat: float | None,
) -> tuple[float | None, float | None, str | None, str | None]:
    """Convert native pixel size to metres.

    Returns ``(gsd_x_m, gsd_y_m, method, axis_unit)``.

    A projected CRS is scaled by its linear unit's metre factor, which handles
    the foot-based projections correctly. A geographic CRS is measured
    geodesically at the scene centre, because a degree of longitude shrinks with
    latitude and a single constant would be wrong away from the equator.
    """
    if crs is None:
        return None, None, None, None

    try:
        pcrs = PyprojCRS.from_user_input(crs)
    except Exception:  # noqa: BLE001 - malformed CRS should not abort ingest
        return None, None, None, None

    axis_unit = pcrs.axis_info[0].unit_name if pcrs.axis_info else None

    if pcrs.is_geographic:
        if centre_lon is None or centre_lat is None:
            return None, None, None, axis_unit
        # Geodesic distance spanned by one pixel at the scene centre.
        _, _, dx_m = GEOD.inv(centre_lon, centre_lat, centre_lon + pixel_x, centre_lat)
        _, _, dy_m = GEOD.inv(centre_lon, centre_lat, centre_lon, centre_lat + pixel_y)
        return abs(dx_m), abs(dy_m), "geodesic-at-centre", axis_unit

    factor = 1.0
    if pcrs.axis_info:
        factor = pcrs.axis_info[0].unit_conversion_factor or 1.0
    return (
        abs(pixel_x) * factor,
        abs(pixel_y) * factor,
        "projected-linear-unit",
        axis_unit,
    )


def _read_georeference(dataset: rasterio.DatasetReader) -> GeoReference:
    transform = dataset.transform
    crs = dataset.crs

    geo = GeoReference(
        transform=[transform.a, transform.b, transform.c,
                   transform.d, transform.e, transform.f],
        pixel_size_native_x=abs(transform.a),
        pixel_size_native_y=abs(transform.e),
    )

    if crs is not None:
        geo.crs_wkt = crs.to_wkt()
        try:
            geo.epsg = crs.to_epsg()
        except Exception:  # noqa: BLE001
            geo.epsg = None
        try:
            pcrs = PyprojCRS.from_user_input(crs)
            geo.crs_name = pcrs.name
            geo.is_projected = pcrs.is_projected
            geo.is_geographic = pcrs.is_geographic
        except Exception:  # noqa: BLE001
            pass

    bounds = dataset.bounds
    geo.bounds_native = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    centre_lon = centre_lat = None
    if crs is not None:
        try:
            west, south, east, north = transform_bounds(
                crs, "EPSG:4326", *bounds, densify_pts=21
            )
            geo.bounds_wgs84 = [west, south, east, north]
            centre_lon = (west + east) / 2.0
            centre_lat = (south + north) / 2.0
            geo.centroid_wgs84 = [centre_lon, centre_lat]
        except Exception as exc:  # noqa: BLE001
            logger.debug("WGS84 bounds transform failed: %s", exc)

    gsd_x, gsd_y, method, axis_unit = compute_gsd(
        crs, abs(transform.a), abs(transform.e), centre_lon, centre_lat
    )
    geo.gsd_x_m = gsd_x
    geo.gsd_y_m = gsd_y
    geo.gsd_method = method
    geo.axis_unit = axis_unit
    if gsd_x is not None and gsd_y is not None:
        geo.gsd_m = (gsd_x + gsd_y) / 2.0
        # Footprint from the pixel grid: rows x columns x pixel area. Preferred
        # over a geodesic polygon area because the grid is what was measured.
        geo.area_km2 = (dataset.width * gsd_x) * (dataset.height * gsd_y) / 1_000_000.0

    return geo


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def _parse_datetime(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None

    # TIFF's own format uses colons in the date part.
    candidates = [text]
    if re.match(r"^\d{4}:\d{2}:\d{2}", text):
        candidates.append(text.replace(":", "-", 2))
    candidates.append(text.replace("Z", "+00:00"))

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d",
                "%Y/%m/%d", "%d-%m-%Y", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract_acquisition_date(
    tags: dict[str, str], filename: str
) -> tuple[datetime | None, DateSource]:
    """Find the acquisition date from file tags, else from the filename."""
    upper = {key.upper(): value for key, value in tags.items()}
    for key in DATE_TAG_KEYS:
        raw = upper.get(key)
        if raw:
            parsed = _parse_datetime(raw)
            if parsed:
                source = (
                    DateSource.MANIFEST
                    if key == "SATQUERY_ACQUISITION_DATE"
                    else DateSource.GEOTIFF_TAG
                )
                return parsed, source

    match = _FILENAME_DATE.search(filename)
    if match:
        parsed = _parse_datetime("".join(match.groups()))
        if parsed:
            return parsed, DateSource.FILENAME

    return None, DateSource.UNKNOWN


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _band_stats(
    dataset: rasterio.DatasetReader, index: int, decimation: int
) -> BandStats:
    out_height = max(1, dataset.height // decimation)
    out_width = max(1, dataset.width // decimation)

    # masked=True honours both the nodata value and any internal/alpha mask.
    array = dataset.read(
        index,
        out_shape=(out_height, out_width),
        resampling=Resampling.nearest,
        masked=True,
    )
    data = np.ma.masked_invalid(array)

    total = int(data.size)
    valid = int(data.count())
    nodata_pixels = total - valid
    compressed = data.compressed()

    if compressed.size == 0:
        return BandStats(
            minimum=0.0, maximum=0.0, mean=0.0, stddev=0.0,
            valid_pixels=0, nodata_pixels=nodata_pixels,
            nodata_fraction=1.0,
            sample_fraction=total / max(1, dataset.width * dataset.height),
        )

    as_float = compressed.astype("float64", copy=False)
    p2, p98 = np.percentile(as_float, [2, 98])

    return BandStats(
        minimum=float(as_float.min()),
        maximum=float(as_float.max()),
        mean=float(as_float.mean()),
        stddev=float(as_float.std()),
        valid_pixels=valid,
        nodata_pixels=nodata_pixels,
        nodata_fraction=nodata_pixels / total if total else 0.0,
        sample_fraction=total / max(1, dataset.width * dataset.height),
        percentile_2=float(p2),
        percentile_98=float(p98),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(
    path: Path,
    original_filename: str | None = None,
    with_stats: bool = True,
) -> RasterMetadata:
    """Open a raster and extract its full metadata record."""
    path = Path(path)
    if not path.exists():
        raise RasterIngestError(f"File not found: {path.name}")

    filename = original_filename or path.name
    notes: list[str] = []

    try:
        dataset = rasterio.open(path)
    except Exception as exc:  # noqa: BLE001 - surfaces as a 422 to the client
        raise RasterIngestError(
            f"Could not open '{filename}' as a raster. GDAL reported: {exc}"
        ) from exc

    with dataset:
        driver = dataset.driver or "unknown"
        format_class = (
            FormatClass.GEOSPATIAL
            if driver in GEOSPATIAL_DRIVERS
            else FormatClass.BENCHMARK_RASTER
        )

        if dataset.count == 0:
            raise RasterIngestError(f"'{filename}' contains no raster bands.")

        tags = {str(k): str(v) for k, v in (dataset.tags() or {}).items()}
        decimation = (
            _decimation_for(dataset.width, dataset.height, STATS_TARGET_PIXELS)
            if with_stats
            else 1
        )

        bands: list[BandInfo] = []
        for index in range(1, dataset.count + 1):
            label = (dataset.descriptions[index - 1] or "").strip()
            band_tags = {str(k): str(v) for k, v in (dataset.tags(index) or {}).items()}
            role, role_source = resolve_band_role(
                label or None, index, dataset.count, band_tags
            )
            stats = _band_stats(dataset, index, decimation) if with_stats else None
            bands.append(
                BandInfo(
                    index=index,
                    label=label or f"Band {index}",
                    role=role,
                    role_source=role_source,
                    dtype=str(dataset.dtypes[index - 1]),
                    wavelength_nm=BAND_WAVELENGTH_NM.get(role),
                    stats=stats,
                )
            )

        geo = _read_georeference(dataset)
        if geo.epsg is None and format_class is FormatClass.GEOSPATIAL:
            notes.append(
                "No EPSG code could be resolved from the CRS; "
                "geospatial measurements may be unavailable."
            )
        if dataset.crs is None:
            notes.append(
                "File carries no CRS. Accepted for benchmark imagery, but area "
                "and distance measurements require georeferencing."
            )

        sample_minimum = next(
            (b.stats.minimum for b in bands if b.stats is not None), None
        )
        modality = infer_modality(
            bands, tags, str(dataset.dtypes[0]), dataset.count, sample_minimum
        )

        nodata_fraction = 0.0
        if with_stats and bands:
            fractions = [b.stats.nodata_fraction for b in bands if b.stats]
            nodata_fraction = max(fractions) if fractions else 0.0

        resolved = [b.role for b in bands if b.role is not BandRole.UNKNOWN]
        if not resolved:
            notes.append(
                "No band roles could be resolved, so spectral indices are "
                "unavailable for this input."
            )

        acquisition_date, date_source = extract_acquisition_date(tags, filename)
        total_pixels = dataset.width * dataset.height

        try:
            overviews = list(dataset.overviews(1))
        except Exception:  # noqa: BLE001
            overviews = []

        block = None
        try:
            block = list(dataset.block_shapes[0])
        except Exception:  # noqa: BLE001
            pass

        return RasterMetadata(
            original_filename=filename,
            size_bytes=path.stat().st_size,
            driver=driver,
            format_class=format_class,
            width=dataset.width,
            height=dataset.height,
            band_count=dataset.count,
            dtype=str(dataset.dtypes[0]),
            megapixels=round(total_pixels / 1_000_000.0, 3),
            geo=geo,
            bands=bands,
            resolved_roles=resolved,
            modality=modality,
            nodata_value=(
                float(dataset.nodata) if dataset.nodata is not None else None
            ),
            nodata_fraction=nodata_fraction,
            acquisition_date=acquisition_date,
            date_source=date_source,
            day_of_year=acquisition_date.timetuple().tm_yday if acquisition_date else None,
            tiling_recommended=total_pixels > TILING_THRESHOLD_PIXELS,
            overview_levels=overviews,
            block_shape=block,
            tags=tags,
            ingest_notes=notes,
        )


def _stretch(
    channel: np.ma.MaskedArray, exclude: np.ndarray | None = None
) -> np.ndarray:
    """Percentile stretch one channel to 0-255 for display only.

    Display stretching never feeds a measurement; masks and statistics are
    always computed from the raw values.

    ``exclude`` marks pixels to leave out of the percentile calculation. It is
    used to drop cloud, whose bright tops otherwise dominate the 98th percentile
    and crush the land surface to black.
    """
    if exclude is not None and exclude.shape == channel.shape:
        sampling = np.ma.masked_array(channel, mask=channel.mask | exclude)
    else:
        sampling = channel

    valid = sampling.compressed().astype("float64", copy=False)
    if valid.size < 16:
        valid = channel.compressed().astype("float64", copy=False)
    if valid.size == 0:
        return np.zeros(channel.shape, dtype="uint8")

    low, high = np.percentile(valid, [2, 98])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        return np.zeros(channel.shape, dtype="uint8")

    scaled = (channel.filled(low).astype("float64") - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype("uint8")


def _to_decibels(channel: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Convert linear SAR power to dB, leaving already-dB data untouched."""
    valid = channel.compressed()
    if valid.size and float(valid.min()) < 0:
        return channel  # already logarithmic
    positive = np.ma.masked_less_equal(channel, 0)
    return 10.0 * np.ma.log10(positive)


def build_thumbnail(path: Path, metadata: RasterMetadata, out_path: Path) -> str | None:
    """Render a display thumbnail and return a short description of the recipe."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    long_edge = max(metadata.width, metadata.height)
    decimation = max(1, math.ceil(long_edge / THUMBNAIL_MAX_EDGE))
    stack, recipe, _ = composite_rgb(
        path,
        metadata,
        max(1, metadata.height // decimation),
        max(1, metadata.width // decimation),
    )
    Image.fromarray(stack, mode="RGB").save(out_path, format="PNG", optimize=True)
    return recipe


def composite_rgb(
    path: Path, metadata: RasterMetadata, out_height: int, out_width: int
) -> tuple[np.ndarray, str, np.ndarray]:
    """Build the display composite for a raster at a requested size.

    Returns the ``HxWx3`` uint8 stack, a description of the recipe, and a mask of
    pixels that actually carry data. The recipe is returned because the UI must be
    able to state what is being shown: a SAR polarisation composite looks nothing
    like a photograph and should never be presented as one.

    Shared by the ingest thumbnail and the map base layer so that the two can
    never disagree about how a scene is rendered.
    """

    def grab(role: BandRole) -> np.ma.MaskedArray | None:
        index = metadata.band_index(role)
        if index is None:
            return None
        with rasterio.open(path) as dataset:
            return dataset.read(
                index,
                out_shape=(out_height, out_width),
                resampling=Resampling.average,
                masked=True,
            )

    def grab_index(index: int) -> np.ma.MaskedArray:
        with rasterio.open(path) as dataset:
            return dataset.read(
                index,
                out_shape=(out_height, out_width),
                resampling=Resampling.average,
                masked=True,
            )

    recipe: str
    channels: list[np.ndarray]

    # Sentinel-2 carries a scene classification layer. Excluding its cloud and
    # shadow classes from the display statistics keeps the land surface properly
    # exposed instead of letting cloud tops set the white point.
    cloud_exclude: np.ndarray | None = None
    cloud_note = ""
    if metadata.band_index(BandRole.SCL) is not None:
        scl_index = metadata.band_index(BandRole.SCL)
        assert scl_index is not None
        with rasterio.open(path) as dataset:
            scl = dataset.read(
                scl_index,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
                masked=True,
            )
        cloud_exclude = np.isin(np.ma.filled(scl, 0).astype("int16"), (3, 8, 9, 10))
        fraction = float(cloud_exclude.mean())
        if fraction > 0.02:
            cloud_note = f"; {fraction:.0%} cloud excluded from the display stretch"
        if fraction > 0.9:
            cloud_exclude = None  # almost everything is cloud; stretch on all of it

    if metadata.modality.modality is Modality.SAR:
        vv = grab(BandRole.VV) if metadata.band_index(BandRole.VV) else None
        vh = grab(BandRole.VH) if metadata.band_index(BandRole.VH) else None
        if vv is None and vh is None:
            vv = grab_index(1)
        if vv is not None and vh is not None:
            vv_db, vh_db = _to_decibels(vv), _to_decibels(vh)
            ratio = vv_db - vh_db
            channels = [_stretch(vv_db), _stretch(vh_db), _stretch(ratio)]
            recipe = "SAR false colour: R=VV dB, G=VH dB, B=VV-VH dB"
        else:
            single = vv if vv is not None else vh
            assert single is not None
            grey = _stretch(_to_decibels(single))
            channels = [grey, grey, grey]
            pol = "VV" if vv is not None else "VH"
            recipe = f"SAR greyscale: {pol} backscatter in dB"
    else:
        red, green, blue = (
            grab(BandRole.RED),
            grab(BandRole.GREEN),
            grab(BandRole.BLUE),
        )
        if red is not None and green is not None and blue is not None:
            channels = [
                _stretch(red, cloud_exclude),
                _stretch(green, cloud_exclude),
                _stretch(blue, cloud_exclude),
            ]
            recipe = f"True colour: R=red, G=green, B=blue{cloud_note}"
        else:
            # Explicit None check: `a or b` on a masked array evaluates its
            # truthiness, which raises for anything with more than one element.
            nir = grab(BandRole.NIR)
            if nir is None:
                nir = grab(BandRole.NIR08)
            if nir is not None and red is not None and green is not None:
                channels = [
                    _stretch(nir, cloud_exclude),
                    _stretch(red, cloud_exclude),
                    _stretch(green, cloud_exclude),
                ]
                recipe = f"False colour infrared: R=NIR, G=red, B=green{cloud_note}"
            elif metadata.band_count >= 3:
                channels = [_stretch(grab_index(i)) for i in (1, 2, 3)]
                recipe = "First three bands as R, G, B"
            else:
                grey = _stretch(grab_index(1))
                channels = [grey, grey, grey]
                recipe = "Greyscale: band 1"

    # Nodata is tracked separately from the stretch: a transparent border is the
    # honest rendering of "nothing was observed here", where black would read as a
    # dark surface.
    reference = grab_index(1)
    valid = ~np.ma.getmaskarray(reference)

    return np.dstack(channels), recipe, valid


# ---------------------------------------------------------------------------
# Band reads for analysis
# ---------------------------------------------------------------------------

# Preference order when a single band is needed to represent scene structure.
# NIR gives the strongest land/water and vegetation contrast in optical data;
# VV is the co-polarised channel and is present in nearly every SAR product.
STRUCTURE_PREFERENCE: tuple[BandRole, ...] = (
    BandRole.NIR,
    BandRole.NIR08,
    BandRole.VV,
    BandRole.VH,
    BandRole.RED,
    BandRole.SWIR16,
    BandRole.GREEN,
    BandRole.GRAY,
)


def reflectance_scale(metadata: RasterMetadata) -> tuple[float, str]:
    """Divisor that converts stored digital numbers to 0-1 reflectance.

    Spectral indices are ratios, so the scale cancels out of them. It matters
    only for checks with an absolute threshold, such as cloud brightness, which
    is why the rationale is returned alongside the number.
    """
    dtype = metadata.dtype
    sample_max = max(
        (band.stats.maximum for band in metadata.bands if band.stats is not None),
        default=None,
    )

    if dtype.startswith("float"):
        if sample_max is not None and sample_max <= 2.0:
            return 1.0, "float data already in 0-1 reflectance"
        return 10000.0, "float data with values above 1, assuming 10000 scaling"
    if dtype in ("uint8", "int8"):
        return 255.0, "8-bit data, scaled by 255"
    return 10000.0, "integer data, assuming Sentinel-2 style 10000 scaling"


def structure_band_index(metadata: RasterMetadata) -> int:
    """Index of the band that best represents scene structure."""
    for role in STRUCTURE_PREFERENCE:
        index = metadata.band_index(role)
        if index is not None:
            return index
    return 1


def intersect_bounds(
    a: list[float] | None, b: list[float] | None
) -> tuple[float, float, float, float] | None:
    """Intersection of two ``[minx, miny, maxx, maxy]`` boxes, or None."""
    if not a or not b:
        return None
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return None
    return left, bottom, right, top


def box_area(bounds: list[float] | tuple[float, ...] | None) -> float:
    if not bounds:
        return 0.0
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def read_band(
    path: Path,
    index: int,
    *,
    out_shape: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ma.MaskedArray:
    """Read one band, optionally windowed to bounds and resampled.

    Always returns a masked array so nodata never silently participates in a
    statistic.
    """
    from rasterio.windows import from_bounds

    with rasterio.open(path) as dataset:
        window = None
        if bounds is not None:
            window = from_bounds(*bounds, transform=dataset.transform)
        kwargs: dict[str, Any] = {"masked": True}
        if window is not None:
            kwargs["window"] = window
            kwargs["boundless"] = True
            kwargs["fill_value"] = (
                dataset.nodata if dataset.nodata is not None else 0
            )
        if out_shape is not None:
            kwargs["out_shape"] = out_shape
            kwargs["resampling"] = resampling
        array = dataset.read(index, **kwargs)
    return np.ma.masked_invalid(array)


def read_role(
    path: Path,
    metadata: RasterMetadata,
    role: BandRole,
    **kwargs: Any,
) -> np.ma.MaskedArray | None:
    """Read the band carrying a role, or None when the role is absent."""
    index = metadata.band_index(role)
    if index is None:
        return None
    return read_band(path, index, **kwargs)


def read_role_reflectance(
    path: Path,
    metadata: RasterMetadata,
    role: BandRole,
    **kwargs: Any,
) -> np.ma.MaskedArray | None:
    """Read a band and convert it to 0-1 reflectance."""
    raw = read_role(path, metadata, role, **kwargs)
    if raw is None:
        return None
    divisor, _ = reflectance_scale(metadata)
    return raw.astype("float64") / divisor
