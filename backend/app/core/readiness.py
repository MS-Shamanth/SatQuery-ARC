"""The data readiness gate.

Before any analysis runs, the input is checked for the properties the analysis
depends on. Every check reports the value it measured and the threshold it was
compared against, so a PASS is as auditable as a FAIL.

When the input cannot support the requested analysis the gate REFUSES and says
what would be needed instead. That refusal is a feature: answering a change
question from a pair that is misregistered by 40 m would produce a confident
number that means nothing.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.raster_io import (
    box_area,
    intersect_bounds,
    read_band,
    read_role,
    read_role_reflectance,
    reflectance_scale,
    structure_band_index,
)
from app.core.sessions import SessionStore
from app.models.schemas import (
    BandRole,
    CheckStatus,
    DataRequirement,
    ImageRole,
    InputConfiguration,
    Modality,
    RasterMetadata,
    ReadinessCheck,
    ReadinessReport,
    ReadinessVerdict,
    SessionRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds. Collected here so they are documented in one place and can be
# asserted in tests rather than being buried as literals in the logic.
# ---------------------------------------------------------------------------

NODATA_WARN = 0.10
NODATA_FAIL = 0.60

CLOUD_WARN = 0.20
CLOUD_FAIL = 0.60

# Sub-pixel registration is required for pixel-wise change detection. Beyond two
# pixels the change map is dominated by edge artefacts rather than real change.
COREG_PASS_PX = 1.0
COREG_WARN_PX = 2.0

OVERLAP_WARN = 0.90
OVERLAP_FAIL = 0.50

GSD_RATIO_WARN = 1.5
GSD_RATIO_FAIL = 3.0

MODALITY_CONFIDENCE_WARN = 0.60

# A difference of more than this many months in acquisition day-of-year makes
# phenology a plausible alternative explanation for any vegetation change.
SEASONAL_MONTH_DELTA = 2

# Largest window read at native resolution for registration estimation.
# Cropping rather than downsampling preserves sub-pixel sensitivity.
COREG_MAX_SAMPLE_PX = 900

# A shift larger than this is treated as an estimator failure rather than a
# measurement. Products whose geolocation is out by 25 pixels are not "slightly
# misregistered", and across modalities a bright feature present in only one
# image can drag the correlation peak arbitrarily far.
COREG_MAX_PLAUSIBLE_PX = 25.0

# Sentinel-2 Scene Classification Layer values that indicate cloud or shadow.
SCL_CLOUD_CLASSES: dict[int, str] = {
    3: "cloud shadow",
    8: "cloud, medium probability",
    9: "cloud, high probability",
    10: "thin cirrus",
}

# Cloud proxy thresholds, in 0-1 reflectance, used only when no SCL band exists.
HEURISTIC_BLUE_REFLECTANCE = 0.25
HEURISTIC_MAX_NDVI = 0.10


# ---------------------------------------------------------------------------
# Co-registration
# ---------------------------------------------------------------------------


@dataclass
class CoregistrationResult:
    dx_px: float
    dy_px: float
    shift_grid_px: float
    shift_m: float
    shift_native_px: float
    grid_pixel_m: float
    sample_shape: tuple[int, int]
    method: str
    # Whether the estimate survived validation. An unreliable estimate is worse
    # than no estimate, because a confident wrong number gets believed.
    reliable: bool = True
    reliability_note: str = ""
    identical_grid: bool = False


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Mutual information between two images.

    Used to validate a shift estimate across modalities. Correlation assumes a
    linear relationship between brightness values, which optical reflectance and
    radar backscatter do not have; mutual information only assumes that one is
    informative about the other.
    """
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 64:
        return 0.0
    hist, _, _ = np.histogram2d(a[finite].ravel(), b[finite].ravel(), bins=bins)
    total = hist.sum()
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nonzero = pxy > 0
    outer = px[:, None] * py[None, :]
    return float((pxy[nonzero] * np.log(pxy[nonzero] / outer[nonzero])).sum())


def validate_shift(
    reference: np.ndarray, moving: np.ndarray, dy: int, dx: int
) -> tuple[float, float] | None:
    """Mutual information before and after applying a candidate shift.

    Both measurements are taken over the *same* reference pixels, with only the
    sampling position in the moving image changing. Comparing MI across
    differently sized crops would be meaningless, because MI rises as the sample
    shrinks for a fixed bin count.

    Returns ``None`` when the shift is so large that no adequately sized common
    region survives, which is itself evidence that the estimate is not usable.
    """
    height, width = reference.shape
    pad_y, pad_x = abs(dy), abs(dx)
    top, bottom = pad_y, height - pad_y
    left, right = pad_x, width - pad_x
    if bottom - top < 32 or right - left < 32:
        return None

    anchor = reference[top:bottom, left:right]
    unshifted = moving[top:bottom, left:right]
    shifted = moving[top - dy : bottom - dy, left - dx : right - dx]
    if shifted.shape != anchor.shape:
        return None

    return (
        _mutual_information(anchor, unshifted),
        _mutual_information(anchor, shifted),
    )


def estimate_coregistration(
    path_a: Path,
    meta_a: RasterMetadata,
    path_b: Path,
    meta_b: RasterMetadata,
) -> CoregistrationResult | None:
    """Measure the translation between two rasters by phase cross-correlation.

    A central window of the overlap is read at image A's native resolution.
    Cropping instead of downsampling keeps the sub-pixel estimate meaningful:
    downsampling by four would turn a one-pixel shift into a quarter-pixel one.

    For a cross-modal pair the correlation runs on gradient magnitude rather than
    intensity. Optical brightness and radar backscatter are not comparable
    quantities, but the edges of the same physical features are.
    """
    from skimage.filters import sobel
    from skimage.registration import phase_cross_correlation

    overlap = intersect_bounds(meta_a.geo.bounds_native, meta_b.geo.bounds_native)
    if overlap is None:
        return None

    gsd = meta_a.geo.gsd_x_m or meta_a.geo.pixel_size_native_x
    if not gsd or gsd <= 0:
        return None

    # Native pixel size in CRS units, which is what the bounds are expressed in.
    native_x = meta_a.geo.pixel_size_native_x or 1.0
    native_y = meta_a.geo.pixel_size_native_y or native_x

    left, bottom, right, top = overlap
    span_x = min(right - left, COREG_MAX_SAMPLE_PX * native_x)
    span_y = min(top - bottom, COREG_MAX_SAMPLE_PX * native_y)
    centre_x = (left + right) / 2.0
    centre_y = (bottom + top) / 2.0
    window = (
        centre_x - span_x / 2.0,
        centre_y - span_y / 2.0,
        centre_x + span_x / 2.0,
        centre_y + span_y / 2.0,
    )

    out_w = max(16, int(round(span_x / native_x)))
    out_h = max(16, int(round(span_y / native_y)))

    index_a = structure_band_index(meta_a)
    index_b = structure_band_index(meta_b)
    try:
        arr_a = read_band(path_a, index_a, out_shape=(out_h, out_w), bounds=window)
        arr_b = read_band(path_b, index_b, out_shape=(out_h, out_w), bounds=window)
    except Exception as exc:  # noqa: BLE001 - a read failure is reported, not fatal
        logger.warning("co-registration read failed: %s", exc)
        return None

    cross_modal = meta_a.modality.modality is not meta_b.modality.modality

    # Cloud is present in optical and absent in radar. Left in, a bright cloud
    # bank contributes strong edges with no counterpart in the other image and
    # can pull the correlation peak metres off.
    excluded_note = ""
    cloud_masks: list[np.ndarray] = []
    for path, meta in ((path_a, meta_a), (path_b, meta_b)):
        scl = read_role(path, meta, BandRole.SCL, out_shape=(out_h, out_w), bounds=window)
        if scl is None:
            cloud_masks.append(np.zeros((out_h, out_w), dtype=bool))
            continue
        cloud_masks.append(
            np.isin(np.ma.filled(scl, 0).astype("int16"), list(SCL_CLOUD_CLASSES))
        )
    shared_exclude = cloud_masks[0] | cloud_masks[1]
    if shared_exclude.any():
        fraction = float(shared_exclude.mean())
        if fraction > 0.75:
            shared_exclude = np.zeros_like(shared_exclude)
        else:
            excluded_note = f", {fraction:.0%} cloud excluded"

    def prepare(array: np.ma.MaskedArray) -> np.ndarray:
        invalid = np.ma.getmaskarray(array) | shared_exclude
        usable = np.ma.masked_array(array, mask=invalid)
        fill = float(usable.mean()) if usable.count() else 0.0
        filled = np.asarray(array, dtype="float64").copy()
        filled[invalid] = fill
        spread = filled.std()
        normed = (filled - filled.mean()) / (spread if spread > 0 else 1.0)
        return sobel(normed) if cross_modal else normed

    reference = prepare(arr_a)
    moving = prepare(arr_b)
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        return None

    # Phase normalisation is the more robust choice across modalities, where the
    # two images share structure but not brightness. Within one modality plain
    # cross-correlation is steadier on speckled, low-texture scenes.
    shift, _error, _phasediff = phase_cross_correlation(
        reference,
        moving,
        upsample_factor=20,
        normalization="phase" if cross_modal else None,
    )
    dy, dx = float(shift[0]), float(shift[1])

    grid_pixel_m = (span_x / out_w) * (gsd / native_x)
    shift_grid_px = math.hypot(dx, dy)

    identical_grid = (
        meta_a.geo.transform == meta_b.geo.transform
        and (meta_a.width, meta_a.height) == (meta_b.width, meta_b.height)
    )

    reliable = True
    note = ""

    if shift_grid_px >= 0.5:
        # Validate the estimate rather than trusting the correlation peak:
        # applying the shift must make the two images more informative about each
        # other. This decides the question on its own, so a genuinely large
        # offset is still reported as a large offset.
        validation = validate_shift(reference, moving, round(dy), round(dx))
        if validation is None:
            reliable = False
            note = (
                f"the correlation peak sits {shift_grid_px:.0f} px away, which "
                "leaves too little common area to check whether the shift is real"
            )
        else:
            baseline, improved = validation
            if improved <= baseline * 1.01:
                reliable = False
                note = (
                    "applying the estimated shift did not increase mutual "
                    f"information ({baseline:.3f} to {improved:.3f}), so the "
                    "correlation peak does not correspond to a real offset"
                )
                if shift_grid_px > COREG_MAX_PLAUSIBLE_PX:
                    note += (
                        f"; the peak also sits {shift_grid_px:.0f} px away, beyond "
                        f"the {COREG_MAX_PLAUSIBLE_PX:.0f} px bound for a plausible "
                        "product offset"
                    )

    return CoregistrationResult(
        dx_px=dx,
        dy_px=dy,
        shift_grid_px=shift_grid_px,
        shift_m=shift_grid_px * grid_pixel_m,
        shift_native_px=shift_grid_px * grid_pixel_m / gsd,
        grid_pixel_m=grid_pixel_m,
        sample_shape=(out_h, out_w),
        method=(
            f"phase cross-correlation on gradient magnitude{excluded_note}"
            if cross_modal
            else f"phase cross-correlation on intensity{excluded_note}"
        ),
        reliable=reliable,
        reliability_note=note,
        identical_grid=identical_grid,
    )


# ---------------------------------------------------------------------------
# Cloud and shadow
# ---------------------------------------------------------------------------


@dataclass
class CloudEstimate:
    fraction: float | None
    method: str
    detail: str


def estimate_cloud_fraction(path: Path, metadata: RasterMetadata) -> CloudEstimate:
    """Fraction of the scene under cloud or cloud shadow.

    Sentinel-2's Scene Classification Layer is used when present, which is a real
    classification rather than an approximation. Otherwise a brightness proxy is
    used, and the method string says so, because a proxy and a classification do
    not deserve equal confidence.
    """
    if metadata.modality.modality is Modality.SAR:
        return CloudEstimate(
            fraction=None,
            method="not-applicable-sar",
            detail="Radar penetrates cloud, so cloud cover does not apply.",
        )

    scl = read_role(path, metadata, BandRole.SCL, out_shape=(512, 512))
    if scl is not None and scl.count():
        values = scl.astype("int16")
        total = int(values.count())
        cloudy = int(
            np.isin(values.compressed(), list(SCL_CLOUD_CLASSES.keys())).sum()
        )
        present = sorted(
            {
                SCL_CLOUD_CLASSES[int(v)]
                for v in np.unique(values.compressed())
                if int(v) in SCL_CLOUD_CLASSES
            }
        )
        return CloudEstimate(
            fraction=cloudy / total if total else 0.0,
            method="sentinel-2-scl",
            detail=(
                f"Scene Classification Layer classes {sorted(SCL_CLOUD_CLASSES)}"
                + (f"; found {', '.join(present)}" if present else "; none present")
            ),
        )

    blue = read_role_reflectance(path, metadata, BandRole.BLUE, out_shape=(512, 512))
    red = read_role_reflectance(path, metadata, BandRole.RED, out_shape=(512, 512))
    nir = read_role_reflectance(path, metadata, BandRole.NIR, out_shape=(512, 512))
    if nir is None:
        nir = read_role_reflectance(path, metadata, BandRole.NIR08, out_shape=(512, 512))

    if blue is None or red is None or nir is None:
        missing = [
            role
            for role, array in (("blue", blue), ("red", red), ("nir", nir))
            if array is None
        ]
        return CloudEstimate(
            fraction=None,
            method="unavailable",
            detail=(
                "No SCL band and the brightness proxy needs "
                f"{', '.join(missing)}, which are absent."
            ),
        )

    denominator = nir + red
    ndvi = np.ma.where(denominator != 0, (nir - red) / denominator, 0.0)
    cloudy_mask = (blue > HEURISTIC_BLUE_REFLECTANCE) & (ndvi < HEURISTIC_MAX_NDVI)

    valid = int(blue.count())
    cloudy = int(np.ma.filled(cloudy_mask, False).sum())
    _, scale_note = reflectance_scale(metadata)

    return CloudEstimate(
        fraction=cloudy / valid if valid else None,
        method="brightness-heuristic",
        detail=(
            f"Proxy: blue reflectance above {HEURISTIC_BLUE_REFLECTANCE:.2f} "
            f"with NDVI below {HEURISTIC_MAX_NDVI:.2f} ({scale_note}). "
            "Less reliable than a scene classification layer."
        ),
    )


# ---------------------------------------------------------------------------
# Per-image checks
# ---------------------------------------------------------------------------


def _check_crs(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    if meta.geo.epsg is not None:
        return ReadinessCheck(
            id=f"crs.{role.value}",
            label="Coordinate reference system",
            status=CheckStatus.PASS,
            measured=f"EPSG:{meta.geo.epsg}",
            measured_numeric=float(meta.geo.epsg),
            threshold="a resolvable CRS",
            message=f"{meta.geo.crs_name or 'CRS'} resolved from the file.",
            method="rasterio dataset CRS",
            applies_to=[role],
        )
    if meta.geo.crs_wkt is not None:
        return ReadinessCheck(
            id=f"crs.{role.value}",
            label="Coordinate reference system",
            status=CheckStatus.WARN,
            measured="CRS present, no EPSG code",
            threshold="a resolvable CRS",
            message=(
                "The file has a CRS but it does not map to an EPSG code. "
                "Measurements remain possible; reprojection may be imprecise."
            ),
            method="rasterio dataset CRS",
            applies_to=[role],
        )
    return ReadinessCheck(
        id=f"crs.{role.value}",
        label="Coordinate reference system",
        status=CheckStatus.FAIL,
        measured="absent",
        threshold="a resolvable CRS",
        message=(
            "No coordinate reference system. Areas, distances, and map overlays "
            "cannot be produced from this file."
        ),
        method="rasterio dataset CRS",
        applies_to=[role],
    )


def _check_resolution(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    gsd = meta.geo.gsd_m
    if gsd is None:
        return ReadinessCheck(
            id=f"resolution.{role.value}",
            label="Ground sample distance",
            status=CheckStatus.FAIL,
            measured="unknown",
            threshold="derivable from the geotransform",
            message="Pixel size could not be converted to metres without a CRS.",
            applies_to=[role],
        )
    return ReadinessCheck(
        id=f"resolution.{role.value}",
        label="Ground sample distance",
        status=CheckStatus.PASS,
        measured=f"{gsd:.2f} m" if gsd < 10 else f"{gsd:.1f} m",
        measured_numeric=gsd,
        unit="m",
        threshold="derivable from the geotransform",
        message=(
            f"{meta.geo.gsd_x_m:.2f} m across by {meta.geo.gsd_y_m:.2f} m down."
            if meta.geo.gsd_x_m and meta.geo.gsd_y_m
            else "Derived from the affine transform."
        ),
        method=meta.geo.gsd_method,
        applies_to=[role],
    )


def _check_bands(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    resolved = len(meta.resolved_roles)
    names = ", ".join(r.value for r in meta.resolved_roles) or "none"
    if resolved == 0:
        return ReadinessCheck(
            id=f"bands.{role.value}",
            label="Band roles",
            status=CheckStatus.FAIL,
            measured=f"0 of {meta.band_count}",
            measured_numeric=0.0,
            threshold="at least 1 identified band",
            message=(
                "No band could be identified, so no spectral index or "
                "polarisation analysis is possible."
            ),
            applies_to=[role],
        )
    inferred = [band for band in meta.bands if band.role_source == "convention"]
    if inferred:
        return ReadinessCheck(
            id=f"bands.{role.value}",
            label="Band roles",
            status=CheckStatus.WARN,
            measured=f"{resolved} of {meta.band_count}",
            measured_numeric=float(resolved),
            threshold="at least 1 identified band",
            message=(
                f"Resolved {names}. {len(inferred)} band(s) were inferred from "
                "band-count convention rather than declared in the file."
            ),
            method="band descriptions and positional convention",
            applies_to=[role],
        )
    return ReadinessCheck(
        id=f"bands.{role.value}",
        label="Band roles",
        status=CheckStatus.PASS,
        measured=f"{resolved} of {meta.band_count}",
        measured_numeric=float(resolved),
        threshold="at least 1 identified band",
        message=f"Resolved {names} from band descriptions.",
        method="band descriptions",
        applies_to=[role],
    )


def _check_nodata(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    fraction = meta.nodata_fraction
    status = (
        CheckStatus.FAIL
        if fraction > NODATA_FAIL
        else CheckStatus.WARN
        if fraction > NODATA_WARN
        else CheckStatus.PASS
    )
    message = {
        CheckStatus.PASS: "Coverage is essentially complete.",
        CheckStatus.WARN: "Gaps will reduce the area available for analysis.",
        CheckStatus.FAIL: "Most of the scene carries no data.",
    }[status]
    return ReadinessCheck(
        id=f"nodata.{role.value}",
        label="Valid pixel coverage",
        status=status,
        measured=f"{fraction * 100:.1f}% nodata",
        measured_numeric=fraction,
        unit="fraction",
        threshold=f"warn above {NODATA_WARN:.0%}, fail above {NODATA_FAIL:.0%}",
        message=message,
        method="masked read across all bands, worst band reported",
        applies_to=[role],
    )


def _check_cloud(role: ImageRole, path: Path, meta: RasterMetadata) -> ReadinessCheck:
    estimate = estimate_cloud_fraction(path, meta)
    if estimate.fraction is None:
        return ReadinessCheck(
            id=f"cloud.{role.value}",
            label="Cloud and shadow",
            status=CheckStatus.NOT_APPLICABLE,
            measured="not assessed",
            threshold=f"warn above {CLOUD_WARN:.0%}, fail above {CLOUD_FAIL:.0%}",
            message=estimate.detail,
            method=estimate.method,
            applies_to=[role],
        )

    fraction = estimate.fraction
    status = (
        CheckStatus.FAIL
        if fraction > CLOUD_FAIL
        else CheckStatus.WARN
        if fraction > CLOUD_WARN
        else CheckStatus.PASS
    )
    return ReadinessCheck(
        id=f"cloud.{role.value}",
        label="Cloud and shadow",
        status=status,
        measured=f"{fraction * 100:.1f}%",
        measured_numeric=fraction,
        unit="fraction",
        threshold=f"warn above {CLOUD_WARN:.0%}, fail above {CLOUD_FAIL:.0%}",
        message=estimate.detail,
        method=estimate.method,
        applies_to=[role],
    )


def _check_modality(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    inference = meta.modality
    if inference.modality is Modality.UNKNOWN:
        status = CheckStatus.FAIL
        message = (
            "Could not tell whether this is optical or radar data, so no "
            "specialist tool can be selected for it."
        )
    elif inference.confidence < MODALITY_CONFIDENCE_WARN:
        status = CheckStatus.WARN
        message = f"Low confidence. {' '.join(inference.reasons)}"
    else:
        status = CheckStatus.PASS
        message = " ".join(inference.reasons)
    return ReadinessCheck(
        id=f"modality.{role.value}",
        label="Sensor modality",
        status=status,
        measured=inference.modality.value,
        measured_numeric=inference.confidence,
        threshold=f"confidence at or above {MODALITY_CONFIDENCE_WARN:.0%}",
        message=message,
        method="band names, file tags, dtype and band count",
        applies_to=[role],
    )


def _check_tiling(role: ImageRole, meta: RasterMetadata) -> ReadinessCheck:
    return ReadinessCheck(
        id=f"tiling.{role.value}",
        label="Processing strategy",
        status=CheckStatus.PASS,
        measured=(
            f"{meta.megapixels:.1f} MP, windowed reads"
            if meta.tiling_recommended
            else f"{meta.megapixels:.1f} MP, full read"
        ),
        measured_numeric=meta.megapixels,
        unit="megapixels",
        threshold="windowed above 16.8 MP",
        message=(
            "Large raster: analysis will read in windows and build overviews."
            if meta.tiling_recommended
            else "Small enough to process in one pass."
        ),
        method="pixel count from the raster dimensions",
        applies_to=[role],
    )


# ---------------------------------------------------------------------------
# Pair checks
# ---------------------------------------------------------------------------


def _na(check_id: str, label: str, message: str) -> ReadinessCheck:
    return ReadinessCheck(
        id=check_id,
        label=label,
        status=CheckStatus.NOT_APPLICABLE,
        measured="not applicable",
        message=message,
    )


def _check_crs_match(
    roles: tuple[ImageRole, ImageRole], metas: tuple[RasterMetadata, RasterMetadata]
) -> ReadinessCheck:
    a, b = metas
    same = a.geo.epsg is not None and a.geo.epsg == b.geo.epsg
    if same:
        return ReadinessCheck(
            id="pair.crs_match",
            label="Matching projections",
            status=CheckStatus.PASS,
            measured=f"both EPSG:{a.geo.epsg}",
            measured_numeric=float(a.geo.epsg),
            threshold="identical EPSG codes",
            message="Both images share a coordinate reference system.",
            method="EPSG comparison",
            applies_to=list(roles),
        )
    return ReadinessCheck(
        id="pair.crs_match",
        label="Matching projections",
        status=CheckStatus.FAIL,
        measured=f"EPSG:{a.geo.epsg} vs EPSG:{b.geo.epsg}",
        threshold="identical EPSG codes",
        message=(
            "The two images are in different projections. Comparing them "
            "pixel-by-pixel would compare different ground locations."
        ),
        method="EPSG comparison",
        applies_to=list(roles),
    )


def _check_resolution_match(
    roles: tuple[ImageRole, ImageRole], metas: tuple[RasterMetadata, RasterMetadata]
) -> ReadinessCheck:
    a, b = metas
    if not a.geo.gsd_m or not b.geo.gsd_m:
        return _na(
            "pair.resolution_match",
            "Comparable resolution",
            "Resolution is unknown for at least one image.",
        )
    ratio = max(a.geo.gsd_m, b.geo.gsd_m) / min(a.geo.gsd_m, b.geo.gsd_m)
    status = (
        CheckStatus.FAIL
        if ratio > GSD_RATIO_FAIL
        else CheckStatus.WARN
        if ratio > GSD_RATIO_WARN
        else CheckStatus.PASS
    )
    message = {
        CheckStatus.PASS: "Resolutions are close enough to compare directly.",
        CheckStatus.WARN: (
            "Resolutions differ. The coarser image sets the effective detail, "
            "and small features may be resolved in only one of them."
        ),
        CheckStatus.FAIL: (
            "Resolutions differ too much to compare meaningfully. Features "
            "visible in the finer image have no counterpart in the coarser one."
        ),
    }[status]
    return ReadinessCheck(
        id="pair.resolution_match",
        label="Comparable resolution",
        status=status,
        measured=f"{a.geo.gsd_m:.1f} m vs {b.geo.gsd_m:.1f} m ({ratio:.2f}x)",
        measured_numeric=ratio,
        unit="ratio",
        threshold=f"warn above {GSD_RATIO_WARN}x, fail above {GSD_RATIO_FAIL}x",
        message=message,
        method="ratio of ground sample distances",
        applies_to=list(roles),
    )


def _check_overlap(
    roles: tuple[ImageRole, ImageRole], metas: tuple[RasterMetadata, RasterMetadata]
) -> tuple[ReadinessCheck, tuple[float, ...] | None, float | None]:
    a, b = metas
    overlap = intersect_bounds(a.geo.bounds_native, b.geo.bounds_native)
    if a.geo.epsg is None or a.geo.epsg != b.geo.epsg:
        return (
            _na(
                "pair.overlap",
                "Spatial overlap",
                "Overlap cannot be measured until both images share a projection.",
            ),
            None,
            None,
        )
    if overlap is None:
        return (
            ReadinessCheck(
                id="pair.overlap",
                label="Spatial overlap",
                status=CheckStatus.FAIL,
                measured="0%",
                measured_numeric=0.0,
                unit="fraction",
                threshold=f"fail below {OVERLAP_FAIL:.0%}",
                message=(
                    "The two images cover different ground with no common area, "
                    "so there is nothing to compare."
                ),
                method="intersection of raster footprints",
                applies_to=list(roles),
            ),
            None,
            0.0,
        )

    smaller = min(box_area(a.geo.bounds_native), box_area(b.geo.bounds_native))
    fraction = box_area(list(overlap)) / smaller if smaller else 0.0
    status = (
        CheckStatus.FAIL
        if fraction < OVERLAP_FAIL
        else CheckStatus.WARN
        if fraction < OVERLAP_WARN
        else CheckStatus.PASS
    )
    message = {
        CheckStatus.PASS: "The footprints coincide almost entirely.",
        CheckStatus.WARN: (
            "Only part of each scene is shared. Analysis is confined to the "
            "overlapping area."
        ),
        CheckStatus.FAIL: "Too little common ground to support a comparison.",
    }[status]
    return (
        ReadinessCheck(
            id="pair.overlap",
            label="Spatial overlap",
            status=status,
            measured=f"{fraction * 100:.1f}% of the smaller scene",
            measured_numeric=fraction,
            unit="fraction",
            threshold=f"warn below {OVERLAP_WARN:.0%}, fail below {OVERLAP_FAIL:.0%}",
            message=message,
            method="intersection of raster footprints",
            applies_to=list(roles),
        ),
        overlap,
        fraction,
    )


def _check_coregistration(
    roles: tuple[ImageRole, ImageRole],
    paths: tuple[Path, Path],
    metas: tuple[RasterMetadata, RasterMetadata],
) -> ReadinessCheck:
    a, b = metas
    if a.geo.epsg is None or a.geo.epsg != b.geo.epsg:
        return _na(
            "pair.coregistration",
            "Co-registration",
            "Registration can only be measured once both images share a projection.",
        )

    result = estimate_coregistration(paths[0], a, paths[1], b)
    if result is None:
        return _na(
            "pair.coregistration",
            "Co-registration",
            "Not enough overlapping, valid imagery to estimate a shift.",
        )

    if not result.reliable:
        # The content-based estimate could not be trusted. Where the two rasters
        # already occupy the same grid, the georeferencing is the better
        # authority and alignment is exact by construction. Where they do not,
        # say plainly that alignment is unverified rather than inventing a value.
        if result.identical_grid:
            return ReadinessCheck(
                id="pair.coregistration",
                label="Co-registration",
                status=CheckStatus.PASS,
                measured="aligned by georeferencing",
                measured_numeric=0.0,
                unit="pixels",
                threshold=(
                    f"pass at or below {COREG_PASS_PX} px, "
                    f"fail above {COREG_WARN_PX} px"
                ),
                message=(
                    "Both rasters occupy an identical grid, so they sample the same "
                    "ground by construction. The content correlation was discarded "
                    f"because {result.reliability_note}."
                ),
                method=f"{result.method}; estimate rejected, grid comparison used",
                applies_to=list(roles),
            )
        return ReadinessCheck(
            id="pair.coregistration",
            label="Co-registration",
            status=CheckStatus.WARN,
            measured="not verified",
            unit="pixels",
            threshold=(
                f"pass at or below {COREG_PASS_PX} px, fail above {COREG_WARN_PX} px"
            ),
            message=(
                "Alignment could not be confirmed from image content: "
                f"{result.reliability_note}. Treat any pixel-wise comparison with "
                "caution."
            ),
            method=result.method,
            applies_to=list(roles),
        )

    px = result.shift_native_px
    status = (
        CheckStatus.PASS
        if px <= COREG_PASS_PX
        else CheckStatus.WARN
        if px <= COREG_WARN_PX
        else CheckStatus.FAIL
    )
    message = {
        CheckStatus.PASS: (
            "Sub-pixel alignment. Pixel-wise comparison is sound."
        ),
        CheckStatus.WARN: (
            "Slight misalignment. Thin features and edges may register as false "
            "change, and the confounder tests will account for it."
        ),
        CheckStatus.FAIL: (
            "The images are offset by more than two pixels. A change map built "
            "from this pair would mostly show registration artefacts."
        ),
    }[status]
    return ReadinessCheck(
        id="pair.coregistration",
        label="Co-registration",
        status=status,
        measured=f"{px:.2f} px ({result.shift_m:.2f} m)",
        measured_numeric=px,
        unit="pixels",
        threshold=f"pass at or below {COREG_PASS_PX} px, fail above {COREG_WARN_PX} px",
        message=(
            f"{message} Offset dx {result.dx_px:+.2f}, dy {result.dy_px:+.2f} on a "
            f"{result.sample_shape[1]}x{result.sample_shape[0]} sample."
        ),
        method=result.method,
        applies_to=list(roles),
    )


def _check_temporal(
    roles: tuple[ImageRole, ImageRole], metas: tuple[RasterMetadata, RasterMetadata]
) -> tuple[ReadinessCheck, int | None, int | None, bool]:
    a, b = metas
    if a.acquisition_date is None or b.acquisition_date is None:
        which = [
            role.value
            for role, meta in zip(roles, metas, strict=True)
            if meta.acquisition_date is None
        ]
        return (
            ReadinessCheck(
                id="pair.temporal",
                label="Temporal separation",
                status=CheckStatus.WARN,
                measured="unknown",
                threshold="both acquisition dates known",
                message=(
                    f"No acquisition date for {', '.join(which)}. Change can still "
                    "be measured, but seasonality cannot be ruled out without "
                    "knowing when each image was taken."
                ),
                method="file tags and filename",
                applies_to=list(roles),
            ),
            None,
            None,
            True,
        )

    day_delta = abs((b.acquisition_date - a.acquisition_date).days)
    doy_a = a.day_of_year or 0
    doy_b = b.day_of_year or 0
    circular = min(abs(doy_b - doy_a), 365 - abs(doy_b - doy_a))
    month_delta = int(round(circular / 30.4))
    seasonal = month_delta >= SEASONAL_MONTH_DELTA

    if day_delta == 0:
        status = CheckStatus.WARN
        message = "Both images carry the same date, so there is no interval to compare."
    elif seasonal:
        status = CheckStatus.WARN
        message = (
            f"The acquisitions sit about {month_delta} months apart in the annual "
            "cycle. Phenology is a live alternative explanation for any "
            "vegetation change, and the seasonality test will be run."
        )
    else:
        status = CheckStatus.PASS
        message = (
            f"{day_delta} days apart and within {month_delta} month(s) of the same "
            "point in the annual cycle, so seasonal bias is limited."
        )

    return (
        ReadinessCheck(
            id="pair.temporal",
            label="Temporal separation",
            status=status,
            measured=f"{day_delta} days apart, {month_delta} month(s) of seasonal offset",
            measured_numeric=float(day_delta),
            unit="days",
            threshold=f"seasonal risk flagged at {SEASONAL_MONTH_DELTA} months offset",
            message=message,
            method="acquisition dates and day-of-year difference",
            applies_to=list(roles),
        ),
        day_delta,
        month_delta,
        seasonal,
    )


def _check_modality_complement(
    roles: tuple[ImageRole, ImageRole], metas: tuple[RasterMetadata, RasterMetadata]
) -> ReadinessCheck:
    a, b = metas
    kinds = {a.modality.modality, b.modality.modality}
    if kinds == {Modality.OPTICAL, Modality.SAR}:
        return ReadinessCheck(
            id="pair.modality_complement",
            label="Complementary modalities",
            status=CheckStatus.PASS,
            measured="optical + SAR",
            threshold="one optical and one SAR image",
            message=(
                "Spectral and structural information are both available, which is "
                "what makes a cross-modal comparison worthwhile."
            ),
            method="per-image modality inference",
            applies_to=list(roles),
        )
    only = next(iter(kinds)).value if len(kinds) == 1 else "/".join(
        sorted(k.value for k in kinds)
    )
    return ReadinessCheck(
        id="pair.modality_complement",
        label="Complementary modalities",
        status=CheckStatus.FAIL,
        measured=only,
        threshold="one optical and one SAR image",
        message=(
            "A cross-modal analysis needs one optical and one radar image. "
            f"Both inputs were read as {only}."
        ),
        method="per-image modality inference",
        applies_to=list(roles),
    )


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------

_REQUIREMENT_BY_CHECK: dict[str, DataRequirement] = {
    "crs": DataRequirement(
        what="A georeferenced GeoTIFF carrying a CRS",
        why="Area, distance, and map overlays are impossible without one.",
    ),
    "resolution": DataRequirement(
        what="An image whose geotransform yields a pixel size in metres",
        why="Every area measurement is pixel count multiplied by pixel area.",
    ),
    "bands": DataRequirement(
        what="Band descriptions naming the spectral bands, or a known sensor product",
        why="Spectral indices must know which band is which.",
    ),
    "nodata": DataRequirement(
        what="A scene with substantially complete coverage",
        why="Most of the supplied scene carries no measurable data.",
    ),
    "cloud": DataRequirement(
        what="A scene with less cloud, or a SAR acquisition of the same area",
        why="Cloud hides the surface and its shadows mimic real change.",
    ),
    "modality": DataRequirement(
        what="Imagery whose sensor type is identifiable from bands or tags",
        why="Tool selection depends on knowing whether the data is optical or radar.",
    ),
    "pair.crs_match": DataRequirement(
        what="Both images in the same projection",
        why="Pixel-wise comparison across different projections compares different ground.",
    ),
    "pair.resolution_match": DataRequirement(
        what="Two acquisitions of comparable resolution",
        why="Detail present in only one image cannot be compared.",
    ),
    "pair.overlap": DataRequirement(
        what="Two images covering the same ground",
        why="There must be common area to compare.",
    ),
    "pair.coregistration": DataRequirement(
        what="A co-registered pair with sub-pixel alignment",
        why="Beyond two pixels of offset, a change map shows edge artefacts rather than change.",
    ),
    "pair.modality_complement": DataRequirement(
        what="One optical or multispectral image and one SAR image",
        why="Cross-modal analysis draws on what each sensor sees that the other cannot.",
    ),
}


def _requirement_for(check: ReadinessCheck) -> DataRequirement | None:
    if check.id in _REQUIREMENT_BY_CHECK:
        return _REQUIREMENT_BY_CHECK[check.id]
    family = check.id.split(".")[0]
    return _REQUIREMENT_BY_CHECK.get(family)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_readiness(record: SessionRecord, store: SessionStore) -> ReadinessReport:
    """Run the gate over a session and return a full report."""
    started = time.perf_counter()
    report = ReadinessReport(
        session_id=record.session_id,
        configuration=record.configuration,
        verdict=ReadinessVerdict.READY,
    )

    if not record.images:
        report.verdict = ReadinessVerdict.REFUSED
        report.refusal_reasons.append("No imagery has been supplied yet.")
        report.requirements.append(
            DataRequirement(
                what="At least one raster in a supported format",
                why="The gate has nothing to inspect.",
            )
        )
        report.computed_ms = round((time.perf_counter() - started) * 1000, 1)
        return report

    ordered = [(image.role, image) for image in record.image_list()]
    paths: dict[ImageRole, Path] = {}
    for role, _image in ordered:
        path = store.find_image_path(record.session_id, role)
        if path is not None:
            paths[role] = path

    # Per-image checks.
    for role, image in ordered:
        meta = image.metadata
        report.checks.append(_check_crs(role, meta))
        report.checks.append(_check_resolution(role, meta))
        report.checks.append(_check_bands(role, meta))
        report.checks.append(_check_nodata(role, meta))
        if role in paths:
            report.checks.append(_check_cloud(role, paths[role], meta))
        report.checks.append(_check_modality(role, meta))
        report.checks.append(_check_tiling(role, meta))

    # Pair checks.
    is_pair = len(ordered) == 2
    if is_pair:
        roles = (ordered[0][0], ordered[1][0])
        metas = (ordered[0][1].metadata, ordered[1][1].metadata)

        report.checks.append(_check_crs_match(roles, metas))
        report.checks.append(_check_resolution_match(roles, metas))

        overlap_check, overlap_bounds, overlap_fraction = _check_overlap(roles, metas)
        report.checks.append(overlap_check)
        report.overlap_fraction = overlap_fraction
        if overlap_bounds is not None:
            report.overlap_bounds_native = list(overlap_bounds)
            report.common_epsg = metas[0].geo.epsg
            try:
                from rasterio.warp import transform_bounds

                report.overlap_bounds_wgs84 = list(
                    transform_bounds(
                        f"EPSG:{metas[0].geo.epsg}", "EPSG:4326", *overlap_bounds
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("overlap WGS84 transform failed: %s", exc)

        if roles[0] in paths and roles[1] in paths:
            report.checks.append(
                _check_coregistration(roles, (paths[roles[0]], paths[roles[1]]), metas)
            )

        if record.configuration is InputConfiguration.BI_TEMPORAL_PAIR:
            temporal, day_delta, month_delta, seasonal = _check_temporal(roles, metas)
            report.checks.append(temporal)
            report.day_delta = day_delta
            report.month_of_year_delta = month_delta
            report.seasonal_risk = seasonal
        else:
            report.checks.append(
                _na(
                    "pair.temporal",
                    "Temporal separation",
                    "Only relevant for a bi-temporal pair.",
                )
            )

        if record.configuration is InputConfiguration.CROSS_MODAL_PAIR:
            report.checks.append(_check_modality_complement(roles, metas))
        else:
            report.checks.append(
                _na(
                    "pair.modality_complement",
                    "Complementary modalities",
                    "Only relevant for a cross-modal pair.",
                )
            )
    else:
        for check_id, label in (
            ("pair.crs_match", "Matching projections"),
            ("pair.resolution_match", "Comparable resolution"),
            ("pair.overlap", "Spatial overlap"),
            ("pair.coregistration", "Co-registration"),
            ("pair.temporal", "Temporal separation"),
            ("pair.modality_complement", "Complementary modalities"),
        ):
            report.checks.append(
                _na(check_id, label, "Single-image input, so no pair check applies.")
            )

    # An incomplete configuration is a refusal in its own right.
    if record.configuration is InputConfiguration.INCOMPLETE:
        report.refusal_reasons.append(
            "The selected input configuration is not fully populated."
        )
        report.requirements.append(
            DataRequirement(
                what="Every slot of the chosen configuration filled",
                why="A paired analysis needs both images before it can run.",
            )
        )

    failures = report.by_status(CheckStatus.FAIL)
    for check in failures:
        report.refusal_reasons.append(f"{check.label}: {check.message}")
        requirement = _requirement_for(check)
        if requirement and requirement not in report.requirements:
            report.requirements.append(requirement)

    if report.refusal_reasons:
        report.verdict = ReadinessVerdict.REFUSED
    elif report.by_status(CheckStatus.WARN):
        report.verdict = ReadinessVerdict.READY_WITH_WARNINGS
    else:
        report.verdict = ReadinessVerdict.READY

    report.computed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "readiness %s -> %s (%d pass, %d warn, %d fail) in %.0f ms",
        record.session_id,
        report.verdict.value,
        len(report.by_status(CheckStatus.PASS)),
        len(report.by_status(CheckStatus.WARN)),
        len(failures),
        report.computed_ms,
    )
    return report
