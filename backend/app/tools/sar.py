"""The SAR backscatter engine.

Radar measures surface roughness and dielectric constant, not colour, and it does
so through cloud. That combination is why it belongs in this system: where the
optical instrument is blind, radar is not, and where they both see, their
agreement means something precisely because they measure different physics.

Water is the easiest target and the one that matters for flooding. A calm water
surface is specular: it reflects the radar pulse away from the sensor rather than
scattering it back, so open water is the darkest thing in a SAR scene by a wide
margin. Thresholding backscatter is therefore a well-founded water detector, not
an index dressed up as one.

Three things this module is careful about, because each is a standard way SAR
analysis goes wrong:

* **Speckle.** SAR intensity is multiplicative noise on top of the signal.
  Thresholding raw pixels produces a salt-and-pepper mask that no amount of
  post-processing recovers, so the image is filtered first and the filter is
  reported.
* **Units.** Radiometrically terrain-corrected products are linear power; many
  other products are already decibels. Taking the logarithm of a decibel value
  silently halves the dynamic range, so the conversion checks before it converts.
* **Radar shadow and layover.** Terrain can make a slope dark for geometric
  reasons that have nothing to do with water. This engine cannot separate those
  without a digital elevation model, and it says so rather than reporting them as
  water.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.raster_io import _to_decibels, read_role
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import mask_to_geojson, measurements_for_mask
from app.tools.indices import _crs_of, _transform_of, otsu_threshold

logger = logging.getLogger(__name__)

# Open water in C-band VV sits well below this almost everywhere. Used as a
# sanity bound on the automatic threshold and as the fallback when the histogram
# will not separate: Sentinel-1 studies of flood mapping converge on roughly
# -15 to -18 dB for VV over calm water.
CONVENTIONAL_WATER_DB = -16.0
WATER_REFERENCE = "Sentinel-1 VV flood mapping convention, roughly -15 to -18 dB"

# The automatic threshold is refused outside this band. Otsu on a scene that is
# almost entirely land puts the split in the wrong gap, and a threshold of -3 dB
# would classify a city as water.
PLAUSIBLE_WATER_DB = (-25.0, -10.0)

# Speckle filter window. A 5 by 5 median removes most speckle while keeping edges
# sharper than a mean filter of the same size would.
SPECKLE_WINDOW = 5

# Below this the automatic split is not separating two populations and the
# published threshold is used instead.
WEAK_SEPARABILITY = 0.35

# Radar shadow from terrain is also dark. Without a digital elevation model this
# engine cannot tell a shadowed slope from water, and regions this small are the
# ones most likely to be that rather than a real water body.
MIN_WATER_AREA_M2 = 4000.0


def speckle_filtered_db(
    channel: np.ma.MaskedArray, window: int = SPECKLE_WINDOW
) -> tuple[np.ma.MaskedArray, str]:
    """Convert to decibels and suppress speckle. Returns the array and the recipe.

    Filtering happens in the decibel domain deliberately. Speckle is
    multiplicative in power, which is additive in the logarithm, and a median in
    the log domain is therefore the better-behaved operation as well as the one
    that preserves the edges a water boundary depends on.
    """
    from scipy.ndimage import median_filter

    decibels = _to_decibels(channel)
    invalid = np.ma.getmaskarray(decibels)
    filled = np.ma.filled(decibels, np.nan)

    # Nodata is carried through as NaN so the filter cannot pull it into the
    # valid area; the mask is reapplied afterwards.
    with np.errstate(invalid="ignore"):
        smoothed = median_filter(
            np.nan_to_num(filled, nan=float(np.nanmedian(filled)) if np.isfinite(filled).any() else 0.0),
            size=window,
            mode="nearest",
        )

    result = np.ma.masked_array(smoothed, mask=invalid)
    return result, (
        f"converted to dB then median filtered with a {window} by {window} window "
        "in the logarithmic domain, where speckle is additive"
    )


def water_threshold_db(
    values: np.ndarray, method: str = "otsu"
) -> tuple[float, float, str]:
    """Choose the backscatter level below which a pixel is water.

    Returns the threshold, its separability, and how it was arrived at. An
    automatic threshold outside the physically plausible band is refused rather
    than used, because Otsu will always return something.
    """
    if method == "fixed":
        return CONVENTIONAL_WATER_DB, 0.0, (
            f"fixed at {CONVENTIONAL_WATER_DB:.1f} dB ({WATER_REFERENCE})"
        )

    try:
        threshold, separability = otsu_threshold(values)
    except ToolError as exc:
        return CONVENTIONAL_WATER_DB, 0.0, (
            f"the histogram would not separate ({exc}), so the published "
            f"{CONVENTIONAL_WATER_DB:.1f} dB was used instead"
        )

    low, high = PLAUSIBLE_WATER_DB
    if not (low <= threshold <= high):
        return CONVENTIONAL_WATER_DB, separability, (
            f"Otsu returned {threshold:.1f} dB, outside the plausible "
            f"{low:.0f} to {high:.0f} dB band for open water, so the published "
            f"{CONVENTIONAL_WATER_DB:.1f} dB was used instead"
        )

    if separability < WEAK_SEPARABILITY:
        return CONVENTIONAL_WATER_DB, separability, (
            f"Otsu returned {threshold:.1f} dB but separates the scene with only "
            f"{separability:.2f}, so the published {CONVENTIONAL_WATER_DB:.1f} dB "
            "was used instead"
        )

    return threshold, separability, (
        f"Otsu on the filtered dB histogram, {threshold:.1f} dB, separability "
        f"{separability:.2f}"
    )


class SarBackscatterEngine(BaseTool):
    """Detects open water from radar backscatter, through cloud."""

    name = "sar-backscatter-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Finds open water in radar backscatter by thresholding speckle-filtered "
        "VV in decibels. Sees through cloud, which is what makes it independent "
        "of the optical result rather than a second opinion on the same pixels."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            modalities=[Modality.SAR],
            any_of_band_roles=[[BandRole.VV], [BandRole.VH], [BandRole.HH]],
            requires_crs=True,
            configurations=[
                InputConfiguration.SINGLE,
                InputConfiguration.CROSS_MODAL_PAIR,
                InputConfiguration.BI_TEMPORAL_PAIR,
            ],
            description="A georeferenced radar acquisition carrying VV, VH, or HH.",
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "polarisation": {
                "type": "string",
                "enum": ["vv", "vh", "hh"],
                "default": "vv",
                "description": (
                    "Co-polarised VV gives the strongest water contrast in C-band."
                ),
            },
            "threshold_method": {
                "type": "string",
                "enum": ["otsu", "fixed"],
                "default": "otsu",
                "description": "How the water level is chosen.",
            },
            "speckle_window": {
                "type": "number",
                "default": SPECKLE_WINDOW,
                "minimum": 1,
                "maximum": 11,
                "description": "Median filter size in pixels.",
            },
            "min_water_area_m2": {
                "type": "number",
                "default": MIN_WATER_AREA_M2,
                "minimum": 0.0,
                "description": "Dark patches smaller than this are not reported.",
            },
        }

    def produces(self) -> list[str]:
        return [
            "sar_water_area_km2",
            "sar_water_pixels",
            "sar_water_fraction",
            "sar_water_clusters",
            "sar_water_threshold_db",
            "sar_water_separability",
            "sar_mean_backscatter_db",
        ]

    def execute(self, context: ToolContext) -> ToolOutcome:
        role = self._role_for(context)
        meta = context.metadata(role)

        wanted = str(context.param("polarisation", "vv")).lower()
        method = context.param("threshold_method", "otsu")
        window = int(context.param("speckle_window", SPECKLE_WINDOW))
        min_area = float(context.param("min_water_area_m2", MIN_WATER_AREA_M2))

        band, band_note = self._band_for(meta, wanted)
        outcome = self.outcome(
            parameters={
                "polarisation": band.value,
                "threshold_method": method,
                "speckle_window": window,
                "min_water_area_m2": min_area,
            }
        )
        if band_note:
            outcome.notes.append(band_note)

        raw = read_role(context.path(role), meta, band)
        if raw is None:
            raise ToolError(f"Band {band.value} could not be read from '{role.value}'.")

        decibels, recipe = speckle_filtered_db(raw, window)
        outcome.notes.append(recipe.capitalize() + ".")

        usable = decibels.compressed()
        if usable.size < 64:
            raise ToolError(
                "Too few valid radar pixels to establish a water level."
            )

        threshold, separability, threshold_note = water_threshold_db(usable, method)
        outcome.notes.append(threshold_note.capitalize() + ".")

        invalid = np.ma.getmaskarray(decibels)
        water = (np.ma.filled(decibels, 9999.0) < threshold) & ~invalid

        transform = _transform_of(meta)
        crs = _crs_of(meta)

        # Small dark patches are the ones most likely to be radar shadow rather
        # than water, and this engine has no terrain model to tell them apart.
        kept, dropped = _drop_small(water, transform, crs, min_area)

        layer = MaskLayer(
            key="sar_water",
            label="Water from radar backscatter",
            array=kept,
            transform=transform,
            crs=crs,
            description=(
                f"{band.value.upper()} backscatter below {threshold:.1f} dB after "
                f"a {window} by {window} median filter. Open water is specular, so "
                "it returns almost nothing to the sensor."
            ),
            applies_to=[role],
            threshold=threshold,
            threshold_method=method,
            separability=separability,
            invalid=invalid,
        )
        outcome.masks.append(layer)

        outcome.measurements.extend(
            measurements_for_mask(
                self.name,
                self.version,
                layer,
                label="Radar water",
                key_prefix="sar_water",
                min_component_area_m2=min_area,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="sar_water_threshold_db",
                label="Radar water threshold",
                value=threshold,
                unit="dB",
                formula=threshold_note,
                inputs={
                    "polarisation": band.value,
                    "conventional_threshold_db": CONVENTIONAL_WATER_DB,
                    "reference": WATER_REFERENCE,
                    "plausible_band_db": list(PLAUSIBLE_WATER_DB),
                    "speckle_window": window,
                },
                method=method,
                applies_to=[role],
                precision=1,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="sar_water_separability",
                label="Radar threshold separability",
                value=separability,
                unit="ratio",
                formula=(
                    "between-class variance at the threshold / total variance, "
                    "rescaled so one population scores 0"
                ),
                inputs={"weak_below": WEAK_SEPARABILITY},
                method="Otsu criterion normalised by total variance",
                applies_to=[role],
                precision=3,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="sar_mean_backscatter_db",
                label=f"Mean {band.value.upper()} backscatter",
                value=float(usable.mean()),
                unit="dB",
                formula=f"mean of filtered {band.value.upper()} over valid pixels",
                inputs={
                    "valid_pixels": int(usable.size),
                    "minimum_db": round(float(usable.min()), 2),
                    "maximum_db": round(float(usable.max()), 2),
                },
                applies_to=[role],
                precision=2,
            )
        )

        if dropped:
            outcome.notes.append(
                f"{dropped:,} dark pixel(s) in patches below {min_area:.0f} m2 were "
                "excluded. Radar shadow from terrain is also dark, and without a "
                "digital elevation model this engine cannot tell a shadowed slope "
                "from standing water, so the smallest patches are not claimed."
            )

        outcome.notes.append(
            "Radar shadow, layover, and very rough water roughened by wind are not "
            "tested for. The first two need a terrain model; the third makes water "
            "brighter and so causes an underestimate rather than a false positive."
        )

        outcome.artifacts["sar.decibels"] = decibels
        outcome.artifacts["sar.water"] = kept
        outcome.artifacts["sar.invalid"] = invalid
        outcome.artifacts["sar.threshold_db"] = threshold
        outcome.artifacts["sar.geojson"] = mask_to_geojson(
            kept, transform, crs, min_area_m2=min_area,
            properties={"source": "sar-backscatter-engine"},
        )
        return outcome

    def _role_for(self, context: ToolContext) -> ImageRole:
        sar = context.sar_roles()
        if not sar:
            raise ToolError("No radar image is present in this session.")
        for preferred in (ImageRole.SAR, ImageRole.SINGLE, ImageRole.DATE_B):
            if preferred in sar:
                return preferred
        return sar[0]

    def _band_for(self, meta, wanted: str) -> tuple[BandRole, str]:
        """The polarisation to use, and a note when it is not the one requested."""
        order = {
            "vv": (BandRole.VV, BandRole.HH, BandRole.VH),
            "vh": (BandRole.VH, BandRole.VV, BandRole.HH),
            "hh": (BandRole.HH, BandRole.VV, BandRole.VH),
        }.get(wanted, (BandRole.VV, BandRole.VH, BandRole.HH))

        for index, role in enumerate(order):
            if meta.band_index(role) is not None:
                if index == 0:
                    return role, ""
                return role, (
                    f"{wanted.upper()} is not present in this product, so "
                    f"{role.value.upper()} was used instead. Cross-polarised "
                    "channels give weaker water contrast than co-polarised ones."
                )

        available = ", ".join(r.value for r in meta.resolved_roles) or "none"
        raise ToolError(
            f"No usable polarisation found. This product carries: {available}."
        )


def _drop_small(
    mask: np.ndarray, transform, crs, min_area_m2: float
) -> tuple[np.ndarray, int]:
    """Keep only connected regions above the area floor."""
    from app.tools.gis import connected_components

    if not mask.any() or min_area_m2 <= 0:
        return mask, 0

    components = connected_components(mask, transform, crs, min_area_m2=min_area_m2)
    if not components:
        return np.zeros_like(mask), int(np.count_nonzero(mask))

    from skimage.measure import label as sk_label

    labelled = sk_label(mask.astype(bool), connectivity=2)
    kept = np.isin(labelled, [component.label for component in components])
    return kept, int(np.count_nonzero(mask) - np.count_nonzero(kept))


__all__ = [
    "CONVENTIONAL_WATER_DB",
    "PLAUSIBLE_WATER_DB",
    "SPECKLE_WINDOW",
    "SarBackscatterEngine",
    "speckle_filtered_db",
    "water_threshold_db",
]
