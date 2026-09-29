"""Spectral indices and thresholding.

Four normalised difference indices, each defined by the bands it needs. A band
that is absent means the index is skipped with a reason, never approximated from
a substitute band: NDBI computed without SWIR is not NDBI.

Thresholds are reported with a separability score. Otsu's method always returns a
number, including on a distribution with only one mode, so the score says how
much the threshold is worth. That figure feeds the uncertain band of the
disagreement map later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.raster_io import read_role_reflectance, reflectance_scale
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import measurements_for_mask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexDefinition:
    """A normalised difference index and what it is for."""

    key: str
    label: str
    positive_role: BandRole
    negative_role: BandRole
    highlights: str
    # The value conventionally used to separate the target class, kept as a
    # cross-check against the data-driven threshold.
    conventional_threshold: float
    reference: str

    @property
    def formula(self) -> str:
        return (
            f"{self.key} = ({self.positive_role.value} - {self.negative_role.value})"
            f" / ({self.positive_role.value} + {self.negative_role.value})"
        )

    def required_roles(self) -> tuple[BandRole, BandRole]:
        return (self.positive_role, self.negative_role)


INDEX_DEFINITIONS: dict[str, IndexDefinition] = {
    "NDVI": IndexDefinition(
        key="NDVI",
        label="Normalised Difference Vegetation Index",
        positive_role=BandRole.NIR,
        negative_role=BandRole.RED,
        highlights="live green vegetation",
        conventional_threshold=0.3,
        reference="Rouse et al. 1974",
    ),
    "NDWI": IndexDefinition(
        key="NDWI",
        label="Normalised Difference Water Index",
        positive_role=BandRole.GREEN,
        negative_role=BandRole.NIR,
        highlights="open surface water",
        conventional_threshold=0.0,
        reference="McFeeters 1996",
    ),
    "MNDWI": IndexDefinition(
        key="MNDWI",
        label="Modified Normalised Difference Water Index",
        positive_role=BandRole.GREEN,
        negative_role=BandRole.SWIR16,
        highlights="open water, with much less built-up confusion than NDWI",
        conventional_threshold=0.0,
        reference="Xu 2006",
    ),
    "NDBI": IndexDefinition(
        key="NDBI",
        label="Normalised Difference Built-up Index",
        positive_role=BandRole.SWIR16,
        negative_role=BandRole.NIR,
        highlights="built-up and impervious surfaces",
        conventional_threshold=0.0,
        reference="Zha et al. 2003",
    ),
    "NDMI": IndexDefinition(
        key="NDMI",
        label="Normalised Difference Moisture Index",
        positive_role=BandRole.NIR,
        negative_role=BandRole.SWIR16,
        highlights="vegetation and soil moisture",
        conventional_threshold=0.1,
        reference="Gao 1996",
    ),
}

# Indices whose target class is the low tail rather than the high tail.
INVERTED_INDICES: frozenset[str] = frozenset()

# Words a user might type, mapped to the index that answers them. Used by the
# grounding tool and the contract router.
#
# Water maps to MNDWI rather than NDWI on purpose. On the real Patna scene,
# NDWI marked scattered urban pixels as water: dark roofs and building shadows
# are bright in green relative to NIR. Substituting SWIR for NIR, which is what
# MNDWI does, pushes built-up further from the water threshold. NDWI is still
# computed, and the two make useful independent evidence precisely because they
# disagree over cities.
#
# Callers must fall back to NDWI when SWIR is absent, since MNDWI needs it.
TARGET_TO_INDEX: dict[str, str] = {
    "water": "MNDWI",
    "water body": "MNDWI",
    "waterbody": "MNDWI",
    "lake": "MNDWI",
    "reservoir": "MNDWI",
    "river": "MNDWI",
    "flood": "MNDWI",
    "flooded": "MNDWI",
    "vegetation": "NDVI",
    "crop": "NDVI",
    "cropland": "NDVI",
    "forest": "NDVI",
    "green": "NDVI",
    "built-up": "NDBI",
    "builtup": "NDBI",
    "built up": "NDBI",
    "urban": "NDBI",
    "settlement": "NDBI",
    "buildings": "NDBI",
    "impervious": "NDBI",
    "moisture": "NDMI",
    "wetness": "NDMI",
}

SCL_CLOUD_CLASSES: tuple[int, ...] = (3, 8, 9, 10)

# Otsu's between-class variance ratio for a single normal distribution split at
# its optimum is exactly 2/pi. That is the floor: any unimodal population scores
# this much purely from being cut in half, so separability is rescaled against it
# and a genuinely unimodal index reports 0.0 rather than a misleading 0.64.
UNIMODAL_OTSU_RATIO = 2.0 / np.pi

# A constant array does not report exactly zero variance: numpy's accumulation
# leaves a residue around 1e-33. Normalised difference indices live in [-1, 1],
# so a variance below this (standard deviation under a millionth of the range)
# means the index carries no structure and no threshold can separate it.
CONSTANT_VARIANCE_EPSILON = 1e-12

# Below this, the threshold is separating noise rather than two populations.
WEAK_SEPARABILITY = 0.35

# When Otsu and the published conventional threshold select areas differing by
# more than this fraction of the scene, the discrepancy is reported. On a scene
# with three spectral populations Otsu's two-class assumption breaks, and that is
# worth saying out loud rather than discovering downstream.
THRESHOLD_DIVERGENCE_NOTE = 0.05


@dataclass
class IndexResult:
    """A computed index plus the mask derived from it."""

    definition: IndexDefinition
    values: np.ma.MaskedArray
    threshold: float
    threshold_method: str
    separability: float
    mask: np.ndarray
    invalid: np.ndarray
    conventional_mask_fraction: float


def normalised_difference(
    positive: np.ma.MaskedArray, negative: np.ma.MaskedArray
) -> np.ma.MaskedArray:
    """(a - b) / (a + b), with a zero denominator masked rather than filled.

    A zero sum means both bands read zero, which is nodata, not an index of zero.
    """
    denominator = positive + negative
    safe = np.ma.masked_where(np.abs(denominator) < 1e-12, denominator)
    return (positive - negative) / safe


def otsu_threshold(values: np.ndarray) -> tuple[float, float]:
    """Otsu's threshold and how much the split is worth.

    Otsu returns a number for any input, including a single population that has
    simply been cut in half. Separability is what distinguishes the two cases: it
    is the between-class variance ratio rescaled so that a unimodal Gaussian
    scores 0 and two perfectly separated populations score 1.

    Without the rescaling the raw ratio for a Gaussian is 2/pi, about 0.64, which
    reads as a confident split when it is nothing of the kind.
    """
    from skimage.filters import threshold_otsu

    finite = values[np.isfinite(values)]
    if finite.size < 32:
        raise ToolError("Too few valid pixels to derive a threshold.")

    total_variance = float(finite.var())
    if total_variance < CONSTANT_VARIANCE_EPSILON:
        raise ToolError(
            "The index is effectively constant "
            f"(variance {total_variance:.2e}), so no threshold separates it."
        )

    # A capped histogram keeps this fast and stable on large scenes.
    threshold = float(threshold_otsu(finite, nbins=256))

    below = finite[finite <= threshold]
    above = finite[finite > threshold]
    if below.size == 0 or above.size == 0:
        return threshold, 0.0

    weight_below = below.size / finite.size
    weight_above = above.size / finite.size
    between = weight_below * weight_above * (below.mean() - above.mean()) ** 2
    ratio = between / total_variance

    rescaled = (ratio - UNIMODAL_OTSU_RATIO) / (1.0 - UNIMODAL_OTSU_RATIO)
    return threshold, float(min(1.0, max(0.0, rescaled)))


def cloud_invalid_mask(
    context: ToolContext, role: ImageRole, shape: tuple[int, int], **read_kwargs: Any
) -> np.ndarray:
    """Pixels that cannot be classified: nodata handled elsewhere, cloud here.

    ``read_kwargs`` are passed through to the band read, so a caller working on a
    resampled common grid can obtain the cloud mask on that same grid. Nearest
    neighbour is forced because the classification layer is categorical and
    interpolating class codes would invent classes that do not exist.
    """
    meta = context.metadata(role)
    if meta.band_index(BandRole.SCL) is None:
        return np.zeros(shape, dtype=bool)

    from rasterio.enums import Resampling

    from app.core.raster_io import read_role

    if "out_shape" in read_kwargs:
        read_kwargs.setdefault("resampling", Resampling.nearest)

    scl = read_role(context.path(role), meta, BandRole.SCL, **read_kwargs)
    if scl is None or scl.shape != shape:
        return np.zeros(shape, dtype=bool)
    return np.isin(np.ma.filled(scl, 0).astype("int16"), SCL_CLOUD_CLASSES)


def compute_index(
    context: ToolContext,
    role: ImageRole,
    definition: IndexDefinition,
    *,
    threshold_method: str = "otsu",
    fixed_threshold: float | None = None,
    exclude_cloud: bool = True,
) -> IndexResult:
    """Compute one index on one image and threshold it."""
    meta = context.metadata(role)
    missing = meta.missing_roles(*definition.required_roles())
    if missing:
        names = ", ".join(role_.value for role_ in missing)
        raise ToolError(
            f"{definition.key} needs {names}, which this image does not provide"
        )

    path = context.path(role)
    positive = read_role_reflectance(path, meta, definition.positive_role)
    negative = read_role_reflectance(path, meta, definition.negative_role)
    if positive is None or negative is None:
        raise ToolError(f"{definition.key} bands could not be read.")
    if positive.shape != negative.shape:
        raise ToolError(
            f"{definition.key} bands have different shapes "
            f"({positive.shape} and {negative.shape})."
        )

    values = normalised_difference(positive, negative)

    invalid = np.ma.getmaskarray(values).copy()
    if exclude_cloud:
        cloud = cloud_invalid_mask(context, role, values.shape)
        invalid |= cloud

    usable = np.ma.masked_array(values, mask=invalid).compressed()

    if threshold_method == "fixed":
        if fixed_threshold is None:
            fixed_threshold = definition.conventional_threshold
        threshold = float(fixed_threshold)
        separability = 0.0
        try:
            _, separability = otsu_threshold(usable)
        except ToolError:
            separability = 0.0
        method = f"fixed at {threshold:.4f} ({definition.reference})"
    else:
        threshold, separability = otsu_threshold(usable)
        method = "Otsu (maximised between-class variance)"

    filled = np.ma.filled(values, -np.inf)
    mask = (filled > threshold) & ~invalid

    conventional = (filled > definition.conventional_threshold) & ~invalid
    valid_count = int(np.count_nonzero(~invalid))
    conventional_fraction = (
        int(np.count_nonzero(conventional)) / valid_count if valid_count else 0.0
    )

    return IndexResult(
        definition=definition,
        values=np.ma.masked_array(values, mask=invalid),
        threshold=threshold,
        threshold_method=method,
        separability=separability,
        mask=mask,
        invalid=invalid,
        conventional_mask_fraction=conventional_fraction,
    )


class SpectralIndexEngine(BaseTool):
    """Computes normalised difference indices and thresholds them into masks."""

    name = "spectral-index-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "NDVI, NDWI, NDBI, and NDMI from reflectance, each thresholded with Otsu "
        "and reported with a separability score and its area."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            modalities=[Modality.OPTICAL],
            any_of_band_roles=[
                [BandRole.NIR, BandRole.RED],
                [BandRole.GREEN, BandRole.NIR],
                [BandRole.SWIR16, BandRole.NIR],
                [BandRole.GREEN, BandRole.SWIR16],
            ],
            configurations=[
                InputConfiguration.SINGLE,
                InputConfiguration.CROSS_MODAL_PAIR,
                InputConfiguration.BI_TEMPORAL_PAIR,
            ],
            description=(
                "An optical or multispectral image with at least one index's "
                "band pair."
            ),
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "indices": {
                "type": "array",
                "items": {"enum": sorted(INDEX_DEFINITIONS)},
                "default": sorted(INDEX_DEFINITIONS),
                "description": "Which indices to compute.",
            },
            "threshold_method": {
                "type": "string",
                "enum": ["otsu", "fixed"],
                "default": "otsu",
                "description": "How the separating value is chosen.",
            },
            "fixed_threshold": {
                "type": "number",
                "default": None,
                "description": "Used when threshold_method is 'fixed'.",
            },
            "exclude_cloud": {
                "type": "boolean",
                "default": True,
                "description": "Drop SCL cloud and shadow pixels before thresholding.",
            },
        }

    def produces(self) -> list[str]:
        keys: list[str] = []
        for name in INDEX_DEFINITIONS:
            lower = name.lower()
            keys.extend(
                [
                    f"{lower}_mean",
                    f"{lower}_threshold",
                    f"{lower}_separability",
                    f"{lower}_area_km2",
                    f"{lower}_fraction",
                    f"{lower}_clusters",
                ]
            )
        return keys

    def execute(self, context: ToolContext) -> ToolOutcome:
        requested = context.param("indices") or sorted(INDEX_DEFINITIONS)
        unknown = [name for name in requested if name not in INDEX_DEFINITIONS]
        if unknown:
            raise ToolError(
                f"Unknown index/indices {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(INDEX_DEFINITIONS))}."
            )

        threshold_method = context.param("threshold_method", "otsu")
        fixed_threshold = context.param("fixed_threshold")
        exclude_cloud = bool(context.param("exclude_cloud", True))

        outcome = self.outcome(
            parameters={
                "indices": list(requested),
                "threshold_method": threshold_method,
                "fixed_threshold": fixed_threshold,
                "exclude_cloud": exclude_cloud,
            }
        )

        roles = context.optical_roles() or [context.primary_role()]
        computed_any = False

        for role in roles:
            meta = context.metadata(role)
            divisor, scale_note = reflectance_scale(meta)

            for name in requested:
                definition = INDEX_DEFINITIONS[name]
                try:
                    result = compute_index(
                        context,
                        role,
                        definition,
                        threshold_method=threshold_method,
                        fixed_threshold=fixed_threshold,
                        exclude_cloud=exclude_cloud,
                    )
                except ToolError as exc:
                    # Availability gating: say what is missing rather than
                    # quietly omitting the index.
                    outcome.notes.append(f"{role.value}: {exc}")
                    continue

                computed_any = True
                suffix = f"{name.lower()}.{role.value}"
                layer = MaskLayer(
                    key=f"{name.lower()}_mask.{role.value}",
                    label=f"{name} above threshold",
                    array=result.mask,
                    transform=_transform_of(meta),
                    crs=_crs_of(meta),
                    description=(
                        f"{definition.formula}; highlights {definition.highlights}. "
                        f"Threshold {result.threshold:.4f} via {result.threshold_method}."
                    ),
                    applies_to=[role],
                    threshold=result.threshold,
                    threshold_method=result.threshold_method,
                    separability=result.separability,
                    invalid=result.invalid,
                )
                outcome.masks.append(layer)

                usable = result.values.compressed()
                outcome.measurements.extend(
                    [
                        self.measurement(
                            key=f"{suffix}_mean",
                            label=f"Mean {name}",
                            value=float(usable.mean()) if usable.size else 0.0,
                            unit="index",
                            formula=f"mean of {definition.formula} over observable pixels",
                            inputs={
                                "observable_pixels": int(usable.size),
                                "reflectance_scale": divisor,
                                "scale_basis": scale_note,
                                "cloud_excluded": exclude_cloud,
                            },
                            method=f"{definition.reference}",
                            applies_to=[role],
                            precision=4,
                        ),
                        self.measurement(
                            key=f"{suffix}_threshold",
                            label=f"{name} threshold",
                            value=result.threshold,
                            unit="index",
                            formula=result.threshold_method,
                            inputs={
                                "conventional_threshold": definition.conventional_threshold,
                                "conventional_mask_fraction": round(
                                    result.conventional_mask_fraction, 6
                                ),
                                "observable_pixels": int(usable.size),
                            },
                            method=result.threshold_method,
                            applies_to=[role],
                            precision=4,
                        ),
                        self.measurement(
                            key=f"{suffix}_separability",
                            label=f"{name} threshold separability",
                            value=result.separability,
                            unit="ratio",
                            formula=(
                                "between-class variance at the threshold / "
                                "total variance"
                            ),
                            inputs={
                                "threshold": result.threshold,
                                "weak_below": WEAK_SEPARABILITY,
                            },
                            method="Otsu criterion normalised by total variance",
                            applies_to=[role],
                            precision=3,
                        ),
                    ]
                )

                outcome.measurements.extend(
                    measurements_for_mask(
                        self.name,
                        self.version,
                        layer,
                        label=f"{name} positive",
                        key_prefix=suffix,
                        min_component_area_m2=context.param(
                            "min_component_area_m2", 2000.0
                        ),
                    )
                )

                if result.separability < WEAK_SEPARABILITY:
                    outcome.notes.append(
                        f"{role.value} {name}: threshold separability "
                        f"{result.separability:.2f} is below {WEAK_SEPARABILITY}, so "
                        "the split is weakly supported and the mask edge is uncertain."
                    )

                # Otsu assumes two populations. A scene holding water, vegetation
                # and built-up has three, and the split can land in the wrong gap,
                # which shows up as a large gap against the published threshold.
                valid_count = int(np.count_nonzero(~result.invalid))
                otsu_fraction = (
                    int(np.count_nonzero(result.mask)) / valid_count
                    if valid_count
                    else 0.0
                )
                divergence = abs(otsu_fraction - result.conventional_mask_fraction)
                if (
                    threshold_method == "otsu"
                    and divergence > THRESHOLD_DIVERGENCE_NOTE
                ):
                    outcome.notes.append(
                        f"{role.value} {name}: the Otsu threshold "
                        f"{result.threshold:.3f} selects "
                        f"{otsu_fraction:.1%} of the observable area while the "
                        f"conventional threshold "
                        f"{definition.conventional_threshold:.2f} "
                        f"({definition.reference}) selects "
                        f"{result.conventional_mask_fraction:.1%}. Otsu assumes two "
                        "populations, so a scene with more than two land-cover "
                        "classes can place the split in the wrong gap."
                    )

                outcome.artifacts[f"index.{suffix}"] = result

        if not computed_any:
            raise ToolError(
                "No index could be computed: "
                + ("; ".join(outcome.notes) if outcome.notes else "no usable bands")
            )

        return outcome


def _transform_of(meta) -> Any:
    from rasterio.transform import Affine

    coefficients = meta.geo.transform
    if len(coefficients) != 6:
        raise ToolError("The image has no usable affine transform.")
    return Affine(*coefficients)


def _crs_of(meta) -> Any:
    if meta.geo.epsg is not None:
        from rasterio.crs import CRS

        return CRS.from_epsg(meta.geo.epsg)
    if meta.geo.crs_wkt:
        from rasterio.crs import CRS

        return CRS.from_wkt(meta.geo.crs_wkt)
    return None


__all__ = [
    "INDEX_DEFINITIONS",
    "IndexDefinition",
    "IndexResult",
    "SpectralIndexEngine",
    "TARGET_TO_INDEX",
    "WEAK_SEPARABILITY",
    "compute_index",
    "normalised_difference",
    "otsu_threshold",
]
