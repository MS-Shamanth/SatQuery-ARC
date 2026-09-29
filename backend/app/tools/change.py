"""The change engine.

Given two acquisitions of the same place, this measures what moved. Two methods
run together because they fail differently, and the point of the system is to
notice when they disagree:

* **Index delta.** The target's index is computed on each date and subtracted.
  Directional and interpretable: a pixel that was not built-up and now is shows
  up as a positive NDBI step. It inherits the index's blind spots.
* **Change vector analysis.** The per-band spectral change vector's magnitude,
  after Wiemker's formulation: change is a distance in reflectance space, so it
  catches change the target's own index misses, at the cost of not saying what
  kind of change it is.

Three rules are non-negotiable here, because getting them wrong is the classic
way bi-temporal change detection lies:

1. **Both dates are read onto one grid.** Comparing arrays of different shapes,
   or with different pixel footprints, measures resampling rather than change.
2. **A pixel unobservable on either date is unobservable, full stop.** Cloud on
   one date and clear on the other is not change. Those pixels are excluded from
   every numerator and every denominator, and the excluded fraction is reported.
3. **Change is reported as gain, loss, and net, never as net alone.** A scene
   that gained 2 km2 and lost 2 km2 has a net of zero, and calling that "no
   change" would be false.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.core.raster_io import intersect_bounds, read_role_reflectance
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import mask_area_km2, measurements_for_mask, pixel_area_m2
from app.tools.indices import (
    INDEX_DEFINITIONS,
    TARGET_TO_INDEX,
    WEAK_SEPARABILITY,
    IndexDefinition,
    cloud_invalid_mask,
    normalised_difference,
    otsu_threshold,
)

logger = logging.getLogger(__name__)

# A step in a normalised index below this is within the noise of atmospheric
# correction and sensor calibration between two acquisitions, so it is not
# reported as change. Chosen to sit above typical residual BRDF and aerosol
# differences in Sentinel-2 L2A while staying well below a real class transition,
# which moves an index by 0.3 or more.
DEFAULT_DELTA_THRESHOLD = 0.15

# Change vector magnitude is thresholded at mean + k standard deviations over the
# observable pixels. Two is the conventional choice and is reported alongside the
# resulting value so the reader can see what it selected.
DEFAULT_CVA_SIGMA = 2.0

# Bands used for the change vector when they are present. Visible plus NIR plus
# SWIR spans the reflectance behaviour that distinguishes the land-cover classes
# this system reasons about.
CVA_BANDS: tuple[BandRole, ...] = (
    BandRole.BLUE,
    BandRole.GREEN,
    BandRole.RED,
    BandRole.NIR,
    BandRole.SWIR16,
)

# Three is the floor for calling something a change *vector*. Across two bands the
# magnitude is little more than a band ratio and adds nothing the index difference
# has not already said, so below this the method is declined rather than run and
# presented as independent evidence.
MIN_CVA_BANDS = 3

# Below this observable fraction the pair is not worth measuring: whatever is
# reported would describe a small unrepresentative corner of the scene.
MIN_OBSERVABLE_FRACTION = 0.25

# A second index whose expected direction is known for a given target, used to
# corroborate the primary measurement independently.
#
# New construction replaces vegetation or soil with impervious surface, so NDBI
# rising should coincide with NDVI falling. Water expanding likewise removes
# vegetation. When both say the same thing the finding is stronger than either
# alone; when they disagree that disagreement is the interesting result, which is
# why this is emitted as separate evidence rather than folded into the primary
# number.
CORROBORATING_INDEX: dict[str, tuple[str, int]] = {
    "NDBI": ("NDVI", -1),
    "MNDWI": ("NDVI", -1),
    "NDWI": ("NDVI", -1),
}

# The corroborating index only has to move half as far as the primary, because it
# is a secondary consequence of the transition rather than the transition itself.
CORROBORATION_FACTOR = 0.5


@dataclass(frozen=True)
class CommonGrid:
    """The single grid both dates are read onto."""

    bounds: tuple[float, float, float, float]
    transform: Affine
    crs: CRS | None
    width: int
    height: int
    pixel_m: float
    note: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def read_kwargs(self) -> dict[str, Any]:
        return {"bounds": self.bounds, "out_shape": self.shape}


def build_common_grid(
    context: ToolContext, role_a: ImageRole, role_b: ImageRole
) -> CommonGrid:
    """Resolve the overlapping footprint and a shared pixel grid for two images.

    The finer of the two resolutions is kept, so the comparison is not silently
    degraded to the coarser sensor. Both dates are then resampled onto it, which
    means any resampling error is applied to both and cannot masquerade as change.
    """
    meta_a, meta_b = context.metadata(role_a), context.metadata(role_b)

    if meta_a.geo.epsg is None or meta_b.geo.epsg is None:
        raise ToolError(
            "Change measurement needs both dates georeferenced, and at least one "
            "carries no coordinate reference system."
        )
    if meta_a.geo.epsg != meta_b.geo.epsg:
        raise ToolError(
            f"The two dates are in different projections (EPSG:{meta_a.geo.epsg} "
            f"and EPSG:{meta_b.geo.epsg}). Reproject them to a common grid first."
        )

    overlap = intersect_bounds(meta_a.geo.bounds_native, meta_b.geo.bounds_native)
    if overlap is None:
        raise ToolError("The two dates do not overlap, so there is nothing to compare.")

    sizes = [
        size
        for size in (meta_a.geo.gsd_x_m, meta_b.geo.gsd_x_m)
        if size is not None and size > 0
    ]
    if not sizes:
        raise ToolError("Neither date has a resolvable pixel size.")
    pixel = min(sizes)

    left, bottom, right, top = overlap
    width = max(1, int(round((right - left) / pixel)))
    height = max(1, int(round((top - bottom) / pixel)))
    transform = Affine(pixel, 0.0, left, 0.0, -pixel, top)

    coarse = max(sizes)
    note = (
        f"both dates resampled onto a {pixel:.2f} m grid over the "
        f"{(right - left) / 1000:.2f} x {(top - bottom) / 1000:.2f} km overlap"
    )
    if abs(coarse - pixel) > 1e-6:
        note += (
            f"; the coarser input is {coarse:.2f} m, so its pixels are upsampled "
            "rather than the finer input being thrown away"
        )

    return CommonGrid(
        bounds=overlap,
        transform=transform,
        crs=CRS.from_epsg(meta_a.geo.epsg),
        width=width,
        height=height,
        pixel_m=pixel,
        note=note,
    )


def _unobservable(
    context: ToolContext, grid: CommonGrid, roles: tuple[ImageRole, ImageRole],
    arrays: list[np.ma.MaskedArray], exclude_cloud: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pixels that cannot be compared, and why.

    Union, not intersection: a pixel has to be observable on *both* dates for a
    difference between them to mean anything.
    """
    invalid = np.zeros(grid.shape, dtype=bool)
    for array in arrays:
        invalid |= np.ma.getmaskarray(array)
    nodata_fraction = float(invalid.mean())

    cloud_fraction = 0.0
    if exclude_cloud:
        cloud = np.zeros(grid.shape, dtype=bool)
        for role in roles:
            cloud |= cloud_invalid_mask(
                context, role, grid.shape, **grid.read_kwargs()
            )
        cloud_fraction = float(cloud.mean())
        invalid |= cloud

    return invalid, {
        "nodata_or_offgrid_fraction": round(nodata_fraction, 6),
        "cloud_fraction_either_date": round(cloud_fraction, 6),
        "unobservable_fraction": round(float(invalid.mean()), 6),
    }


def _index_on_grid(
    context: ToolContext,
    role: ImageRole,
    definition: IndexDefinition,
    grid: CommonGrid,
) -> np.ma.MaskedArray:
    meta = context.metadata(role)
    missing = meta.missing_roles(*definition.required_roles())
    if missing:
        names = ", ".join(item.value for item in missing)
        raise ToolError(
            f"{definition.key} needs {names}, which '{role.value}' does not provide"
        )
    path = context.path(role)
    kwargs = grid.read_kwargs()
    positive = read_role_reflectance(path, meta, definition.positive_role, **kwargs)
    negative = read_role_reflectance(path, meta, definition.negative_role, **kwargs)
    if positive is None or negative is None:
        raise ToolError(f"{definition.key} bands could not be read for '{role.value}'.")
    return normalised_difference(positive, negative)


# How far the threshold is nudged when testing how much the answer depends on it.
# A twentieth of the index range is larger than the disagreement between published
# thresholds for the same index, so surviving it is a meaningful result.
THRESHOLD_PROBE = 0.05


def _threshold_sensitivity(
    values_a: np.ma.MaskedArray,
    values_b: np.ma.MaskedArray,
    invalid: np.ndarray,
    threshold: float,
    transform: Affine,
) -> tuple[float, dict[str, Any]]:
    """How much the measured change moves when the threshold is nudged.

    The measurement that matters for trust. If shifting the boundary by a
    twentieth of the index range barely moves the answer, the answer does not
    depend on where the boundary was put; if it moves the answer a lot, it does,
    and the figure should be read with that in mind.

    Returned as a fraction of the starting area so it is comparable across scenes.
    """
    areas: list[tuple[float, float, float]] = []
    for offset in (-THRESHOLD_PROBE, 0.0, THRESHOLD_PROBE):
        level = threshold + offset
        before = (np.ma.filled(values_a, -9.0) > level) & ~invalid
        after = (np.ma.filled(values_b, -9.0) > level) & ~invalid
        _, before_km2 = mask_area_km2(before, transform)
        _, after_km2 = mask_area_km2(after, transform)
        areas.append((level, before_km2 or 0.0, after_km2 or 0.0))

    changes = [after - before for _, before, after in areas]
    baseline = areas[1][1]
    spread = max(changes) - min(changes)
    sensitivity = spread / baseline if baseline > 1e-9 else 1.0

    return min(1.0, sensitivity), {
        "probe": THRESHOLD_PROBE,
        "change_at_low_threshold_km2": round(changes[0], 6),
        "change_at_threshold_km2": round(changes[1], 6),
        "change_at_high_threshold_km2": round(changes[2], 6),
        "baseline_area_km2": round(baseline, 6),
    }


def _separability_at(values: np.ndarray, threshold: float) -> float:
    """How well a given threshold splits a distribution into two populations.

    Same normalisation as the Otsu criterion used elsewhere, so a score from a
    fixed published threshold is comparable with one Otsu chose: a single
    population scores 0 and two well-separated ones approach 1.
    """
    from app.tools.indices import UNIMODAL_OTSU_RATIO

    finite = values[np.isfinite(values)]
    if finite.size < 32:
        return 0.0
    total = float(finite.var())
    if total < 1e-12:
        return 0.0

    below = finite[finite <= threshold]
    above = finite[finite > threshold]
    if below.size == 0 or above.size == 0:
        return 0.0

    weight = (below.size / finite.size) * (above.size / finite.size)
    between = weight * (below.mean() - above.mean()) ** 2
    rescaled = (between / total - UNIMODAL_OTSU_RATIO) / (1.0 - UNIMODAL_OTSU_RATIO)
    return float(min(1.0, max(0.0, rescaled)))


def resolve_change_index(targets: list[str], context: ToolContext) -> tuple[str, str]:
    """Which index measures the target's change, and why that one.

    Falls back to NDVI, which is the most broadly meaningful single index, when
    the request names no class this vocabulary recognises.
    """
    roles = context.temporal_roles()
    meta = context.metadata(roles[0]) if roles else None

    for target in targets:
        name = TARGET_TO_INDEX.get(target.strip().lower())
        if name is None:
            continue
        definition = INDEX_DEFINITIONS[name]
        if meta is None or not meta.missing_roles(*definition.required_roles()):
            return name, f"'{target}' is measured by {name} ({definition.reference})"
        # MNDWI needs SWIR; NDWI is the documented substitute.
        if name == "MNDWI" and not meta.missing_roles(
            *INDEX_DEFINITIONS["NDWI"].required_roles()
        ):
            return "NDWI", (
                f"'{target}' is normally measured by MNDWI, but this sensor carries "
                "no SWIR, so NDWI was used instead"
            )

    return "NDVI", (
        "no recognised target class, so NDVI is used as the general-purpose "
        "measure of surface change (Rouse 1974)"
    )


class ChangeCvaEngine(BaseTool):
    """Measures change between two dates by index delta and by change vector."""

    name = "change-cva-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Bi-temporal change: the target index differenced between dates for "
        "direction, and the multi-band change vector magnitude for extent. "
        "Reports gain, loss, and net separately, and excludes anything either "
        "date could not see."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            modalities=[Modality.OPTICAL],
            any_of_band_roles=[
                [BandRole.NIR, BandRole.RED],
                [BandRole.GREEN, BandRole.NIR],
                [BandRole.SWIR16, BandRole.NIR],
            ],
            requires_crs=True,
            configurations=[InputConfiguration.BI_TEMPORAL_PAIR],
            description="Two georeferenced optical acquisitions of the same area.",
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "index": {
                "type": "string",
                "enum": sorted(INDEX_DEFINITIONS),
                "default": None,
                "description": "Index to difference. Derived from the target if unset.",
            },
            "delta_threshold": {
                "type": "number",
                "default": DEFAULT_DELTA_THRESHOLD,
                "minimum": 0.0,
                "maximum": 2.0,
                "description": "Index step treated as real change rather than noise.",
            },
            "cva_sigma": {
                "type": "number",
                "default": DEFAULT_CVA_SIGMA,
                "minimum": 0.5,
                "maximum": 6.0,
                "description": "Change vector threshold, in standard deviations.",
            },
            "class_threshold_method": {
                "type": "string",
                "enum": ["fixed", "otsu"],
                "default": "fixed",
                "description": "How each date's class mask is separated.",
            },
            "exclude_cloud": {
                "type": "boolean",
                "default": True,
                "description": "Drop pixels either date saw through cloud.",
            },
            "min_component_area_m2": {
                "type": "number",
                "default": 4000.0,
                "minimum": 0.0,
                "description": "Change patches smaller than this are not counted.",
            },
        }

    def produces(self) -> list[str]:
        return [
            "area_km2_before",
            "area_km2_after",
            "area_change_km2",
            "percentage_change",
            "gain_km2",
            "loss_km2",
            "net_change_km2",
            "persistent_km2",
            "changed_fraction",
            "class_separability",
            "threshold_sensitivity",
            "unconfirmed_crossings_km2",
            "corroborated_gain_km2",
            "cva_threshold",
            "cva_mean_magnitude",
            "observable_fraction",
            "delta_mean",
        ]

    def execute(self, context: ToolContext) -> ToolOutcome:
        roles = context.temporal_roles()
        if roles is None:
            raise ToolError(
                "Change measurement needs a dated pair, and this session does not "
                "hold one."
            )
        role_a, role_b = roles

        index_name = context.param("index")
        if index_name:
            provenance = f"{index_name} requested explicitly"
        else:
            index_name, provenance = resolve_change_index(
                context.target_classes, context
            )
        if index_name not in INDEX_DEFINITIONS:
            raise ToolError(
                f"'{index_name}' is not a known index. "
                f"Available: {', '.join(sorted(INDEX_DEFINITIONS))}."
            )
        definition = INDEX_DEFINITIONS[index_name]

        delta_threshold = float(
            context.param("delta_threshold", DEFAULT_DELTA_THRESHOLD)
        )
        cva_sigma = float(context.param("cva_sigma", DEFAULT_CVA_SIGMA))
        class_method = context.param("class_threshold_method", "fixed")
        exclude_cloud = bool(context.param("exclude_cloud", True))
        min_area = float(context.param("min_component_area_m2", 4000.0))

        outcome = self.outcome(
            parameters={
                "index": index_name,
                "delta_threshold": delta_threshold,
                "cva_sigma": cva_sigma,
                "class_threshold_method": class_method,
                "exclude_cloud": exclude_cloud,
                "min_component_area_m2": min_area,
            }
        )

        grid = build_common_grid(context, role_a, role_b)
        outcome.notes.append(grid.note.capitalize() + ".")

        values_a = _index_on_grid(context, role_a, definition, grid)
        values_b = _index_on_grid(context, role_b, definition, grid)
        invalid, observability = _unobservable(
            context, grid, (role_a, role_b), [values_a, values_b], exclude_cloud
        )

        observable = int(np.count_nonzero(~invalid))
        total = int(invalid.size)
        observable_fraction = observable / total if total else 0.0
        if observable_fraction < MIN_OBSERVABLE_FRACTION:
            raise ToolError(
                f"Only {observable_fraction:.1%} of the overlap is observable on both "
                f"dates, below the {MIN_OBSERVABLE_FRACTION:.0%} needed for a change "
                "measurement that describes the scene rather than a corner of it."
            )

        target_label = (
            context.target_classes[0] if context.target_classes else definition.highlights
        )
        outcome.notes.append(provenance.capitalize() + ".")

        # -- the class on each date ---------------------------------------
        threshold, class_separability, threshold_note = self._class_threshold(
            values_a, values_b, invalid, definition, class_method
        )
        outcome.notes.append(threshold_note)
        before = (np.ma.filled(values_a, -9.0) > threshold) & ~invalid
        after = (np.ma.filled(values_b, -9.0) > threshold) & ~invalid

        before_px, area_before = mask_area_km2(before, grid.transform)
        after_px, area_after = mask_area_km2(after, grid.transform)

        # How much the answer would move if the threshold did. This is the
        # question separability was only ever a proxy for, and the proxy is
        # misleading in both directions: an index can fail to split a scene
        # because it cannot distinguish the class, which makes the areas
        # meaningless, or because the scene is almost entirely one class, which
        # makes them unusually robust. Perturbing the threshold tells those apart.
        sensitivity, sensitivity_inputs = _threshold_sensitivity(
            values_a, values_b, invalid, threshold, grid.transform
        )

        # -- index delta ---------------------------------------------------
        delta = np.ma.filled(values_b - values_a, 0.0)
        delta[invalid] = 0.0
        delta_increase = (delta > delta_threshold) & ~invalid
        delta_decrease = (delta < -delta_threshold) & ~invalid

        usable_delta = delta[~invalid]
        delta_mean = float(usable_delta.mean()) if usable_delta.size else 0.0

        # -- gain, loss, persistence --------------------------------------
        #
        # A transition has to satisfy two independent conditions: the pixel must
        # end up on the other side of the class boundary, and the index must have
        # actually moved by more than the inter-date noise. Without the second
        # condition a pixel sitting on the threshold that drifts from 0.0039 to
        # 0.0041 is counted as newly built-up, and on a scene where the index does
        # not separate cleanly those crossings dominate the answer. They are
        # counted separately rather than discarded, because how many there are is
        # itself a measure of how trustworthy the boundary is.
        crossed_up = after & ~before
        crossed_down = before & ~after
        gain = crossed_up & delta_increase
        loss = crossed_down & delta_decrease
        unconfirmed = (crossed_up & ~delta_increase) | (crossed_down & ~delta_decrease)
        persistent = before & after

        _, gain_km2 = mask_area_km2(gain, grid.transform)
        _, loss_km2 = mask_area_km2(loss, grid.transform)
        _, persistent_km2 = mask_area_km2(persistent, grid.transform)
        _, unconfirmed_km2 = mask_area_km2(unconfirmed, grid.transform)

        # -- change vector -------------------------------------------------
        magnitude, cva_bands, cva_threshold, cva_note = self._change_vector(
            context, grid, role_a, role_b, invalid, cva_sigma
        )

        pixel_area = pixel_area_m2(grid.transform)
        applies = [role_a, role_b]

        layers = [
            MaskLayer(
                key="change_gain",
                label=f"{target_label} gained",
                array=gain,
                transform=grid.transform,
                crs=grid.crs,
                description=(
                    f"{index_name} above {threshold:.3f} on {role_b.value} but not on "
                    f"{role_a.value}. {threshold_note}"
                ),
                applies_to=applies,
                threshold=threshold,
                threshold_method=class_method,
                invalid=invalid,
            ),
            MaskLayer(
                key="change_loss",
                label=f"{target_label} lost",
                array=loss,
                transform=grid.transform,
                crs=grid.crs,
                description=(
                    f"{index_name} above {threshold:.3f} on {role_a.value} but not on "
                    f"{role_b.value}."
                ),
                applies_to=applies,
                threshold=threshold,
                threshold_method=class_method,
                invalid=invalid,
            ),
            MaskLayer(
                key=f"delta_{index_name.lower()}_increase",
                label=f"{index_name} rose by more than {delta_threshold:.2f}",
                array=delta_increase,
                transform=grid.transform,
                crs=grid.crs,
                description=(
                    f"({index_name} on {role_b.value}) - ({index_name} on "
                    f"{role_a.value}) > {delta_threshold:.2f}"
                ),
                applies_to=applies,
                threshold=delta_threshold,
                threshold_method="fixed step above inter-date noise",
                invalid=invalid,
            ),
            MaskLayer(
                key=f"delta_{index_name.lower()}_decrease",
                label=f"{index_name} fell by more than {delta_threshold:.2f}",
                array=delta_decrease,
                transform=grid.transform,
                crs=grid.crs,
                description=(
                    f"({index_name} on {role_a.value}) - ({index_name} on "
                    f"{role_b.value}) > {delta_threshold:.2f}"
                ),
                applies_to=applies,
                threshold=delta_threshold,
                threshold_method="fixed step above inter-date noise",
                invalid=invalid,
            ),
        ]

        changed = np.zeros(grid.shape, dtype=bool)
        if magnitude is not None and cva_threshold is not None:
            changed = (magnitude > cva_threshold) & ~invalid
            layers.append(
                MaskLayer(
                    key="change_magnitude",
                    label="Spectral change, any kind",
                    array=changed,
                    transform=grid.transform,
                    crs=grid.crs,
                    description=cva_note,
                    applies_to=applies,
                    threshold=cva_threshold,
                    threshold_method=f"mean + {cva_sigma:g} sigma of the magnitude",
                    invalid=invalid,
                )
            )
        else:
            outcome.notes.append(cva_note)

        outcome.masks.extend(layers)

        # -- measurements --------------------------------------------------
        before_inputs = {
            "pixel_count": int(np.count_nonzero(before)),
            "pixel_area_m2": pixel_area,
            "index": index_name,
            "threshold": round(threshold, 6),
            "class_separability": round(class_separability, 4),
            "observable_pixels": observable,
        }
        outcome.measurements.append(
            self.measurement(
                key="area_km2_before",
                label=f"{target_label} area on {role_a.value}",
                value=area_before or 0.0,
                unit="km2",
                formula=(
                    f"area = {before_inputs['pixel_count']} px x "
                    f"{pixel_area:.2f} m2 / 1e6"
                ),
                inputs=before_inputs,
                method=f"{index_name} > {threshold:.3f} ({definition.reference})",
                applies_to=[role_a],
                precision=3,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="area_km2_after",
                label=f"{target_label} area on {role_b.value}",
                value=area_after or 0.0,
                unit="km2",
                formula=(
                    f"area = {int(np.count_nonzero(after))} px x "
                    f"{pixel_area:.2f} m2 / 1e6"
                ),
                inputs={
                    **before_inputs,
                    "pixel_count": int(np.count_nonzero(after)),
                },
                method=f"{index_name} > {threshold:.3f} ({definition.reference})",
                applies_to=[role_b],
                precision=3,
            )
        )

        change_km2 = (area_after or 0.0) - (area_before or 0.0)
        outcome.measurements.append(
            self.measurement(
                key="area_change_km2",
                label=f"Change in {target_label} area",
                value=change_km2,
                unit="km2",
                formula=f"{area_after or 0.0:.4f} - {area_before or 0.0:.4f}",
                inputs={
                    "area_km2_before": round(area_before or 0.0, 6),
                    "area_km2_after": round(area_after or 0.0, 6),
                },
                applies_to=applies,
                precision=3,
            )
        )

        # A percentage of nothing is not zero, it is undefined. Saying so is more
        # useful than printing a number that cannot mean anything.
        if area_before and area_before > 0:
            outcome.measurements.append(
                self.measurement(
                    key="percentage_change",
                    label=f"{target_label} change",
                    value=100.0 * change_km2 / area_before,
                    unit="%",
                    formula=(
                        f"100 x ({area_after or 0.0:.4f} - {area_before:.4f}) / "
                        f"{area_before:.4f}"
                    ),
                    inputs={
                        "area_km2_before": round(area_before, 6),
                        "area_km2_after": round(area_after or 0.0, 6),
                    },
                    applies_to=applies,
                    precision=1,
                )
            )
        else:
            outcome.notes.append(
                f"No {target_label} was detected on {role_a.value}, so a percentage "
                "change has no denominator and none is reported. The absolute area "
                "on each date is reported instead."
            )

        for key, label, value, mask, note in (
            ("gain_km2", f"{target_label} gained", gain_km2, gain,
             f"present on {role_b.value}, absent on {role_a.value}"),
            ("loss_km2", f"{target_label} lost", loss_km2, loss,
             f"present on {role_a.value}, absent on {role_b.value}"),
            ("persistent_km2", f"{target_label} on both dates", persistent_km2,
             persistent, "present on both dates"),
        ):
            outcome.measurements.append(
                self.measurement(
                    key=key,
                    label=label,
                    value=value or 0.0,
                    unit="km2",
                    formula=(
                        f"area = {int(np.count_nonzero(mask))} px x "
                        f"{pixel_area:.2f} m2 / 1e6"
                    ),
                    inputs={
                        "pixel_count": int(np.count_nonzero(mask)),
                        "pixel_area_m2": pixel_area,
                        "definition": note,
                    },
                    applies_to=applies,
                    precision=3,
                )
            )

        outcome.measurements.append(
            self.measurement(
                key="net_change_km2",
                label="Net change",
                value=(gain_km2 or 0.0) - (loss_km2 or 0.0),
                unit="km2",
                formula=f"{gain_km2 or 0.0:.4f} gained - {loss_km2 or 0.0:.4f} lost",
                inputs={
                    "gain_km2": round(gain_km2 or 0.0, 6),
                    "loss_km2": round(loss_km2 or 0.0, 6),
                },
                method=(
                    "reported alongside gain and loss, never instead of them: "
                    "equal gain and loss nets to zero without the scene being "
                    "unchanged"
                ),
                applies_to=applies,
                precision=3,
            )
        )

        outcome.measurements.append(
            self.measurement(
                key="threshold_sensitivity",
                label="How much the answer depends on where the threshold sits",
                value=sensitivity,
                unit="fraction",
                formula=(
                    f"spread of the measured change as the threshold moves "
                    f"+/-{THRESHOLD_PROBE} / the starting area"
                ),
                inputs=sensitivity_inputs,
                method=(
                    "the threshold is nudged and the answer re-measured; a result "
                    "that survives the nudge does not rest on the exact boundary. "
                    "Separability cannot substitute for this, because an index "
                    "fails to split a scene both when it cannot distinguish the "
                    "class and when the scene is almost entirely that class, and "
                    "those two mean opposite things for trust"
                ),
                applies_to=applies,
                precision=4,
            )
        )

        outcome.measurements.append(
            self.measurement(
                key="class_separability",
                label=f"{index_name} class separability",
                value=class_separability,
                unit="ratio",
                formula=(
                    "between-class variance at the threshold / total variance, "
                    "rescaled so one population scores 0"
                ),
                inputs={
                    "threshold": round(threshold, 6),
                    "method": class_method,
                    "weak_below": WEAK_SEPARABILITY,
                },
                method=(
                    "decides whether the class areas describe a class at all: an "
                    "index that does not split the scene still yields a threshold "
                    "and an area"
                ),
                applies_to=applies,
                precision=3,
            )
        )

        outcome.measurements.append(
            self.measurement(
                key="unconfirmed_crossings_km2",
                label="Boundary crossings not confirmed by the index moving",
                value=unconfirmed_km2 or 0.0,
                unit="km2",
                formula=(
                    f"area = {int(np.count_nonzero(unconfirmed))} px x "
                    f"{pixel_area:.2f} m2 / 1e6"
                ),
                inputs={
                    "pixel_count": int(np.count_nonzero(unconfirmed)),
                    "delta_threshold": delta_threshold,
                    "crossed_up_px": int(np.count_nonzero(crossed_up)),
                    "crossed_down_px": int(np.count_nonzero(crossed_down)),
                },
                method=(
                    "pixels that ended up on the other side of the class boundary "
                    "without the index moving further than inter-date noise; "
                    "excluded from gain and loss"
                ),
                applies_to=applies,
                precision=3,
            )
        )

        # -- independent corroboration -------------------------------------
        corroborated = self._corroborate(
            context, grid, role_a, role_b, index_name, gain, invalid, delta_threshold
        )
        if corroborated is not None:
            second_name, expected, agreeing, second_delta = corroborated
            _, agreeing_km2 = mask_area_km2(agreeing, grid.transform)
            outcome.masks.append(
                MaskLayer(
                    key="gain_corroborated",
                    label=f"{target_label} gained, corroborated by {second_name}",
                    array=agreeing,
                    transform=grid.transform,
                    crs=grid.crs,
                    description=(
                        f"{index_name} rose past the class boundary and {second_name} "
                        f"{'fell' if expected < 0 else 'rose'} by more than "
                        f"{delta_threshold * CORROBORATION_FACTOR:.3f} at the same "
                        "pixel"
                    ),
                    applies_to=applies,
                    invalid=invalid,
                )
            )
            outcome.measurements.append(
                self.measurement(
                    key="corroborated_gain_km2",
                    label=f"{target_label} gained, on two indices",
                    value=agreeing_km2 or 0.0,
                    unit="km2",
                    formula=(
                        f"area = {int(np.count_nonzero(agreeing))} px x "
                        f"{pixel_area:.2f} m2 / 1e6"
                    ),
                    inputs={
                        "primary_index": index_name,
                        "corroborating_index": second_name,
                        "expected_direction": (
                            "decrease" if expected < 0 else "increase"
                        ),
                        "corroborating_step": round(
                            delta_threshold * CORROBORATION_FACTOR, 4
                        ),
                        "gain_km2_primary": round(gain_km2 or 0.0, 6),
                        "mean_corroborating_delta": round(second_delta, 6),
                    },
                    method=(
                        f"independent check: new {target_label} should also show "
                        f"{second_name} moving "
                        f"{'down' if expected < 0 else 'up'}"
                    ),
                    applies_to=applies,
                    precision=3,
                )
            )
            share = (agreeing_km2 or 0.0) / gain_km2 if gain_km2 else 0.0
            outcome.notes.append(
                f"{second_name} corroborates {share:.0%} of the measured "
                f"{target_label} gain ({agreeing_km2 or 0.0:.3f} of "
                f"{gain_km2 or 0.0:.3f} km2). The two indices are independent "
                "measurements of the same transition and are reported separately "
                "rather than combined."
            )

        if class_separability < WEAK_SEPARABILITY:
            outcome.notes.append(
                f"{index_name} separates this scene with a score of "
                f"{class_separability:.2f}, below {WEAK_SEPARABILITY}. The scene does "
                "not divide into two populations at this index, so the "
                f"{target_label} areas on each date describe where "
                f"{index_name} sits relative to a boundary rather than where "
                f"{target_label} is. Read the gain and loss figures, which require "
                "the index to have actually moved, in preference to the totals."
            )

        if unconfirmed_km2 and area_before and unconfirmed_km2 > 0.1 * area_before:
            outcome.notes.append(
                f"{unconfirmed_km2:.3f} km2 crossed the class boundary without "
                f"{index_name} moving by more than {delta_threshold:.2f}, so it is "
                "excluded from gain and loss. The raw difference between the two "
                "class totals includes those crossings and the net figure does not, "
                "which is why the two do not agree."
            )

        outcome.measurements.append(
            self.measurement(
                key="delta_mean",
                label=f"Mean {index_name} change",
                value=delta_mean,
                unit="index",
                formula=f"mean({index_name} on {role_b.value} - on {role_a.value})",
                inputs={
                    "observable_pixels": observable,
                    "delta_threshold": delta_threshold,
                    "pixels_above": int(np.count_nonzero(delta_increase)),
                    "pixels_below": int(np.count_nonzero(delta_decrease)),
                },
                applies_to=applies,
                precision=4,
            )
        )

        outcome.measurements.append(
            self.measurement(
                key="observable_fraction",
                label="Comparable on both dates",
                value=observable_fraction,
                unit="fraction",
                formula=f"{observable} observable px / {total} px in the overlap",
                inputs=observability,
                method=(
                    "a pixel must be clear of nodata and cloud on both dates for a "
                    "difference between them to mean anything"
                ),
                applies_to=applies,
                precision=4,
            )
        )

        if magnitude is not None and cva_threshold is not None:
            usable_magnitude = magnitude[~invalid]
            outcome.measurements.append(
                self.measurement(
                    key="cva_threshold",
                    label="Change vector threshold",
                    value=cva_threshold,
                    unit="reflectance",
                    formula=(
                        f"mean + {cva_sigma:g} x sd of the change vector magnitude "
                        "over observable pixels"
                    ),
                    inputs={
                        "bands": [role.value for role in cva_bands],
                        "observable_pixels": observable,
                        "sigma": cva_sigma,
                    },
                    method="change vector analysis magnitude",
                    applies_to=applies,
                    precision=4,
                )
            )
            outcome.measurements.append(
                self.measurement(
                    key="cva_mean_magnitude",
                    label="Mean change vector magnitude",
                    value=float(usable_magnitude.mean()) if usable_magnitude.size else 0.0,
                    unit="reflectance",
                    formula=(
                        "sqrt(sum over bands of (reflectance_after - "
                        "reflectance_before)^2)"
                    ),
                    inputs={"bands": [role.value for role in cva_bands]},
                    method="Wiemker change vector analysis",
                    applies_to=applies,
                    precision=4,
                )
            )
            outcome.measurements.append(
                self.measurement(
                    key="changed_fraction",
                    label="Observable area that changed spectrally",
                    value=(
                        int(np.count_nonzero(changed)) / observable if observable else 0.0
                    ),
                    unit="fraction",
                    formula=(
                        f"{int(np.count_nonzero(changed))} changed px / "
                        f"{observable} observable px"
                    ),
                    inputs={
                        "cva_threshold": round(cva_threshold, 6),
                        "observable_pixels": observable,
                    },
                    applies_to=applies,
                    precision=4,
                )
            )

        outcome.measurements.extend(
            measurements_for_mask(
                self.name,
                self.version,
                outcome.masks[0],
                label=f"{target_label} gained",
                key_prefix="gain",
                min_component_area_m2=min_area,
            )
        )

        if observability["cloud_fraction_either_date"] > 0.01:
            outcome.notes.append(
                f"{observability['cloud_fraction_either_date']:.1%} of the overlap was "
                "cloudy on at least one date and is excluded from every figure above, "
                "including the denominators."
            )

        outcome.artifacts["change.grid"] = grid
        outcome.artifacts["change.delta"] = delta
        outcome.artifacts["change.before"] = before
        outcome.artifacts["change.after"] = after
        outcome.artifacts["change.invalid"] = invalid
        outcome.artifacts["change.index"] = index_name
        if magnitude is not None:
            outcome.artifacts["change.magnitude"] = magnitude
        return outcome

    # -- helpers -----------------------------------------------------------
    def _class_threshold(
        self,
        values_a: np.ma.MaskedArray,
        values_b: np.ma.MaskedArray,
        invalid: np.ndarray,
        definition: IndexDefinition,
        method: str,
    ) -> tuple[float, float, str]:
        """One threshold for both dates, with how well it separates them.

        Deliberately one and not two. Thresholding each date independently lets
        the boundary itself move between them, and a moving boundary is
        indistinguishable from the ground changing.

        The separability is returned because it decides whether the class areas
        mean anything at all. An index that does not split the scene into two
        populations still yields a threshold, and that threshold still yields an
        area, and that area is not a measurement of the class.
        """
        pooled = np.concatenate(
            [
                np.ma.masked_array(values_a, mask=invalid).compressed(),
                np.ma.masked_array(values_b, mask=invalid).compressed(),
            ]
        )

        if method != "otsu":
            # The published threshold is fixed, so separability is measured
            # against it rather than chosen by it: it still tells the reader
            # whether the scene actually holds two populations.
            separability = _separability_at(pooled, definition.conventional_threshold)
            return definition.conventional_threshold, separability, (
                f"Both dates use the same published threshold "
                f"{definition.conventional_threshold:.2f} ({definition.reference}), so "
                f"the boundary cannot move between them. It separates this scene with "
                f"a score of {separability:.2f}."
            )

        try:
            threshold, separability = otsu_threshold(pooled)
        except ToolError:
            return (
                definition.conventional_threshold,
                _separability_at(pooled, definition.conventional_threshold),
                (
                    "The pooled distribution would not separate, so the published "
                    f"threshold {definition.conventional_threshold:.2f} was used."
                ),
            )
        return threshold, separability, (
            f"Otsu on both dates pooled gives {threshold:.3f} "
            f"(separability {separability:.2f}); pooling is what keeps one boundary "
            "for both dates."
        )

    def _corroborate(
        self,
        context: ToolContext,
        grid: CommonGrid,
        role_a: ImageRole,
        role_b: ImageRole,
        index_name: str,
        gain: np.ndarray,
        invalid: np.ndarray,
        delta_threshold: float,
    ) -> tuple[str, int, np.ndarray, float] | None:
        """Check the measured gain against a second, independent index.

        Returns None when no corroborating index applies or its bands are absent,
        which is a real answer: the finding then rests on one index and the
        evidence ledger should weigh it accordingly.
        """
        pairing = CORROBORATING_INDEX.get(index_name)
        if pairing is None:
            return None
        second_name, expected = pairing
        definition = INDEX_DEFINITIONS[second_name]

        for role in (role_a, role_b):
            if context.metadata(role).missing_roles(*definition.required_roles()):
                return None

        try:
            first = _index_on_grid(context, role_a, definition, grid)
            second = _index_on_grid(context, role_b, definition, grid)
        except ToolError:
            return None

        delta = np.ma.filled(second - first, 0.0)
        delta[invalid] = 0.0
        step = delta_threshold * CORROBORATION_FACTOR
        moved = (delta < -step) if expected < 0 else (delta > step)
        agreeing = gain & moved & ~invalid

        on_gain = delta[gain & ~invalid]
        mean_delta = float(on_gain.mean()) if on_gain.size else 0.0
        return second_name, expected, agreeing, mean_delta

    def _change_vector(
        self,
        context: ToolContext,
        grid: CommonGrid,
        role_a: ImageRole,
        role_b: ImageRole,
        invalid: np.ndarray,
        sigma: float,
    ) -> tuple[np.ndarray | None, list[BandRole], float | None, str]:
        """Magnitude of the per-pixel spectral change vector."""
        meta_a, meta_b = context.metadata(role_a), context.metadata(role_b)
        shared = [
            role
            for role in CVA_BANDS
            if meta_a.band_index(role) is not None and meta_b.band_index(role) is not None
        ]
        if len(shared) < MIN_CVA_BANDS:
            return None, [], None, (
                f"Change vector analysis needs at least {MIN_CVA_BANDS} bands present "
                f"on both dates; {len(shared)} were available, so only the index "
                "difference was used."
            )

        kwargs = grid.read_kwargs()
        squared = np.zeros(grid.shape, dtype="float64")
        used: list[BandRole] = []
        for role in shared:
            first = read_role_reflectance(
                context.path(role_a), meta_a, role, **kwargs
            )
            second = read_role_reflectance(
                context.path(role_b), meta_b, role, **kwargs
            )
            if first is None or second is None:
                continue
            difference = np.ma.filled(second - first, 0.0)
            squared += difference**2
            used.append(role)

        if len(used) < MIN_CVA_BANDS:
            return None, [], None, (
                "Change vector bands could not be read on both dates, so only the "
                "index difference was used."
            )

        magnitude = np.sqrt(squared)
        usable = magnitude[~invalid]
        if usable.size < 32:
            return None, used, None, (
                "Too few observable pixels to set a change vector threshold."
            )

        threshold = float(usable.mean() + sigma * usable.std())
        note = (
            "sqrt of the summed squared reflectance difference across "
            f"{', '.join(role.value for role in used)}, thresholded at mean + "
            f"{sigma:g} sd = {threshold:.4f}"
        )
        return magnitude, used, threshold, note


__all__ = [
    "CVA_BANDS",
    "ChangeCvaEngine",
    "CommonGrid",
    "DEFAULT_CVA_SIGMA",
    "DEFAULT_DELTA_THRESHOLD",
    "build_common_grid",
    "resolve_change_index",
]
