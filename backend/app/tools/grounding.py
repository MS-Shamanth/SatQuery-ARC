"""The grounding adapter.

"Highlight the water body" is a referring-expression grounding task: a phrase
comes in and a region must come out. A trained grounding model is the right
long-term answer, and this is not one. It is a deterministic adapter that resolves
the phrase to a published spectral index, thresholds it at the value the
literature gives, and returns the region with its measured extent.

Being explicit about that matters more than dressing it up. The output is real: a
polygon derived from real reflectance, with an area computed from the pixel grid
and a centroid in real coordinates. What it is not is a learned understanding of
language. It resolves a vocabulary, and when a phrase falls outside that
vocabulary it says so rather than returning its best guess at a region.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import (
    connected_components,
    mask_to_geojson,
    measurements_for_mask,
)
from app.tools.indices import (
    INDEX_DEFINITIONS,
    TARGET_TO_INDEX,
    WEAK_SEPARABILITY,
    _crs_of,
    _transform_of,
    compute_index,
)

logger = logging.getLogger(__name__)

# How many regions are described individually. Beyond this the tail is summarised,
# because a list of six hundred ponds is not an answer to "highlight the water".
MAX_DESCRIBED_REGIONS = 8

# MNDWI needs SWIR. Where a sensor does not carry it, NDWI is the documented
# fallback and the substitution is reported rather than made silently.
INDEX_FALLBACK: dict[str, str] = {"MNDWI": "NDWI"}


def resolve_index(target: str, meta) -> tuple[str, str]:
    """Map a phrase to an index, returning the index and how it was chosen."""
    phrase = (target or "").strip().lower()
    if not phrase:
        raise ToolError(
            "No target was given, so there is nothing to locate. Name what should "
            "be highlighted, for example water, vegetation, or built-up."
        )

    index = TARGET_TO_INDEX.get(phrase)
    if index is None:
        # Try the individual words before giving up: "the water body" should
        # resolve, "sentiment" should not.
        for token in phrase.replace("-", " ").split():
            if token in TARGET_TO_INDEX:
                index = TARGET_TO_INDEX[token]
                break

    if index is None:
        known = ", ".join(sorted(set(TARGET_TO_INDEX)))
        raise ToolError(
            f"'{target}' is not in the vocabulary this adapter can ground. "
            f"It resolves: {known}."
        )

    definition = INDEX_DEFINITIONS[index]
    missing = meta.missing_roles(*definition.required_roles())
    if not missing:
        return index, f"'{target}' resolves to {index} ({definition.reference})"

    fallback = INDEX_FALLBACK.get(index)
    if fallback is not None and not meta.missing_roles(
        *INDEX_DEFINITIONS[fallback].required_roles()
    ):
        return fallback, (
            f"'{target}' resolves to {index}, but this sensor carries no "
            f"{', '.join(r.value for r in missing)}, so {fallback} "
            f"({INDEX_DEFINITIONS[fallback].reference}) was used instead"
        )

    names = ", ".join(role.value for role in missing)
    raise ToolError(
        f"Locating '{target}' needs {index}, which requires {names}; this image "
        "does not provide them."
    )


class GroundingEngine(BaseTool):
    """Resolves a target phrase to a highlighted region with a measured extent."""

    name = "grounding-engine"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Locates a named land-cover target and returns it as a highlighted region "
        "with its area, cluster count, and centroid. Resolves a fixed vocabulary "
        "through published spectral indices rather than learned language."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            modalities=[Modality.OPTICAL],
            any_of_band_roles=[
                [BandRole.GREEN, BandRole.SWIR16],
                [BandRole.GREEN, BandRole.NIR],
                [BandRole.NIR, BandRole.RED],
                [BandRole.SWIR16, BandRole.NIR],
            ],
            requires_crs=True,
            configurations=[
                InputConfiguration.SINGLE,
                InputConfiguration.CROSS_MODAL_PAIR,
                InputConfiguration.BI_TEMPORAL_PAIR,
            ],
            description=(
                "A georeferenced optical image carrying the bands the target's "
                "index needs."
            ),
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "target": {
                "type": "string",
                "enum": sorted(set(TARGET_TO_INDEX)),
                "default": None,
                "description": "What to locate. Falls back to the contract target.",
            },
            "threshold_method": {
                "type": "string",
                "enum": ["fixed", "otsu"],
                "default": "fixed",
                "description": (
                    "Fixed uses the published threshold for the index, which is "
                    "the safer choice when a scene holds more than two classes."
                ),
            },
            "fixed_threshold": {
                "type": "number",
                "default": None,
                "description": "Overrides the index's conventional threshold.",
            },
            "min_region_area_m2": {
                "type": "number",
                "default": 4000.0,
                "minimum": 0.0,
                "description": "Regions smaller than this are not reported.",
            },
            "exclude_cloud": {
                "type": "boolean",
                "default": True,
                "description": "Drop cloud and shadow before deciding.",
            },
        }

    def produces(self) -> list[str]:
        return [
            "grounded_area_km2",
            "grounded_pixels",
            "grounded_fraction",
            "grounded_regions",
            "grounded_largest_area_km2",
            "grounded_threshold",
            "grounded_separability",
        ]

    def execute(self, context: ToolContext) -> ToolOutcome:
        target = context.param("target") or (
            context.target_classes[0] if context.target_classes else ""
        )
        role = self._role_for(context)
        meta = context.metadata(role)

        index_name, provenance = resolve_index(str(target), meta)
        definition = INDEX_DEFINITIONS[index_name]

        method = context.param("threshold_method", "fixed")
        fixed = context.param("fixed_threshold")
        min_area = float(context.param("min_region_area_m2", 4000.0))
        exclude_cloud = bool(context.param("exclude_cloud", True))

        outcome = self.outcome(
            parameters={
                "target": target,
                "index": index_name,
                "threshold_method": method,
                "fixed_threshold": fixed,
                "min_region_area_m2": min_area,
                "exclude_cloud": exclude_cloud,
            }
        )
        outcome.notes.append(provenance + ".")

        result = compute_index(
            context,
            role,
            definition,
            threshold_method=method,
            fixed_threshold=fixed,
            exclude_cloud=exclude_cloud,
        )

        transform = _transform_of(meta)
        crs = _crs_of(meta)

        # Regions below the floor are dropped from the answer, so the mask that is
        # drawn and the area that is quoted describe the same thing.
        components = connected_components(
            result.mask, transform, crs, min_area_m2=min_area
        )
        kept = np.zeros_like(result.mask, dtype=bool)
        if components:
            from skimage.measure import label as sk_label

            labelled = sk_label(result.mask.astype(bool), connectivity=2)
            keep_labels = {component.label for component in components}
            kept = np.isin(labelled, list(keep_labels))
        dropped = int(np.count_nonzero(result.mask) - np.count_nonzero(kept))

        layer = MaskLayer(
            key=f"grounded_{target}".strip("_").replace(" ", "_").lower()
            or "grounded_region",
            label=f"{target or 'target'} (grounded)",
            array=kept,
            transform=transform,
            crs=crs,
            description=(
                f"{provenance}. {definition.formula} thresholded at "
                f"{result.threshold:.4f} via {result.threshold_method}; regions "
                f"below {min_area:.0f} m2 removed."
            ),
            applies_to=[role],
            threshold=result.threshold,
            threshold_method=result.threshold_method,
            separability=result.separability,
            invalid=result.invalid,
        )
        outcome.masks.append(layer)

        outcome.measurements.extend(
            measurements_for_mask(
                self.name,
                self.version,
                layer,
                label=f"{target or 'target'} region",
                key_prefix="grounded",
                min_component_area_m2=min_area,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="grounded_threshold",
                label=f"{index_name} decision threshold",
                value=result.threshold,
                unit="index",
                formula=result.threshold_method,
                inputs={
                    "index": index_name,
                    "conventional_threshold": definition.conventional_threshold,
                    "reference": definition.reference,
                },
                method=result.threshold_method,
                applies_to=[role],
                precision=4,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="grounded_separability",
                label="Threshold separability",
                value=result.separability,
                unit="ratio",
                formula="between-class variance at the threshold / total variance",
                inputs={"weak_below": WEAK_SEPARABILITY},
                method="Otsu criterion normalised by total variance",
                applies_to=[role],
                precision=3,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="grounded_regions",
                label="Regions found",
                value=float(len(components)),
                unit="count",
                formula=(
                    f"contiguous components of the mask with area >= {min_area:.0f} m2"
                ),
                inputs={
                    "connectivity": 8,
                    "min_region_area_m2": min_area,
                    "pixels_dropped_as_too_small": dropped,
                },
                applies_to=[role],
                precision=0,
            )
        )

        if components:
            largest = components[0]
            if largest.centroid_lonlat is not None:
                outcome.notes.append(
                    f"Largest region: {largest.area_km2:.3f} km2 centred at "
                    f"{largest.centroid_lonlat[1]:.4f}N "
                    f"{largest.centroid_lonlat[0]:.4f}E."
                )
            outcome.artifacts["grounding.components"] = components[
                :MAX_DESCRIBED_REGIONS
            ]
            outcome.artifacts["grounding.geojson"] = mask_to_geojson(
                kept,
                transform,
                crs,
                min_area_m2=min_area,
                properties={"target": target, "index": index_name},
            )
        else:
            outcome.notes.append(
                f"No region of '{target}' above {min_area:.0f} m2 was found in this "
                f"image at the {index_name} threshold of {result.threshold:.3f}. "
                "That is a result, not a failure to look."
            )

        if dropped:
            outcome.notes.append(
                f"{dropped:,} scattered pixel(s) were below the "
                f"{min_area:.0f} m2 region floor and are excluded from both the "
                "highlighted area and the reported total."
            )

        if result.separability < WEAK_SEPARABILITY:
            outcome.notes.append(
                f"Threshold separability {result.separability:.2f} is below "
                f"{WEAK_SEPARABILITY}: the index does not cleanly split this scene, "
                "so the region boundary is uncertain."
            )

        outcome.artifacts["grounding.index"] = result
        return outcome

    def _role_for(self, context: ToolContext) -> ImageRole:
        """Which image to ground in.

        Grounding is a single-image task. Given a pair, the optical member is the
        one spectral indices are defined on, and for a bi-temporal pair the later
        date is the one a present-tense question is asking about.
        """
        optical = context.optical_roles()
        if not optical:
            raise ToolError("Grounding needs an optical image, and none is present.")
        for preferred in (ImageRole.SINGLE, ImageRole.OPTICAL, ImageRole.DATE_B):
            if preferred in optical:
                return preferred
        return optical[0]


__all__ = ["GroundingEngine", "resolve_index"]
