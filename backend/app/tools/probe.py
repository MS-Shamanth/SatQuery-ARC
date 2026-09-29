"""The learned land-cover probe, as a tool like any other.

This is the only tool in the system whose output comes from fitted weights rather
than a published formula, and it is treated accordingly: it declares itself
``LEARNED``, the interface labels it a trained model, and it carries the held-out
performance of the model that produced its answer.

It exists to address one specific, measured failure. On the sample library's real
construction scene the built-up index scores zero at separating the imagery,
because at ten metres concrete and dry soil occupy the same part of the histogram.
A single global threshold cannot resolve that. A classifier weighing all six bands
and five indices at once can do better, and the training record says exactly how
much better on data it never saw: around eighty percent recall on built-up against
bare, which is a real improvement and nowhere near solved.

Where it gets compared to the deterministic masks, the comparison is reported as a
disagreement and not as a score. There is no labelled reference for these scenes,
so declaring either side correct would be inventing the one number this system
refuses to invent. This is the same rule the optical-versus-radar debate follows,
applied to learned against computed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.adapter import (
    BAND_FEATURES,
    CLASSES,
    FEATURES,
    ProbeWeights,
    build_features,
    predict,
)
from app.core.raster_io import read_role_reflectance
from app.models.schemas import (
    BandRole,
    ImageRole,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import intersection_over_union, mask_area_km2

logger = logging.getLogger(__name__)

# The band roles the feature vector needs, in feature order.
REQUIRED_BANDS: tuple[BandRole, ...] = (
    BandRole.BLUE,
    BandRole.GREEN,
    BandRole.RED,
    BandRole.NIR,
    BandRole.SWIR16,
    BandRole.SWIR22,
)

# Below this probability the pixel is left unassigned rather than given to the most
# likely class. A classifier always has a favourite; saying so when it is barely
# ahead is more useful than a complete map with hidden guesses in it.
MIN_CONFIDENCE = 0.55

# Which deterministic mask each class can be compared against, and the index that
# produces it. Only classes with a published counterpart are comparable; there is
# no single accepted index for bare ground, so it is reported without comparison.
COMPARABLE: dict[str, tuple[str, ...]] = {
    "water": ("mndwi_mask", "ndwi_mask"),
    "vegetation": ("ndvi_mask",),
    "built-up": ("ndbi_mask",),
}


class LandCoverProbe(BaseTool):
    """Per-pixel land cover from fitted weights, with its own limits attached."""

    name = "rs-landcover-probe"
    version = "0.1.0"
    implementation = ToolImplementation.LEARNED
    summary = (
        "Classifies each pixel as water, vegetation, built-up or bare from all six "
        "reflectance bands and five indices at once, using weights fitted on "
        "synthetic spectra. Carries its own held-out accuracy and compares itself "
        "to the published indices without claiming to beat them."
    )

    def __init__(self, weights: ProbeWeights) -> None:
        self.weights = weights

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            band_roles=list(REQUIRED_BANDS),
            description=(
                "All six Sentinel-2 reflectance bands the probe was fitted on. A "
                "model cannot be asked for a prediction from features it never saw."
            ),
        )

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "min_confidence": {
                "type": "number",
                "default": MIN_CONFIDENCE,
                "minimum": 0.25,
                "maximum": 0.99,
                "description": (
                    "Below this probability a pixel is left unassigned rather than "
                    "given to the class that happens to lead."
                ),
            },
        }

    def produces(self) -> list[str]:
        keys = ["probe_held_out_accuracy", "probe_unassigned_fraction"]
        for name in CLASSES:
            keys.append(f"probe_area_km2.{name}")
            keys.append(f"probe_confidence.{name}")
        for name in COMPARABLE:
            keys.append(f"probe_agreement_iou.{name}")
        return keys

    def execute(self, context: ToolContext) -> ToolOutcome:
        role = context.primary_role()
        meta = context.metadata(role)
        missing = meta.missing_roles(*REQUIRED_BANDS)
        if missing:
            raise ToolError(
                "The probe needs every band it was fitted on. Missing: "
                + ", ".join(band.value for band in missing)
            )

        floor = float(context.param("min_confidence") or MIN_CONFIDENCE)
        outcome = self.outcome(
            parameters={
                "min_confidence": floor,
                "model_version": self.version,
                "trained_at": self.weights.training.trained_at.isoformat(),
            }
        )

        path = context.path(role)
        bands: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        invalid = None
        for band in REQUIRED_BANDS:
            values = read_role_reflectance(path, meta, band)
            if values is None:
                raise ToolError(f"Band {band.value} could not be read.")
            shape = values.shape
            bands[band.value] = np.ma.filled(values, 0.0)
            band_invalid = np.ma.getmaskarray(values)
            invalid = band_invalid if invalid is None else (invalid | band_invalid)

        if shape is None or invalid is None:  # pragma: no cover - guarded above
            raise ToolError("No bands were read.")

        features = build_features(bands)
        indices, confidence = predict(features, self.weights)
        labels = indices.reshape(shape)
        strength = confidence.reshape(shape)

        # A pixel the model is unsure about is not assigned. The share of those is
        # reported, because a map that hides its own hesitation is less useful than
        # one that shows it.
        unassigned = (strength < floor) | invalid
        transform = _transform_of(path)
        crs = _crs_of(path)

        self._declare_benchmark(outcome)

        for index, class_name in enumerate(CLASSES):
            selected = (labels == index) & ~unassigned
            layer = MaskLayer(
                key=f"probe_{class_name.replace('-', '_')}",
                label=f"{class_name.capitalize()}, by the trained probe",
                array=selected,
                transform=transform,
                crs=crs,
                description=(
                    f"Predicted {class_name} at or above {floor:.2f} probability, "
                    f"from {len(FEATURES)} features. Fitted weights, not a published "
                    "formula."
                ),
                applies_to=[role],
                invalid=unassigned,
            )
            outcome.masks.append(layer)

            _, area = mask_area_km2(selected, transform)
            outcome.measurements.append(
                self.measurement(
                    key=f"probe_area_km2.{class_name}",
                    label=f"{class_name.capitalize()} area, trained probe",
                    value=area or 0.0,
                    unit="km2",
                    formula=(
                        f"{int(np.count_nonzero(selected))} px x pixel area / 1e6"
                    ),
                    inputs={
                        "pixel_count": int(np.count_nonzero(selected)),
                        "min_confidence": floor,
                        "model_features": len(FEATURES),
                    },
                    method=(
                        "multinomial logistic regression fitted on synthetic "
                        f"spectra; held-out accuracy "
                        f"{self.weights.training.accuracy:.4f}"
                    ),
                    applies_to=[role],
                    precision=3,
                )
            )

            if selected.any():
                mean_confidence = float(strength[selected].mean())
                outcome.measurements.append(
                    self.measurement(
                        key=f"probe_confidence.{class_name}",
                        label=f"Mean probability where {class_name} was predicted",
                        value=mean_confidence,
                        unit="probability",
                        formula="mean softmax probability over the predicted pixels",
                        inputs={
                            "pixel_count": int(np.count_nonzero(selected)),
                            "held_out_recall": round(
                                self.weights.training.per_class_recall.get(
                                    class_name, 0.0
                                ),
                                4,
                            ),
                        },
                        method=(
                            "the model's own confidence, which is not the same as "
                            "being right"
                        ),
                        applies_to=[role],
                        precision=3,
                    )
                )

        outcome.measurements.append(
            self.measurement(
                key="probe_unassigned_fraction",
                label="Left unassigned by the probe",
                value=float(unassigned.mean()),
                unit="fraction",
                formula=(
                    f"pixels below {floor:.2f} probability or unreadable / all pixels"
                ),
                inputs={
                    "unassigned_pixels": int(np.count_nonzero(unassigned)),
                    "total_pixels": int(unassigned.size),
                    "min_confidence": floor,
                },
                method="counted, not estimated",
                applies_to=[role],
                precision=4,
            )
        )

        self._compare_with_indices(context, outcome, labels, unassigned, transform)
        self._state_limits(outcome)
        return outcome

    # -- reporting ---------------------------------------------------------
    def _declare_benchmark(self, outcome: ToolOutcome) -> None:
        """The model's held-out accuracy, as a measurement with its provenance.

        Recorded as a measurement so any figure the interface shows about this
        model's quality traces to the training run that produced it, rather than
        appearing as a claim in prose.
        """
        record = self.weights.training
        outcome.measurements.append(
            self.measurement(
                key="probe_held_out_accuracy",
                label="Probe accuracy on data it never saw",
                value=record.accuracy,
                unit="fraction",
                formula=(
                    f"correct predictions / {record.held_out} held-out spectra"
                ),
                inputs={
                    "held_out_samples": record.held_out,
                    "fitted_samples": record.samples,
                    "classes": record.classes,
                    "per_class_recall": record.per_class_recall,
                    "seed": record.seed,
                    "trained_at": record.trained_at.isoformat(),
                },
                method=(
                    "held out before fitting, on the synthetic training "
                    f"distribution: {record.generator}"
                ),
                precision=4,
            )
        )

    def _compare_with_indices(
        self,
        context: ToolContext,
        outcome: ToolOutcome,
        labels: np.ndarray,
        unassigned: np.ndarray,
        transform: Any,
    ) -> None:
        """Overlap with the published-index masks, reported without a winner.

        There is no labelled reference for these scenes. Calling the difference an
        error on either side would be asserting a ground truth that does not exist,
        so the overlap is stated and the reading is left open.
        """
        for tool_name, upstream in context.upstream.items():
            if not upstream.ok or tool_name == self.name:
                continue
            for class_name, prefixes in COMPARABLE.items():
                if f"probe_agreement_iou.{class_name}" in {
                    item.key for item in outcome.measurements
                }:
                    continue
                layer = next(
                    (
                        mask
                        for prefix in prefixes
                        for mask in upstream.masks
                        if mask.key.startswith(prefix)
                    ),
                    None,
                )
                if layer is None or layer.array.shape != labels.shape:
                    continue

                predicted = (labels == CLASSES.index(class_name)) & ~unassigned
                overlap = intersection_over_union(predicted, layer.array)
                _, probe_area = mask_area_km2(predicted, transform)
                _, index_area = mask_area_km2(layer.array, transform)

                outcome.measurements.append(
                    self.measurement(
                        key=f"probe_agreement_iou.{class_name}",
                        label=(
                            f"Overlap between the probe and {layer.key} on "
                            f"{class_name}"
                        ),
                        value=overlap,
                        unit="ratio",
                        formula="intersection / union of the two masks",
                        inputs={
                            "probe_area_km2": round(probe_area or 0.0, 6),
                            "index_area_km2": round(index_area or 0.0, 6),
                            "index_mask": layer.key,
                            "index_tool": tool_name,
                        },
                        method=(
                            "a comparison, not a score: no labelled reference exists "
                            "for this scene, so neither mask is treated as the answer"
                        ),
                        precision=4,
                    )
                )
                outcome.notes.append(
                    f"On {class_name} the probe and {layer.key} overlap at "
                    f"{overlap:.0%} ({probe_area or 0.0:.3f} km2 against "
                    f"{index_area or 0.0:.3f} km2). Which is closer to the ground "
                    "cannot be settled from this imagery alone."
                )

    def _state_limits(self, outcome: ToolOutcome) -> None:
        record = self.weights.training
        outcome.notes.append(
            f"This result comes from fitted weights, not a published formula. The "
            f"model scores {record.accuracy:.1%} on {record.held_out} held-out "
            f"spectra from its training distribution, which is synthetic. That is "
            "not a measurement of its accuracy on this scene."
        )
        weakest = min(record.per_class_recall.items(), key=lambda item: item[1])
        outcome.notes.append(
            f"Its weakest class is {weakest[0]} at {weakest[1]:.1%} recall. The "
            "built-up and bare confusion is the reason this probe exists and it is "
            "reduced rather than removed."
        )
        outcome.notes.append(
            "Trained on synthetic spectra with no scene-specific labels, so it has "
            "not learned this landscape. A probe trained on labelled imagery of the "
            "area would be the next step, and its accuracy there would be a real "
            "figure rather than an indicative one."
        )


def _transform_of(path: Any) -> Any:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.transform


def _crs_of(path: Any) -> Any:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.crs


__all__ = ["COMPARABLE", "MIN_CONFIDENCE", "REQUIRED_BANDS", "LandCoverProbe"]
