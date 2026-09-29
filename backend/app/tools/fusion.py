"""Optical and radar fusion, and the disagreement map.

Most systems that combine sensors produce one answer and hide the argument. This
one does the opposite: it keeps both answers, measures how far apart they are, and
draws where they differ.

That is the useful output. Two instruments measuring different physics over the
same ground will disagree, and the pattern of their disagreement is diagnostic:

* **Both agree.** The strongest evidence available here. Reflectance and
  backscatter are unrelated quantities, so agreement between them is not two
  measurements of the same thing, it is corroboration.
* **Radar only.** Usually cloud. The optical instrument could not see the ground,
  so its silence is not evidence of absence. This is exactly what makes radar
  worth having, and the engine checks the cloud mask to say whether that is the
  explanation here.
* **Optical only.** Often wet soil or a smooth dark surface that reflects like
  water in the visible and near infrared while still scattering radar back. Also
  what happens when wind roughens a water surface enough to brighten it.

Two agreement figures are reported rather than one. Raw overlap flatters any pair
on a scene that is mostly land, because both will correctly call 95 per cent of it
dry and agree by accident. Cohen's kappa removes that, and the gap between the two
is itself worth reading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, MaskLayer, ToolContext, ToolError, ToolOutcome
from app.tools.gis import (
    cohens_kappa,
    intersection_over_union,
    mask_area_km2,
    mask_to_geojson,
)
from app.tools.indices import SCL_CLOUD_CLASSES, _crs_of, _transform_of

logger = logging.getLogger(__name__)

# Above this the two sensors are treated as telling the same story. Chosen well
# below what a single sensor's repeatability would give, because two different
# physics measured through different atmospheres cannot be expected to align at
# the pixel like two runs of one algorithm.
STRONG_AGREEMENT_IOU = 0.6
FAIR_AGREEMENT_IOU = 0.35

# Cohen's kappa conventions, Landis and Koch 1977.
KAPPA_LABELS: tuple[tuple[float, str], ...] = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.0, "slight"),
)

# When this much of the radar-only area sits under optical cloud, cloud is the
# explanation for the disagreement rather than a sensor error.
CLOUD_EXPLAINS_SHARE = 0.5

# Optical mask keys this engine will accept, in order of preference. MNDWI is the
# better water index where SWIR exists, which is why it is first.
OPTICAL_WATER_KEYS = (
    "mndwi_mask",
    "ndwi_mask",
    "grounded_water",
    "grounded_region",
)


def kappa_label(value: float) -> str:
    for floor, label in KAPPA_LABELS:
        if value >= floor:
            return label
    return "no better than chance"


class OpticalSarFusion(BaseTool):
    """Compares an optical water result with a radar one and maps the difference."""

    name = "optical-sar-fusion"
    version = "1.0.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = (
        "Compares the optical and radar water results, scores their agreement with "
        "both raw overlap and Cohen's kappa, and draws the disagreement map: where "
        "both agree, where only radar sees water, and where only optical does."
    )

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(
            requires_crs=True,
            configurations=[InputConfiguration.CROSS_MODAL_PAIR],
            description=(
                "An optical and a radar acquisition of the same area, plus a water "
                "result from each."
            ),
        )

    def depends_on(self) -> tuple[str, ...]:
        return ("sar-backscatter-engine", "spectral-index-engine")

    def parameter_spec(self) -> dict[str, Any]:
        return {
            "optical_mask": {
                "type": "string",
                "default": None,
                "description": (
                    "Which optical mask to compare. Defaults to MNDWI, or NDWI "
                    "where SWIR is absent."
                ),
            }
        }

    def produces(self) -> list[str]:
        return [
            "agreement_iou",
            "agreement_kappa",
            "agreed_water_km2",
            "sar_only_km2",
            "optical_only_km2",
            "sar_only_under_cloud_share",
        ]

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]:
        sar = context.upstream.get("sar-backscatter-engine")
        if sar is None:
            return False, (
                "there is no radar water result to compare; the radar engine has "
                "to run first"
            )
        if not sar.ok:
            return False, (
                f"the radar engine produced no water result ({sar.skipped_reason})"
            )
        if sar.mask("sar_water") is None:
            return False, "the radar engine produced no water mask to compare"

        if self._optical_source(context) is None:
            return False, (
                "no optical water mask is available to compare the radar result "
                "against; the spectral index engine has to run first"
            )
        return True, ""

    def execute(self, context: ToolContext) -> ToolOutcome:
        sar_outcome = context.upstream.get("sar-backscatter-engine")
        if sar_outcome is None or not sar_outcome.ok:
            raise ToolError("Fusion needs a radar water result, and none was produced.")

        sar_layer = sar_outcome.mask("sar_water")
        if sar_layer is None:
            raise ToolError("The radar engine produced no water mask to compare.")

        found = self._optical_source(context)
        if found is None:
            raise ToolError(
                "No optical water mask is available, so there is nothing to compare "
                "the radar result against."
            )
        optical_tool, optical_layer = found

        outcome = self.outcome(
            parameters={
                "optical_mask": optical_layer.key,
                "optical_tool": optical_tool,
                "radar_mask": sar_layer.key,
            }
        )

        requested = context.param("optical_mask")
        if requested and requested != optical_layer.key:
            outcome.notes.append(
                f"The plan asked for the optical mask '{requested}', which no tool "
                f"in this run produced. '{optical_layer.key}' was used instead, from "
                f"{optical_tool}. The substitution is recorded rather than silent, "
                "because the comparison is between named masks and the reader has to "
                "know which ones."
            )

        if optical_layer.array.shape != sar_layer.array.shape:
            raise ToolError(
                "The optical and radar masks are on different grids "
                f"({optical_layer.array.shape} and {sar_layer.array.shape}), so they "
                "cannot be compared pixel by pixel. The pair has to be cut onto a "
                "common grid first."
            )

        transform = sar_layer.transform
        crs = sar_layer.crs

        # A pixel either sensor could not use is excluded from the comparison
        # entirely. Counting cloud as optical disagreement would score the optical
        # instrument down for something it never claimed.
        optical_invalid = (
            optical_layer.invalid
            if optical_layer.invalid is not None
            else np.zeros_like(sar_layer.array)
        )
        sar_invalid = (
            sar_layer.invalid
            if sar_layer.invalid is not None
            else np.zeros_like(sar_layer.array)
        )

        optical_water = optical_layer.array & ~optical_invalid
        radar_water = sar_layer.array & ~sar_invalid

        comparable = ~(optical_invalid | sar_invalid)
        agreed = optical_water & radar_water & comparable
        radar_only = radar_water & ~optical_water & comparable
        optical_only = optical_water & ~radar_water & comparable
        # Radar found water where the optical instrument was blind. Not a
        # disagreement: the optical result has nothing to say here.
        radar_where_optical_blind = radar_water & optical_invalid

        iou = intersection_over_union(
            optical_water & comparable, radar_water & comparable
        )
        kappa = cohens_kappa(optical_water & comparable, radar_water & comparable)

        _, agreed_km2 = mask_area_km2(agreed, transform)
        _, radar_only_km2 = mask_area_km2(radar_only, transform)
        _, optical_only_km2 = mask_area_km2(optical_only, transform)
        _, blind_km2 = mask_area_km2(radar_where_optical_blind, transform)
        _, optical_km2 = mask_area_km2(optical_water, transform)
        _, radar_km2 = mask_area_km2(radar_water, transform)

        roles = self._roles(context)

        # -- the disagreement map ----------------------------------------
        layers = [
            MaskLayer(
                key="fusion_agree_water",
                label="Both sensors agree: water",
                array=agreed,
                transform=transform,
                crs=crs,
                description=(
                    "Optical reflectance and radar backscatter independently place "
                    "water here. They measure unrelated physics, so this is "
                    "corroboration rather than repetition."
                ),
                applies_to=roles,
                invalid=~comparable,
            ),
            MaskLayer(
                key="fusion_disagree_sar_only",
                label="Radar only",
                array=radar_only,
                transform=transform,
                crs=crs,
                description=(
                    "Radar sees water where the optical result does not. Thin cloud, "
                    "haze, or shadow can suppress an optical water signal without "
                    "affecting radar at all."
                ),
                applies_to=roles,
                invalid=~comparable,
            ),
            MaskLayer(
                key="fusion_disagree_optical_only",
                label="Optical only",
                array=optical_only,
                transform=transform,
                crs=crs,
                description=(
                    "The optical result sees water where radar does not. Wet soil "
                    "and smooth dark surfaces look like water in reflectance while "
                    "still scattering radar back, and wind-roughened water brightens "
                    "in radar."
                ),
                applies_to=roles,
                invalid=~comparable,
            ),
        ]
        if radar_where_optical_blind.any():
            layers.append(
                MaskLayer(
                    key="fusion_sar_under_cloud",
                    label="Radar water where optical could not see",
                    array=radar_where_optical_blind,
                    transform=transform,
                    crs=crs,
                    description=(
                        "The optical instrument was blocked here, so it made no "
                        "claim. This area is radar's alone and is not counted as "
                        "disagreement."
                    ),
                    applies_to=roles,
                )
            )
        outcome.masks.extend(layers)

        # -- does cloud explain the radar-only area? ----------------------
        cloud_share, cloud_note = self._cloud_explains(
            context, radar_only, optical_invalid
        )

        # -- measurements -------------------------------------------------
        outcome.measurements.append(
            self.measurement(
                key="agreement_iou",
                label="Overlap between the two sensors",
                value=iou,
                unit="ratio",
                formula=(
                    "intersection / union of the optical and radar water masks over "
                    "pixels both could use"
                ),
                inputs={
                    "optical_mask": optical_layer.key,
                    "radar_mask": sar_layer.key,
                    "comparable_pixels": int(np.count_nonzero(comparable)),
                    "strong_above": STRONG_AGREEMENT_IOU,
                    "fair_above": FAIR_AGREEMENT_IOU,
                },
                method="intersection over union",
                applies_to=roles,
                precision=3,
            )
        )
        outcome.measurements.append(
            self.measurement(
                key="agreement_kappa",
                label="Agreement corrected for chance",
                value=kappa,
                unit="kappa",
                formula="(observed agreement - expected agreement) / (1 - expected)",
                inputs={
                    "interpretation": kappa_label(kappa),
                    "raw_overlap": round(iou, 4),
                    "comparable_pixels": int(np.count_nonzero(comparable)),
                },
                method=(
                    "Cohen's kappa. Two sensors that both correctly call most of a "
                    "scene dry agree by accident, and raw overlap rewards them for "
                    "it; kappa does not."
                ),
                applies_to=roles,
                precision=3,
            )
        )

        for key, label, value, mask, note in (
            ("agreed_water_km2", "Water both sensors found", agreed_km2, agreed,
             "optical and radar both positive"),
            ("sar_only_km2", "Water only radar found", radar_only_km2, radar_only,
             "radar positive, optical negative, both able to see"),
            ("optical_only_km2", "Water only the optical result found",
             optical_only_km2, optical_only,
             "optical positive, radar negative, both able to see"),
        ):
            outcome.measurements.append(
                self.measurement(
                    key=key,
                    label=label,
                    value=value or 0.0,
                    unit="km2",
                    formula=(
                        f"area = {int(np.count_nonzero(mask))} px x "
                        f"{abs(transform.a) * abs(transform.e):.2f} m2 / 1e6"
                    ),
                    inputs={
                        "pixel_count": int(np.count_nonzero(mask)),
                        "definition": note,
                        "optical_total_km2": round(optical_km2 or 0.0, 6),
                        "radar_total_km2": round(radar_km2 or 0.0, 6),
                    },
                    applies_to=roles,
                    precision=3,
                )
            )

        if cloud_share is not None:
            outcome.measurements.append(
                self.measurement(
                    key="sar_only_under_cloud_share",
                    label="Radar-only water that sits under optical cloud",
                    value=cloud_share,
                    unit="fraction",
                    formula=(
                        "radar-only pixels within the optical cloud mask or its "
                        "immediate margin / radar-only pixels"
                    ),
                    inputs={
                        "cloud_explains_above": CLOUD_EXPLAINS_SHARE,
                        "radar_only_pixels": int(np.count_nonzero(radar_only)),
                    },
                    method=(
                        "checks whether the disagreement is the optical instrument "
                        "being blocked rather than the two sensors conflicting"
                    ),
                    applies_to=roles,
                    precision=3,
                )
            )
            outcome.notes.append(cloud_note)

        outcome.notes.append(
            f"Raw overlap {iou:.2f}, kappa {kappa:.2f} ({kappa_label(kappa)}). "
            + self._agreement_reading(iou, kappa)
        )
        if blind_km2:
            outcome.notes.append(
                f"{blind_km2:.3f} km2 of radar water lies where the optical "
                "instrument could not see at all. That area is excluded from the "
                "agreement figures, because a sensor that made no claim cannot be "
                "said to disagree."
            )

        outcome.artifacts["fusion.agreed"] = agreed
        outcome.artifacts["fusion.radar_only"] = radar_only
        outcome.artifacts["fusion.optical_only"] = optical_only
        outcome.artifacts["fusion.comparable"] = comparable
        outcome.artifacts["fusion.geojson"] = {
            "agree": mask_to_geojson(
                agreed, transform, crs, properties={"agreement": "both"}
            ),
            "sar_only": mask_to_geojson(
                radar_only, transform, crs, properties={"agreement": "radar only"}
            ),
            "optical_only": mask_to_geojson(
                optical_only, transform, crs,
                properties={"agreement": "optical only"},
            ),
        }
        return outcome

    # -- helpers -----------------------------------------------------------
    def _optical_source(
        self, context: ToolContext
    ) -> tuple[str, Any] | None:
        """The optical water mask to compare against, and which tool made it.

        A requested mask name is matched loosely and then fallen back on, because
        the alternative turned out to be worse than either. The name the index
        engine gives a mask includes the index and the image role, as in
        ``ndwi_mask.optical``, and a plan that asked for ``ndwi`` exactly found
        nothing: the fusion engine skipped itself and the whole comparison this
        demonstration exists for silently did not happen. A parameter that cannot
        be honoured should be reported and worked around, not treated as an
        instruction to do nothing.
        """
        wanted = context.param("optical_mask")
        fallbacks: list[tuple[str, Any]] = []

        for tool_name, outcome in context.upstream.items():
            if not outcome.ok or tool_name == "sar-backscatter-engine":
                continue
            if wanted:
                exact = outcome.mask(wanted)
                if exact is not None:
                    return tool_name, exact
                needle = str(wanted).lower()
                for layer in outcome.masks:
                    if needle in layer.key.lower():
                        return tool_name, layer
            for prefix in OPTICAL_WATER_KEYS:
                for layer in outcome.masks:
                    if layer.key.startswith(prefix):
                        fallbacks.append((tool_name, layer))

        if fallbacks:
            if wanted:
                logger.info(
                    "requested optical mask %r matched nothing, using %s instead",
                    wanted,
                    fallbacks[0][1].key,
                )
            return fallbacks[0]
        return None

    def _roles(self, context: ToolContext) -> list[ImageRole]:
        return sorted(
            set(context.optical_roles()) | set(context.sar_roles()),
            key=lambda role: role.value,
        )

    def _cloud_explains(
        self,
        context: ToolContext,
        radar_only: np.ndarray,
        optical_invalid: np.ndarray,
    ) -> tuple[float | None, str]:
        """Is the radar-only area simply where the optical instrument was blocked?

        The disagreement being explained by cloud is a much better outcome than the
        two sensors conflicting, and it is the case this demo exists to show, so it
        is measured rather than asserted.
        """
        if not radar_only.any():
            return None, ""

        optical = next(iter(context.optical_roles()), None)
        if optical is None:
            return None, ""
        meta = context.metadata(optical)
        if meta.band_index(BandRole.SCL) is None and not optical_invalid.any():
            return None, (
                "This optical product carries no scene classification layer, so "
                "whether cloud explains the radar-only area could not be checked."
            )

        from app.tools.gis import adjacency_fraction

        blocked = optical_invalid
        if meta.band_index(BandRole.SCL) is not None:
            from app.core.raster_io import read_role

            scl = read_role(context.path(optical), meta, BandRole.SCL)
            if scl is not None and scl.shape == radar_only.shape:
                blocked = blocked | np.isin(
                    np.ma.filled(scl, 0).astype("int16"), tuple(SCL_CLOUD_CLASSES)
                )

        if not blocked.any():
            return 0.0, (
                "The optical instrument had a clear view everywhere it was compared, "
                "so cloud does not explain where the two sensors differ."
            )

        # Two pixels of margin: a cloud edge is not sharply bounded, and the
        # shadow it casts falls just outside the classified mass.
        share = adjacency_fraction(radar_only, blocked, within_pixels=2)
        if share >= CLOUD_EXPLAINS_SHARE:
            note = (
                f"{share:.0%} of the water only radar found sits in or against "
                "optical cloud. The two sensors are not conflicting there: the "
                "optical instrument could not see the ground, and radar could. That "
                "is the argument for carrying both."
            )
        else:
            note = (
                f"Only {share:.0%} of the water only radar found sits near optical "
                "cloud, so cloud does not account for the difference. The two "
                "sensors genuinely disagree about that ground, and the reason is not "
                "established here."
            )
        return share, note

    def _agreement_reading(self, iou: float, kappa: float) -> str:
        if iou >= STRONG_AGREEMENT_IOU:
            return (
                "Two instruments measuring unrelated physics placing water in the "
                "same place is the strongest evidence available from this input."
            )
        if iou >= FAIR_AGREEMENT_IOU:
            return (
                "The two sensors broadly agree on where water is but differ at the "
                "margins, which is what to expect when one is blocked by atmosphere "
                "and the other is sensitive to surface roughness."
            )
        gap = iou - max(0.0, kappa)
        return (
            "The two sensors largely do not agree, so neither result should be read "
            "as confirmed."
            + (
                " Raw overlap flatters this pair: most of the agreement is the two "
                "of them calling dry land dry."
                if gap > 0.15
                else ""
            )
        )


__all__ = [
    "CLOUD_EXPLAINS_SHARE",
    "FAIR_AGREEMENT_IOU",
    "STRONG_AGREEMENT_IOU",
    "OpticalSarFusion",
    "kappa_label",
]
