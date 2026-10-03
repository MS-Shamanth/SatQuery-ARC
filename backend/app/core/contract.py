"""Analysis Contract generation.

Three paths produce a contract, and every one of them lands in the same
validator:

1. Gemini structured output, given the registry and the readiness report.
2. A keyword rule router, used when the language model is unavailable, disabled,
   or returns a draft that does not survive validation.
3. The cache, keyed by the question and the capabilities available to answer it.

The validator is the load-bearing part. A tool name that does not resolve against
the registry is rejected, not skipped; a parameter outside a tool's declared
whitelist is dropped and recorded; a task that its input configuration cannot
support is refused. That is what stops a plausible-sounding plan from becoming an
execution trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.registry import ToolRegistry
from app.models.contract import (
    CONFOUNDER_LABEL,
    CONFOUNDER_QUESTION,
    TASK_CONFIGURATIONS,
    AnalysisContract,
    ChangeDirection,
    ConfounderKind,
    ConfounderPlan,
    ContractDraft,
    ContractRejection,
    ContractRepair,
    ContractSource,
    ContractTaskType,
    DraftParameter,
    DraftTool,
    ToolPlan,
)
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
)
from app.tools.base import ToolContext
from app.tools.indices import INDEX_DEFINITIONS, TARGET_TO_INDEX

logger = logging.getLogger(__name__)

CONTRACT_MODEL_TIMEOUT_SECONDS = 20.0


# ---------------------------------------------------------------------------
# Target vocabulary
# ---------------------------------------------------------------------------

# Phrases that name a land-cover target, longest first so "water body" wins over
# "water". Values are the canonical class name used throughout the contract.
TARGET_PHRASES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("water body", "water"),
            ("water bodies", "water"),
            ("waterbody", "water"),
            ("open water", "water"),
            ("surface water", "water"),
            ("water", "water"),
            ("lake", "water"),
            ("reservoir", "water"),
            ("river", "water"),
            ("flooded area", "water"),
            ("flood water", "water"),
            ("flooding", "water"),
            ("flooded", "water"),
            ("flood", "water"),
            ("built-up area", "built-up"),
            ("built up area", "built-up"),
            ("builtup", "built-up"),
            ("built-up", "built-up"),
            ("built up", "built-up"),
            ("urban area", "built-up"),
            ("urban", "built-up"),
            ("settlement", "built-up"),
            ("buildings", "built-up"),
            ("building", "built-up"),
            ("construction", "built-up"),
            ("impervious", "built-up"),
            ("vegetation", "vegetation"),
            ("vegetated", "vegetation"),
            ("green cover", "vegetation"),
            ("tree cover", "vegetation"),
            ("forest", "vegetation"),
            ("cropland", "vegetation"),
            ("crops", "vegetation"),
            ("crop", "vegetation"),
            ("farmland", "vegetation"),
            ("agriculture", "vegetation"),
            ("bare soil", "bare"),
            ("bare land", "bare"),
            ("barren", "bare"),
            ("moisture", "moisture"),
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

# The index that answers a question about each class, and the fallback when the
# preferred index's bands are absent.
TARGET_INDEX: dict[str, tuple[str, ...]] = {
    "water": ("MNDWI", "NDWI"),
    "built-up": ("NDBI",),
    "vegetation": ("NDVI",),
    "bare": ("NDBI",),
    "moisture": ("NDMI",),
}

GROUNDING_VERBS = (
    "highlight", "outline", "delineate", "mark", "locate", "where is",
    "where are", "show me", "show the", "segment", "isolate", "map the",
    "identify the", "find the", "extract the",
)

DESCRIPTION_CUES = (
    "describe", "description", "caption", "what do you see", "what is visible",
    "what can you see", "land cover", "land-cover", "landcover", "summarise",
    "summarize", "overview", "what is in this",
)

CHANGE_CUES = (
    "changed", "change", "difference", "differ", "between these", "since",
    "compared to", "over time", "before and after",
)

DIRECTION_WORDS: tuple[tuple[tuple[str, ...], ChangeDirection], ...] = (
    (
        ("increased", "increase", "grown", "grow", "expanded", "expansion",
         "risen", "rise", "gained", "gain", "more"),
        ChangeDirection.INCREASED,
    ),
    (
        ("decreased", "decrease", "declined", "decline", "lost", "loss",
         "shrunk", "shrank", "reduced", "reduction", "fallen", "fell", "less",
         "degraded", "degradation", "deforest"),
        ChangeDirection.DECREASED,
    ),
    (
        ("unchanged", "remained the same", "stayed the same", "stable",
         "no change"),
        ChangeDirection.UNCHANGED,
    ),
)

CROSS_MODAL_CUES = (
    "optical and sar", "sar and optical", "both sensors", "both modalities",
    "use the optical", "radar and optical", "optical and radar", "fuse",
    "fusion", "together",
)


def _normalise(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def extract_targets(query: str) -> list[str]:
    """Canonical class names the query names, in the order they appear.

    Longest phrases match first, so "built-up area" is not read as two separate
    mentions, and "water body" does not also register plain "water".
    """
    text = _normalise(query)
    found: list[tuple[int, str]] = []
    consumed: list[tuple[int, int]] = []

    for phrase, canonical in TARGET_PHRASES:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index == -1:
                break
            end = index + len(phrase)
            overlaps = any(index < c_end and c_start < end for c_start, c_end in consumed)
            if not overlaps:
                consumed.append((index, end))
                found.append((index, canonical))
            start = end

    ordered: list[str] = []
    for _, canonical in sorted(found):
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def detect_direction(query: str) -> ChangeDirection:
    text = _normalise(query)
    for words, direction in DIRECTION_WORDS:
        if any(word in text for word in words):
            return direction
    return ChangeDirection.UNSPECIFIED


def _mentions(query: str, cues: tuple[str, ...]) -> bool:
    text = _normalise(query)
    return any(cue in text for cue in cues)


def detect_task_type(
    query: str, configuration: InputConfiguration
) -> ContractTaskType:
    """Classify the request. Order matters: the most specific cue wins."""
    text = _normalise(query)
    bi_temporal = configuration is InputConfiguration.BI_TEMPORAL_PAIR
    cross_modal = configuration is InputConfiguration.CROSS_MODAL_PAIR
    direction = detect_direction(query)

    # An asserted direction is a claim to investigate, whatever else is present.
    if direction is not ChangeDirection.UNSPECIFIED and (
        bi_temporal or _mentions(query, CHANGE_CUES)
    ):
        return ContractTaskType.CLAIM_INVESTIGATION

    if _mentions(query, CROSS_MODAL_CUES) and cross_modal:
        return ContractTaskType.CROSS_MODAL_EXTRACTION

    if _mentions(query, GROUNDING_VERBS) and extract_targets(query):
        return ContractTaskType.REGION_GROUNDING

    if bi_temporal and _mentions(query, CHANGE_CUES):
        if text.startswith(("what", "where", "how")) or "?" in text:
            return ContractTaskType.CHANGE_QUESTION
        return ContractTaskType.CHANGE_DETECTION

    if _mentions(query, DESCRIPTION_CUES):
        return ContractTaskType.SCENE_DESCRIPTION

    if cross_modal:
        return ContractTaskType.CROSS_MODAL_EXTRACTION
    if bi_temporal:
        return ContractTaskType.CHANGE_QUESTION
    return ContractTaskType.VISUAL_QUESTION


def compose_claim(
    query: str,
    task_type: ContractTaskType,
    targets: list[str],
    direction: ChangeDirection,
) -> str:
    """Restate the request as something that can be supported or refuted."""
    target = targets[0] if targets else "the land cover"
    joined = " and ".join(targets) if targets else "the land cover"

    if task_type is ContractTaskType.CLAIM_INVESTIGATION:
        if direction is ChangeDirection.INCREASED:
            return f"{target.capitalize()} area increased between the two acquisitions."
        if direction is ChangeDirection.DECREASED:
            return f"{target.capitalize()} area decreased between the two acquisitions."
        if direction is ChangeDirection.UNCHANGED:
            return (
                f"{target.capitalize()} area is unchanged between the two acquisitions."
            )
        return f"{target.capitalize()} area changed between the two acquisitions."

    if task_type in (ContractTaskType.CHANGE_DETECTION, ContractTaskType.CHANGE_QUESTION):
        return (
            f"Measurable change in {joined} occurred between the two acquisitions, "
            "and it can be located."
        )

    if task_type is ContractTaskType.REGION_GROUNDING:
        return f"{target.capitalize()} is present in this scene and can be delineated."

    if task_type is ContractTaskType.CROSS_MODAL_EXTRACTION:
        return (
            f"{joined.capitalize()} can be identified more completely by combining "
            "optical and radar evidence than by either alone."
        )

    if task_type is ContractTaskType.SCENE_DESCRIPTION:
        return "The dominant land-cover classes in this scene can be characterised."

    return f"The question about {joined} can be answered from this imagery."


def plan_confounders(
    task_type: ContractTaskType,
    configuration: InputConfiguration,
    context: ToolContext,
) -> list[ConfounderKind]:
    """Which alternative explanations this particular request has to rule out."""
    kinds: list[ConfounderKind] = []
    readiness = context.readiness
    has_sar = bool(context.sar_roles())

    if configuration is InputConfiguration.BI_TEMPORAL_PAIR:
        kinds.extend(
            [
                ConfounderKind.SEASONALITY,
                ConfounderKind.MISREGISTRATION,
                ConfounderKind.RADIOMETRY,
            ]
        )
        if context.optical_roles():
            kinds.append(ConfounderKind.CLOUD_SHADOW)
    elif configuration is InputConfiguration.CROSS_MODAL_PAIR:
        kinds.extend(
            [
                ConfounderKind.SENSOR_MISMATCH,
                ConfounderKind.MISREGISTRATION,
            ]
        )
        if context.optical_roles():
            kinds.append(ConfounderKind.CLOUD_SHADOW)
        if has_sar:
            kinds.append(ConfounderKind.SAR_SPECIFIC)
    else:
        if context.optical_roles():
            kinds.append(ConfounderKind.CLOUD_SHADOW)
        if has_sar:
            kinds.append(ConfounderKind.SAR_SPECIFIC)
        kinds.append(ConfounderKind.ILLUMINATION_TERRAIN)

    # A readiness warning promotes its confounder regardless of task.
    if readiness is not None:
        if readiness.seasonal_risk and ConfounderKind.SEASONALITY not in kinds:
            kinds.append(ConfounderKind.SEASONALITY)

    return kinds


def _confounder_reason(
    kind: ConfounderKind, context: ToolContext
) -> str:
    """Why this confounder applies here, using measured values where available."""
    readiness = context.readiness
    if kind is ConfounderKind.SEASONALITY and readiness is not None:
        if readiness.month_of_year_delta is not None:
            return (
                f"the acquisitions sit about {readiness.month_of_year_delta} month(s) "
                "apart in the annual cycle"
            )
        return "acquisition dates are unknown, so seasonal offset cannot be ruled out"
    if kind is ConfounderKind.MISREGISTRATION and readiness is not None:
        for check in readiness.checks:
            if check.id == "pair.coregistration":
                return f"co-registration measured {check.measured}"
    if kind is ConfounderKind.CLOUD_SHADOW and readiness is not None:
        clouds = [
            check
            for check in readiness.checks
            if check.id.startswith("cloud.") and check.measured_numeric is not None
        ]
        if clouds:
            worst = max(clouds, key=lambda c: c.measured_numeric or 0.0)
            return f"cloud cover measured {worst.measured} on {worst.applies_to[0].value}"
    if kind is ConfounderKind.SENSOR_MISMATCH and readiness is not None:
        for check in readiness.checks:
            if check.id == "pair.resolution_match":
                return f"resolutions compared at {check.measured}"
    if kind is ConfounderKind.SAR_SPECIFIC:
        return "radar backscatter carries speckle and geometric effects"
    if kind is ConfounderKind.ILLUMINATION_TERRAIN:
        return "a single acquisition cannot separate surface change from illumination"
    if kind is ConfounderKind.RADIOMETRY:
        return "two acquisitions can differ in calibration and illumination"
    return ""


# ---------------------------------------------------------------------------
# Tool planning
# ---------------------------------------------------------------------------

# Tools each task would ideally use. Names not yet in the registry are reported
# as pending rather than silently omitted, so the contract stays honest about
# what the current build can and cannot do.
# The disagreement engine is admitted by every task that can produce two opinions
# about the same ground. It measures nothing itself; it compares what the others
# concluded, so it is listed after them and declines when fewer than two methods
# produced a comparable mask.
IDEAL_TOOLS: dict[ContractTaskType, tuple[str, ...]] = {
    ContractTaskType.SCENE_DESCRIPTION: (
        "spectral-index-engine", "gis-measure-engine", "rs-landcover-probe",
        "evidence-disagreement-engine", "rsvlm-narrator",
    ),
    ContractTaskType.VISUAL_QUESTION: (
        "spectral-index-engine", "gis-measure-engine", "rsvlm-narrator",
    ),
    ContractTaskType.REGION_GROUNDING: (
        "grounding-engine", "spectral-index-engine", "gis-measure-engine",
        "rs-landcover-probe", "evidence-disagreement-engine",
    ),
    ContractTaskType.CHANGE_DETECTION: (
        "change-cva-engine", "confounder-engine", "spectral-index-engine",
        "gis-measure-engine", "evidence-disagreement-engine",
    ),
    ContractTaskType.CHANGE_QUESTION: (
        "change-cva-engine", "confounder-engine", "spectral-index-engine",
        "gis-measure-engine", "evidence-disagreement-engine", "verdict-engine",
        "rsvlm-narrator",
    ),
    # No verdict engine here on purpose. "Find flooded built-up areas" asks for an
    # extraction, not a claim, and there is nothing to support or refute. The
    # fusion engine's agreement reading is the conclusion for this task.
    ContractTaskType.CROSS_MODAL_EXTRACTION: (
        "spectral-index-engine", "sar-backscatter-engine", "optical-sar-fusion",
        "gis-measure-engine", "evidence-disagreement-engine",
    ),
    ContractTaskType.CLAIM_INVESTIGATION: (
        "change-cva-engine", "confounder-engine", "spectral-index-engine",
        "sar-backscatter-engine", "gis-measure-engine",
        "evidence-disagreement-engine", "verdict-engine", "rsvlm-narrator",
    ),
}


def indices_for_targets(
    targets: list[str], context: ToolContext
) -> tuple[list[str], list[str]]:
    """Indices that answer the question, and notes about substitutions.

    MNDWI is preferred for water but needs SWIR. Where SWIR is absent the caller
    falls back to NDWI, and the substitution is recorded rather than silent.
    """
    chosen: list[str] = []
    notes: list[str] = []
    roles = context.optical_roles() or context.roles

    def bands_available(index_name: str) -> bool:
        definition = INDEX_DEFINITIONS[index_name]
        return all(
            context.metadata(role).has_roles(*definition.required_roles())
            for role in roles
        )

    for target in targets:
        candidates = TARGET_INDEX.get(target)
        if not candidates:
            continue
        for position, name in enumerate(candidates):
            if bands_available(name):
                if name not in chosen:
                    chosen.append(name)
                if position > 0:
                    notes.append(
                        f"{candidates[0]} is preferred for {target} but its bands are "
                        f"absent, so {name} is used instead"
                    )
                break
        else:
            missing = INDEX_DEFINITIONS[candidates[0]].required_roles()
            notes.append(
                f"no index for {target} is available: "
                f"{candidates[0]} needs {', '.join(r.value for r in missing)}"
            )

    return chosen, notes


def runnable_once_scheduled(
    name: str,
    registry: ToolRegistry,
    context: ToolContext,
    *,
    planned: set[str],
) -> bool:
    """Whether a tool that reads another's output could run on this input.

    A tool's dependencies have not executed at planning time and never will have,
    so asking whether its inputs are ready is the wrong question. Availability now
    means only "can this input support this tool", which ``can_run`` answers
    directly; whether the run has reached the tool yet is ``upstream_ready``, and
    the orchestrator asks that at execution time.

    Kept as a named function because the planner's intent is worth stating: a
    declared dependency that is itself registered is schedulable, and one that is
    not present in this build is not.
    """
    tool = registry.get(name)
    dependencies = tool.depends_on() if hasattr(tool, "depends_on") else ()
    if not all(dep in planned or dep in registry for dep in dependencies):
        return False
    allowed, _ = tool.can_run(context)
    return allowed


def plan_tools(
    task_type: ContractTaskType,
    targets: list[str],
    registry: ToolRegistry,
    context: ToolContext,
) -> tuple[list[ToolPlan], list[str]]:
    """Choose registered, runnable tools for this task.

    The router only ever plans tools that exist and can run on this input. Tools
    a task would want but that are not registered in this build are returned as
    notes, which keeps the contract truthful as the registry grows.
    """
    available = set(registry.available_names(context))
    ideal = IDEAL_TOOLS.get(task_type, ())
    notes: list[str] = []
    plans: list[ToolPlan] = []

    indices, index_notes = indices_for_targets(targets, context)
    notes.extend(index_notes)

    order = 0
    for name in ideal:
        if name not in registry:
            notes.append(f"{name} is not part of this build yet")
            continue
        if name not in available and not runnable_once_scheduled(
            name, registry, context, planned={plan.tool for plan in plans}
        ):
            reason = next(
                (
                    item.reason
                    for item in registry.availability(context)
                    if item.tool_name == name
                ),
                "unavailable on this input",
            )
            notes.append(f"{name} cannot run here: {reason}")
            continue

        tool = registry.get(name)
        parameters: dict[str, Any] = {}
        rationale = tool.summary

        if name == "grounding-engine":
            if not targets:
                notes.append(
                    "grounding-engine was not planned because the request names no "
                    "target to locate"
                )
                continue
            parameters = {"target": targets[0], "threshold_method": "fixed"}
            rationale = (
                f"locate '{targets[0]}' and return it as a region with a measured "
                "extent"
            )

        elif name == "change-cva-engine":
            if indices:
                parameters = {"index": indices[0]}
                rationale = (
                    f"measure the change in {indices[0]} between the two dates, "
                    "reporting gain and loss separately"
                )
            else:
                rationale = (
                    "measure what changed between the two dates by index difference "
                    "and change vector magnitude"
                )

        elif name == "spectral-index-engine":
            if task_type is ContractTaskType.REGION_GROUNDING and indices:
                # Grounding isolates one class, so the published threshold is the
                # right instrument: Otsu's two-class split merges classes on a
                # scene that holds more than two.
                parameters = {
                    "indices": indices,
                    "threshold_method": "fixed",
                    "fixed_threshold": 0.0,
                }
                rationale = (
                    f"compute {', '.join(indices)} and threshold at the published "
                    "value to isolate the target"
                )
            elif indices:
                parameters = {"indices": indices}
                rationale = f"compute {', '.join(indices)} as independent evidence"
            else:
                rationale = "compute all available indices to characterise the scene"

        plans.append(
            ToolPlan(
                tool=name,
                version=tool.version,
                rationale=rationale,
                parameters=parameters,
                order=order,
            )
        )
        order += 1

    return plans, notes


# ---------------------------------------------------------------------------
# The rule router
# ---------------------------------------------------------------------------


def route_offline(query: str, context: ToolContext) -> ContractDraft:
    """Build a draft from keyword rules, with no language model involved.

    Produces exactly the same shape as the Gemini path so the validator and the
    UI cannot tell them apart other than by the recorded source.
    """
    configuration = context.configuration
    task_type = detect_task_type(query, configuration)
    targets = extract_targets(query)
    direction = detect_direction(query)

    metrics: list[str] = []
    if task_type in (
        ContractTaskType.REGION_GROUNDING,
        ContractTaskType.CROSS_MODAL_EXTRACTION,
    ):
        metrics = ["area_km2", "coverage_fraction", "cluster_count"]
    elif task_type in (
        ContractTaskType.CHANGE_DETECTION,
        ContractTaskType.CHANGE_QUESTION,
        ContractTaskType.CLAIM_INVESTIGATION,
    ):
        metrics = ["area_km2_before", "area_km2_after", "percentage_change"]
    elif task_type is ContractTaskType.SCENE_DESCRIPTION:
        metrics = ["class_coverage_fraction"]

    outputs: list[str] = []
    if task_type is ContractTaskType.REGION_GROUNDING:
        outputs = ["a highlighted mask over the target", "its measured area"]
    elif task_type in (
        ContractTaskType.CHANGE_DETECTION,
        ContractTaskType.CHANGE_QUESTION,
        ContractTaskType.CLAIM_INVESTIGATION,
    ):
        outputs = ["a change map", "before and after measurements", "a verdict"]
    elif task_type is ContractTaskType.CROSS_MODAL_EXTRACTION:
        outputs = [
            "per-sensor results",
            "a disagreement map",
            "a combined interpretation",
        ]
    else:
        outputs = ["a written answer grounded in measured values"]

    return ContractDraft(
        claim=compose_claim(query, task_type, targets, direction),
        task_type=task_type,
        target_classes=targets,
        change_direction=direction,
        metrics_requested=metrics,
        confounders_to_test=plan_confounders(task_type, configuration, context),
        tools=[],  # filled by the validator from plan_tools
        parameters=[],
        expected_outputs=outputs,
        interpretation_note=(
            f"Read as a {task_type.value.replace('_', ' ')} request"
            + (f" about {', '.join(targets)}" if targets else "")
            + ", classified by keyword rules without a language model."
        ),
    )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def build_system_instruction(registry: ToolRegistry, context: ToolContext) -> str:
    """Tell the model exactly what it may plan with, and what it may not do."""
    availability = registry.availability(context)
    usable = [item for item in availability if item.available]
    blocked = [item for item in availability if not item.available]

    lines: list[str] = [
        "You plan remote-sensing analyses. You do not perform them.",
        "",
        "Absolute rule: you never produce a measurement, an area, a percentage, "
        "or any other number describing the imagery. Deterministic tools compute "
        "those. Your job is to restate the request as a testable claim, classify "
        "the task, choose tools, and name the alternative explanations that must "
        "be checked.",
        "",
        f"Input configuration: {context.configuration.value}",
    ]

    for role in context.roles:
        meta = context.metadata(role)
        roles_present = ", ".join(r.value for r in meta.resolved_roles) or "unknown"
        lines.append(
            f"- slot '{role.value}': {meta.modality.modality.value}, "
            f"bands [{roles_present}]"
            + (
                f", acquired {meta.acquisition_date.date().isoformat()}"
                if meta.acquisition_date
                else ", acquisition date unknown"
            )
        )

    if context.readiness is not None:
        warnings = [
            f"{check.label} = {check.measured}"
            for check in context.readiness.checks
            if check.status.value == "warn"
        ]
        if warnings:
            lines.append("")
            lines.append("Readiness warnings already measured: " + "; ".join(warnings))

    lines.extend(["", "Tools you may use, by exact name:"])
    for item in usable:
        tool = registry.get(item.tool_name)
        spec = tool.parameter_spec()
        parameter_text = ", ".join(sorted(spec)) if spec else "none"
        lines.append(f"- {tool.name}: {tool.summary} Parameters: {parameter_text}.")

    if blocked:
        lines.extend(["", "Tools that cannot run on this input, do not select them:"])
        for item in blocked:
            lines.append(f"- {item.tool_name}: {item.reason}")

    lines.extend(
        [
            "",
            "Choose tools only from the usable list above. Never invent a tool "
            "name. If no listed tool fits, return an empty tool list.",
            "",
            "Confounders must come from this set: "
            + ", ".join(kind.value for kind in ConfounderKind)
            + ".",
            "",
            f"Task type must be one of: "
            + ", ".join(task.value for task in ContractTaskType)
            + ". Change tasks require a bi-temporal pair; cross-modal extraction "
            "requires an optical and SAR pair.",
            "",
            "Choosing between change_detection and claim_investigation matters. "
            "If the request asserts or presupposes a direction, that it increased, "
            "decreased, was lost, grew, or stayed the same, it is a "
            "claim_investigation: the direction is a claim to be tested and the "
            "answer is a verdict. Use change_detection or change_question only for "
            "an open request to find or describe change with no direction implied. "
            "'Did vegetation decrease?' is claim_investigation. 'What changed "
            "between these dates?' is change_question.",
            "",
            "Parameter values must match the allowed spelling exactly, in lower "
            "case, for example 'otsu' not 'Otsu'.",
        ]
    )
    return "\n".join(lines)


@dataclass
class GeminiResult:
    draft: ContractDraft | None
    model: str
    error: str | None = None
    attempts: list[str] | None = None


# A 503 or 429 means "try again, or try another model". A 404 or 400 means this
# model will never work for this key, so moving on immediately is right but
# retrying it would be pointless.
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timeout",
                     "Timeout", "deadline")


def _is_transient(message: str) -> bool:
    return any(marker in message for marker in TRANSIENT_MARKERS)


def _primary_model(settings: Any) -> str:
    """The model the configured provider would try first.

    Part of the contract cache key, so a change of provider or model produces a
    different key rather than serving a plan that a different model drafted.
    """
    # Qualified by provider. Two vendors can offer a model of the same name, and a
    # cache key that ignored the vendor would serve a plan drafted elsewhere.
    provider = settings.selected_provider
    if provider == "mistral":
        chain = settings.mistral_model_chain
        first = chain[0] if chain else settings.mistral_model
    elif provider == "openrouter":
        chain = settings.openrouter_model_chain
        first = chain[0] if chain else settings.openrouter_model
    elif provider == "gemini":
        first = settings.gemini_model
    else:
        return ""
    return f"{provider}/{first}" if first else ""


def request_draft_from_model(
    query: str, registry: ToolRegistry, context: ToolContext
) -> GeminiResult:
    """Ask the configured provider for a contract draft.

    Provider-agnostic by design. Whatever comes back is parsed and then handed to
    ``validate_draft``, which is where the actual guarantees live: an invented tool
    name is rejected, a tool the task does not admit is refused, a missing engine
    is added back, and a plan the input cannot support is reclassified. Nothing
    downstream trusts the model, which is precisely why the provider can be swapped
    in configuration without touching any of it.

    A failure here is not an error path. The offline rule router plans the same
    request from keyword rules, so the worst case is a contract with a different
    provenance line and identical coverage.
    """
    from app.core.llm import LlmError, Message, get_provider
    from app.core.llm.openrouter import extract_json

    provider = get_provider()
    if not provider.configured:
        return GeminiResult(
            None, provider.name, "no language model is configured"
        )

    instruction = build_system_instruction(registry, context)
    messages = [
        Message(role="system", content=instruction),
        Message(
            role="system",
            content=(
                "Reply with a single JSON object matching the schema and nothing "
                "else: no prose, no explanation, no code fences."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Request from the user: {query!r}\n\n"
                "Produce the analysis contract for this request."
            ),
        ),
    ]

    try:
        response = provider.complete(
            messages,
            schema=ContractDraft.model_json_schema(),
            temperature=0.1,
        )
    except LlmError as exc:
        # Already redacted by the provider. Logged at warning because dropping to
        # the rule router is worth noticing, and not at error because the run is
        # unaffected.
        logger.warning("contract provider %s unusable: %s", provider.name, exc)
        return GeminiResult(None, provider.name, str(exc)[:600])

    try:
        draft = ContractDraft.model_validate(extract_json(response.text))
    except LlmError as exc:
        return GeminiResult(None, response.model, str(exc)[:600])
    except Exception as exc:  # noqa: BLE001 - a malformed draft is not a crash
        return GeminiResult(
            None, response.model, f"draft did not match the schema: {exc}"[:600]
        )

    return GeminiResult(draft, response.model, None, [response.model])


# The old name, kept so nothing that imported it breaks. The function was never
# Gemini-specific in anything but its call into the SDK.
request_draft_from_gemini = request_draft_from_model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _coerce(value: str, spec: dict[str, Any]) -> Any:
    """Convert a string parameter to the type the tool declares."""
    declared = spec.get("type")
    text = value.strip()

    if declared == "boolean":
        if text.lower() in ("true", "yes", "1"):
            return True
        if text.lower() in ("false", "no", "0"):
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if declared == "number":
        return float(text)
    if declared == "integer":
        return int(text)
    if declared == "array":
        items = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
        allowed = (spec.get("items") or {}).get("enum")
        if allowed:
            # Matched case-insensitively and normalised to the declared spelling,
            # so "ndwi" becomes "NDWI" rather than being refused.
            lookup = {str(option).lower(): option for option in allowed}
            invalid = [item for item in items if item.lower() not in lookup]
            if invalid:
                raise ValueError(
                    f"{', '.join(invalid)} not in {', '.join(map(str, allowed))}"
                )
            return [lookup[item.lower()] for item in items]
        return items
    if declared == "string":
        allowed = spec.get("enum")
        if allowed:
            # Case-insensitive match against the whitelist. "Otsu" and "otsu"
            # mean the same thing, and rejecting the former would discard a
            # correct choice over capitalisation. Unknown values are still
            # refused, so the whitelist keeps its force.
            for option in allowed:
                if text.lower() == str(option).lower():
                    return option
            raise ValueError(f"{text!r} not in {', '.join(map(str, allowed))}")
        return text
    return text


def validate_draft(
    draft: ContractDraft,
    *,
    session_id: str,
    query: str,
    registry: ToolRegistry,
    context: ToolContext,
    source: ContractSource,
    model: str | None,
) -> AnalysisContract:
    """Turn a proposal into a commitment, rejecting what cannot be honoured."""
    repairs: list[ContractRepair] = []
    rejections: list[ContractRejection] = []
    configuration = context.configuration

    # -- task type must suit the input ---------------------------------
    task_type = draft.task_type

    # A request that asserts a direction is a claim to be tested, not an open
    # request to describe change. The distinction decides whether the run ends in
    # a verdict, so it is enforced here rather than left to the model's judgement.
    # Gated on a bi-temporal input: a directional question cannot be answered
    # from one acquisition, so promoting it there would dress an unanswerable
    # request as an answerable one. Left alone, the configuration check below
    # refuses it and says why.
    asserted_direction = detect_direction(query)
    if (
        asserted_direction is not ChangeDirection.UNSPECIFIED
        and configuration is InputConfiguration.BI_TEMPORAL_PAIR
        and task_type
        in (ContractTaskType.CHANGE_DETECTION, ContractTaskType.CHANGE_QUESTION)
    ):
        repairs.append(
            ContractRepair(
                field="task_type",
                detail=(
                    f"the request asserts that it {asserted_direction.value}, so "
                    f"{task_type.value} was promoted to claim_investigation and the "
                    "run will end in a verdict"
                ),
            )
        )
        task_type = ContractTaskType.CLAIM_INVESTIGATION

    allowed = TASK_CONFIGURATIONS.get(task_type, frozenset())
    if configuration not in allowed:
        replacement = detect_task_type(query, configuration)
        rejections.append(
            ContractRejection(
                field="task_type",
                value=task_type.value,
                reason=(
                    f"{task_type.value} requires "
                    f"{' or '.join(sorted(c.value for c in allowed))}, but this "
                    f"session is {configuration.value}"
                ),
            )
        )
        repairs.append(
            ContractRepair(
                field="task_type",
                detail=f"reclassified as {replacement.value}",
            )
        )
        task_type = replacement

    # -- tools must resolve against the registry -----------------------
    requested = [entry.name for entry in draft.tools]
    resolved, unknown = registry.resolve(requested)
    for name in unknown:
        rejections.append(
            ContractRejection(
                field="tools",
                value=name,
                reason="not a registered tool",
            )
        )

    rationales = {entry.name: entry.why for entry in draft.tools}
    availability = {item.tool_name: item for item in registry.availability(context)}

    # What the task admits, and what it cannot do without, regardless of who
    # drafted the plan.
    #
    # This used to be enforced only by the rule router, which meant the guarantee
    # held on the offline path and quietly lapsed on the model path. When the
    # language model planned a verdict engine for a cross-modal extraction, the
    # run duly produced a verdict at 5% confidence: an extraction has nothing to
    # support or refute, so the figure was noise wearing the clothes of a
    # conclusion. A rule that only applies when the model happens to agree with it
    # is not a rule.
    admitted = set(IDEAL_TOOLS.get(task_type, ()))

    plans: list[ToolPlan] = []
    order = 0
    for tool in resolved:
        if admitted and tool.name not in admitted:
            rejections.append(
                ContractRejection(
                    field="tools",
                    value=tool.name,
                    reason=(
                        f"not admitted by a {task_type.value.replace('_', ' ')} "
                        "task; this task's tools are "
                        + ", ".join(sorted(admitted))
                    ),
                )
            )
            continue
        entry = availability.get(tool.name)
        if entry is not None and not entry.available:
            rejections.append(
                ContractRejection(
                    field="tools",
                    value=tool.name,
                    reason=f"cannot run on this input: {entry.reason}",
                )
            )
            continue
        plans.append(
            ToolPlan(
                tool=tool.name,
                version=tool.version,
                rationale=rationales.get(tool.name, tool.summary),
                parameters={},
                order=order,
            )
        )
        order += 1

    # -- parameters must be in each tool's whitelist -------------------
    by_name = {plan.tool: plan for plan in plans}
    for parameter in draft.parameters:
        plan = by_name.get(parameter.tool)
        if plan is None:
            rejections.append(
                ContractRejection(
                    field="parameters",
                    value=f"{parameter.tool}.{parameter.name}",
                    reason="names a tool that is not in the plan",
                )
            )
            continue
        spec = registry.get(plan.tool).parameter_spec()
        if parameter.name not in spec:
            rejections.append(
                ContractRejection(
                    field="parameters",
                    value=f"{parameter.tool}.{parameter.name}",
                    reason=(
                        "not a permitted parameter; allowed: "
                        + (", ".join(sorted(spec)) or "none")
                    ),
                )
            )
            continue
        try:
            plan.parameters[parameter.name] = _coerce(
                parameter.value, spec[parameter.name]
            )
        except ValueError as exc:
            rejections.append(
                ContractRejection(
                    field="parameters",
                    value=f"{parameter.tool}.{parameter.name}={parameter.value}",
                    reason=str(exc),
                )
            )

    targets = [target.strip().lower() for target in draft.target_classes if target.strip()]

    # -- if nothing survived, plan from the rules instead ---------------
    planning_notes: list[str] = []
    if not plans:
        plans, planning_notes = plan_tools(task_type, targets, registry, context)
        if requested:
            repairs.append(
                ContractRepair(
                    field="tools",
                    detail=(
                        "no proposed tool survived validation, so the plan was "
                        "rebuilt from the rule router"
                    ),
                )
            )
        elif plans:
            repairs.append(
                ContractRepair(
                    field="tools",
                    detail="tool plan filled in by the rule router",
                )
            )
    else:
        _, planning_notes = indices_for_targets(targets, context)
        # A grounding plan needs its threshold pinned even when the model omitted
        # parameters, otherwise Otsu merges classes on a multi-class scene.
        if task_type is ContractTaskType.REGION_GROUNDING:
            indices, _ = indices_for_targets(targets, context)
            index_plan = by_name.get("spectral-index-engine")
            if index_plan is not None and indices:
                if "indices" not in index_plan.parameters:
                    index_plan.parameters["indices"] = indices
                    repairs.append(
                        ContractRepair(
                            field="parameters",
                            detail=(
                                "restricted spectral-index-engine to "
                                f"{', '.join(indices)} for the named target"
                            ),
                        )
                    )
                # Overridden, not merely defaulted. Otsu assumes two populations;
                # isolating one class in a scene that also holds vegetation and
                # built-up puts the split in the wrong gap and inflates the area.
                # Measured on the synthetic three-class fixture: Otsu reports
                # 1.64 km2 where the published threshold recovers the true 1.0.
                if index_plan.parameters.get("threshold_method") != "fixed":
                    previous = index_plan.parameters.get("threshold_method", "otsu")
                    index_plan.parameters["threshold_method"] = "fixed"
                    repairs.append(
                        ContractRepair(
                            field="parameters",
                            detail=(
                                f"threshold_method changed from '{previous}' to "
                                "'fixed' for grounding: Otsu separates two "
                                "populations, so isolating one class in a "
                                "multi-class scene would merge it with another"
                            ),
                        )
                    )
                index_plan.parameters.setdefault("fixed_threshold", 0.0)

    # -- confounders ----------------------------------------------------
    kinds: list[ConfounderKind] = []
    for kind in draft.confounders_to_test:
        if kind not in kinds:
            kinds.append(kind)

    # The alternative explanations this input demands, whoever drafted the plan.
    #
    # The model used to be free to propose a shorter list, and on the same question
    # over the same imagery it proposed four one run and three the next. The second
    # run reached a different verdict. A finding whose thoroughness depends on which
    # tools the model felt like naming is not reproducible, and reproducibility is
    # most of what an audit trail is for. So the model may add an explanation it
    # thinks of; it may not drop one the input requires.
    required_kinds = plan_confounders(task_type, configuration, context)
    if not kinds:
        kinds = list(required_kinds)
        repairs.append(
            ContractRepair(
                field="confounders_to_test",
                detail="filled in from the configuration",
            )
        )
    else:
        missing = [kind for kind in required_kinds if kind not in kinds]
        if missing:
            kinds.extend(missing)
            repairs.append(
                ContractRepair(
                    field="confounders_to_test",
                    detail=(
                        "added "
                        + ", ".join(kind.value for kind in missing)
                        + ": this input requires them to be ruled out, and which "
                        "explanations get tested is not the planner's choice"
                    ),
                )
            )

    # Drop confounders that cannot apply to this input.
    applicable = set(plan_confounders(task_type, configuration, context))
    confounders: list[ConfounderPlan] = []
    for kind in kinds:
        if kind not in applicable and kind in (
            ConfounderKind.SEASONALITY,
            ConfounderKind.MISREGISTRATION,
            ConfounderKind.RADIOMETRY,
        ) and configuration is InputConfiguration.SINGLE:
            rejections.append(
                ContractRejection(
                    field="confounders_to_test",
                    value=kind.value,
                    reason="needs two acquisitions to test",
                )
            )
            continue
        confounders.append(
            ConfounderPlan(
                kind=kind,
                label=CONFOUNDER_LABEL[kind],
                question=CONFOUNDER_QUESTION[kind],
                reason=_confounder_reason(kind, context),
            )
        )

    # If every proposal was rejected as inapplicable, the list is empty and the
    # contract would commit to challenging nothing. Rebuild from the
    # configuration: a contract without confounders is not a contract this system
    # should offer.
    if not confounders:
        for kind in plan_confounders(task_type, configuration, context):
            confounders.append(
                ConfounderPlan(
                    kind=kind,
                    label=CONFOUNDER_LABEL[kind],
                    question=CONFOUNDER_QUESTION[kind],
                    reason=_confounder_reason(kind, context),
                )
            )
        if confounders:
            repairs.append(
                ContractRepair(
                    field="confounders_to_test",
                    detail=(
                        "no proposed confounder applied to this input, so the set "
                        "was rebuilt from the configuration"
                    ),
                )
            )

    # A contract that names alternative explanations has committed to testing
    # them, so the tool that does the testing has to be in the plan. A language
    # model asked for a tool list will not reliably include it, and without this
    # the contract promises a challenge it never carries out. Appended last, since
    # it reads the change engine's output.
    if (
        confounders
        and "confounder-engine" in registry
        and "confounder-engine" not in {plan.tool for plan in plans}
    ):
        engine = registry.get("confounder-engine")
        planned_names = {plan.tool for plan in plans}
        runnable, why_not = engine.can_run(context)
        if runnable or runnable_once_scheduled(
            "confounder-engine", registry, context, planned=planned_names
        ):
            plans.append(
                ToolPlan(
                    tool="confounder-engine",
                    version=engine.version,
                    rationale=(
                        "test the alternative explanations this contract committed "
                        "to challenging"
                    ),
                    parameters={"kinds": [kind.value for kind in kinds]},
                    order=max((plan.order for plan in plans), default=-1) + 1,
                )
            )
            repairs.append(
                ContractRepair(
                    field="tools",
                    detail=(
                        "added confounder-engine: the contract commits to testing "
                        f"{len(confounders)} alternative explanation(s) and this is "
                        "the tool that tests them"
                    ),
                )
            )
        else:
            planning_notes.append(
                f"confounder-engine cannot run here: {why_not}"
            )

    # Same rule for the verdict: a contract whose task is to test a claim has
    # committed to reaching a conclusion, so the tool that reaches one has to be
    # in the plan. Without this the run measures diligently and then stops short
    # of answering, which is the one thing the user asked for.
    if (
        task_type
        in (ContractTaskType.CLAIM_INVESTIGATION, ContractTaskType.CHANGE_QUESTION)
        and "verdict-engine" in registry
        and "verdict-engine" not in {plan.tool for plan in plans}
    ):
        engine = registry.get("verdict-engine")
        if runnable_once_scheduled(
            "verdict-engine",
            registry,
            context,
            planned={plan.tool for plan in plans},
        ):
            plans.append(
                ToolPlan(
                    tool="verdict-engine",
                    version=engine.version,
                    rationale=(
                        "weigh the evidence and commit to a verdict, or say why one "
                        "cannot be reached"
                    ),
                    order=max((plan.order for plan in plans), default=-1) + 1,
                )
            )
            repairs.append(
                ContractRepair(
                    field="tools",
                    detail=(
                        "added verdict-engine: this task tests a claim, so a "
                        "conclusion has to be reached rather than only measured"
                    ),
                )
            )

    # The plan is completed from the rule router, for whatever the model left out.
    #
    # This started as two special cases, written one at a time as the model was
    # caught omitting first the confounder engine and then the verdict engine. It
    # was also omitting the measurement engines, which is how the same question over
    # the same imagery came back with one fewer tool and a different verdict on
    # consecutive runs. Rather than keep naming the tools that must not be dropped,
    # the rule is now general: what a task measures is a property of the task and
    # the input, and the model's contribution is its reading of the question, the
    # parameters, and anything it wants to add.
    #
    # The router's own plan is the source rather than freshly constructed entries,
    # and both reasons for that showed up as bugs first. A tool added bare runs on
    # its defaults: the index engine computed every index instead of the one the
    # target needs, and the fusion engine then found no water mask to compare
    # against. A tool appended at the end runs last: the fusion engine was scheduled
    # before the index engine it depends on and skipped itself, so the whole
    # optical-versus-radar comparison silently did not happen. The router already
    # knows the parameters and the order.
    router_plans, router_notes = plan_tools(task_type, targets, registry, context)
    planned_names = {plan.tool for plan in plans}

    for template in router_plans:
        if template.tool in planned_names:
            continue
        plans.append(
            ToolPlan(
                tool=template.tool,
                version=template.version,
                rationale=template.rationale,
                parameters=dict(template.parameters),
                order=0,
            )
        )
        planned_names.add(template.tool)
        repairs.append(
            ContractRepair(
                field="tools",
                detail=(
                    f"added {template.tool}: a "
                    f"{task_type.value.replace('_', ' ')} on this input measures it, "
                    "and what gets measured is not left to the planner"
                ),
            )
        )

    # Dependency order, from the router. A tool the router did not plan keeps its
    # proposed position, after the ones it did.
    if plans:
        router_order = {plan.tool: index for index, plan in enumerate(router_plans)}
        fallback = len(router_order)
        plans.sort(
            key=lambda plan: (router_order.get(plan.tool, fallback + plan.order))
        )
        for index, plan in enumerate(plans):
            plan.order = index
        if router_notes:
            planning_notes.extend(
                note for note in router_notes if note not in planning_notes
            )

    modalities: list[Modality] = []
    for role in context.roles:
        modality = context.metadata(role).modality.modality
        if modality not in modalities:
            modalities.append(modality)

    # The direction the query asserts is a property of the query, so a draft that
    # omitted it is completed rather than left blank.
    direction = draft.change_direction
    if (
        direction is ChangeDirection.UNSPECIFIED
        and asserted_direction is not ChangeDirection.UNSPECIFIED
    ):
        direction = asserted_direction
        repairs.append(
            ContractRepair(
                field="change_direction",
                detail=f"read from the query as '{asserted_direction.value}'",
            )
        )

    note = draft.interpretation_note.strip()
    if planning_notes:
        note = (note + " " if note else "") + " ".join(
            text[0].upper() + text[1:] + "." for text in planning_notes
        )

    return AnalysisContract(
        session_id=session_id,
        query=query,
        claim=draft.claim.strip() or compose_claim(
            query, task_type, targets, direction
        ),
        task_type=task_type,
        configuration=configuration,
        required_modalities=modalities,
        acts_on=context.roles,
        target_classes=targets,
        change_direction=direction,
        metrics_requested=[m.strip() for m in draft.metrics_requested if m.strip()],
        confounders=confounders,
        tools=plans,
        expected_outputs=[o.strip() for o in draft.expected_outputs if o.strip()],
        source=source,
        model=model,
        interpretation_note=note,
        repairs=repairs,
        rejections=rejections,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def contract_hash(
    query: str, context: ToolContext, registry: ToolRegistry, model: str | None
) -> str:
    """Key a contract by the question and the capabilities available to answer it.

    Tool availability is part of the key because the same question against a
    different set of usable tools deserves a different plan.
    """
    payload = json.dumps(
        {
            "query": _normalise(query),
            "configuration": context.configuration.value,
            "tools": sorted(registry.available_names(context)),
            "model": model or "offline",
            "seasonal_risk": bool(
                context.readiness.seasonal_risk if context.readiness else False
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ContractCache:
    """Disk cache, so a repeated demo question does not re-query the model."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> AnalysisContract | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return AnalysisContract.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a stale entry is not fatal
            logger.warning("discarding unreadable cached contract %s: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, contract: AnalysisContract) -> None:
        self.path_for(key).write_text(
            contract.model_dump_json(indent=2), encoding="utf-8"
        )

    def clear(self) -> None:
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)


def get_cache() -> ContractCache:
    return ContractCache(get_settings().contract_cache_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_contract(
    query: str,
    context: ToolContext,
    registry: ToolRegistry,
    *,
    force_offline: bool = False,
    refresh: bool = False,
    cache: ContractCache | None = None,
) -> AnalysisContract:
    """Produce a validated Analysis Contract for a query."""
    started = time.perf_counter()
    settings = get_settings()
    cache = cache if cache is not None else get_cache()

    use_model = (
        settings.selected_provider != "none"
        and not settings.force_offline_contract
        and not force_offline
    )
    model_name = _primary_model(settings) if use_model else None
    key = contract_hash(query, context, registry, model_name)

    if not refresh:
        cached = cache.get(key)
        if cached is not None:
            cached.source = ContractSource.CACHE
            # The key covers the question and the capabilities, not the session,
            # so an identical question about an identically shaped input reuses
            # the plan. Rebind it to the session asking, or the contract would
            # name whichever session happened to draft it first.
            cached.session_id = context.session.session_id
            cached.acts_on = context.roles
            cached.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info("contract served from cache %s", key)
            return cached

    fallback_reason: str | None = None
    draft: ContractDraft | None = None
    source = ContractSource.OFFLINE_RULE_ROUTER

    if use_model:
        result = request_draft_from_model(query, registry, context)
        if result.draft is not None:
            draft = result.draft
            source = ContractSource.LANGUAGE_MODEL
            # Record which model actually answered, not which one was requested.
            model_name = result.model
        else:
            fallback_reason = result.error
            logger.warning(
                "no Gemini model produced a contract (%s); using rule router",
                result.error,
            )
    elif settings.force_offline_contract:
        fallback_reason = "offline contract router forced by configuration"
    elif force_offline:
        fallback_reason = "offline contract router requested"
    else:
        fallback_reason = "no Gemini API key configured"

    if draft is None:
        draft = route_offline(query, context)

    contract = validate_draft(
        draft,
        session_id=context.session.session_id,
        query=query,
        registry=registry,
        context=context,
        source=source,
        model=model_name if source is ContractSource.LANGUAGE_MODEL else None,
    )
    contract.fallback_reason = fallback_reason
    contract.contract_hash = key
    contract.duration_ms = round((time.perf_counter() - started) * 1000, 2)

    cache.put(key, contract)
    logger.info(
        "contract %s via %s: %s",
        key,
        contract.source.value,
        contract.summary_line(),
    )
    return contract


class ContractUnavailable(LookupError):
    """The submitted contract hash is not on file."""


class ContractStale(ValueError):
    """The submitted contract no longer describes the session's input."""


def candidate_hashes(
    query: str, context: ToolContext, registry: ToolRegistry
) -> set[str]:
    """Every key this query could legitimately have been cached under.

    A contract drafted with the language model and one drafted by the offline
    router key differently, and the caller does not tell us which path produced
    the contract being submitted, so both are admissible.
    """
    settings = get_settings()
    keys = {contract_hash(query, context, registry, None)}
    if settings.selected_provider != "none" and not settings.force_offline_contract:
        keys.add(contract_hash(query, context, registry, _primary_model(settings)))
    return keys


def contract_for_execution(
    submitted_hash: str,
    context: ToolContext,
    registry: ToolRegistry,
    *,
    cache: ContractCache | None = None,
) -> AnalysisContract:
    """Fetch a contract by hash and confirm it still describes this input.

    Execution accepts a hash rather than a contract body, which gives two things
    that matter. What runs is necessarily the plan the user was shown, since the
    body never leaves the server. And because the hash covers the question, the
    configuration, and the set of usable tools, a hash that no longer matches the
    session means the inputs moved under the plan; that is refused rather than
    run against imagery the plan was not made for.
    """
    cache = cache if cache is not None else get_cache()
    contract = cache.get(submitted_hash)
    if contract is None:
        raise ContractUnavailable(
            "That contract is no longer on file. Ask the question again to draft "
            "a fresh one."
        )

    if submitted_hash not in candidate_hashes(contract.query, context, registry):
        raise ContractStale(
            "The imagery or the available tools changed after this contract was "
            "drafted, so it no longer describes what would run. Ask again to draft "
            "a contract for the current input."
        )

    # The cache is keyed by question and capability, not by session.
    contract.session_id = context.session.session_id
    contract.acts_on = context.roles
    contract.contract_hash = submitted_hash
    return contract


__all__ = [
    "ContractCache",
    "ContractStale",
    "ContractUnavailable",
    "build_system_instruction",
    "candidate_hashes",
    "compose_claim",
    "contract_for_execution",
    "contract_hash",
    "detect_direction",
    "detect_task_type",
    "extract_targets",
    "generate_contract",
    "get_cache",
    "indices_for_targets",
    "plan_confounders",
    "plan_tools",
    "request_draft_from_gemini",
    "request_draft_from_model",
    "request_draft_from_model",
    "route_offline",
    "validate_draft",
]
