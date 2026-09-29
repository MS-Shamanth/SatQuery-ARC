"""The Analysis Contract.

Before anything runs, the system states what it believes it was asked, what it
will do about it, and what could make the answer wrong. The contract is shown to
the user and only then executed.

Two models, deliberately:

* ``ContractDraft`` is the wire format the language model fills in. Every field is
  a primitive or a list of primitives, because a structured-output schema cannot
  express an arbitrary parameter dictionary, and a model asked for one will
  improvise.
* ``AnalysisContract`` is what the rest of the pipeline consumes. It only ever
  comes out of the validator, which resolves tool names against the registry and
  filters parameters against each tool's declared whitelist.

The draft is a proposal. The contract is a commitment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.schemas import (
    ImageRole,
    InputConfiguration,
    Modality,
)


class ContractTaskType(str, Enum):
    """The task the query is asking for.

    These map onto the functional scope the problem statement defines: a
    single-image baseline of VQA plus captioning or grounding, multitemporal
    change understanding, and cross-modal extraction. ``claim_investigation`` is
    the one addition: a question phrased as an assertion to be tested rather than
    a fact to be looked up.
    """

    SCENE_DESCRIPTION = "scene_description"
    VISUAL_QUESTION = "visual_question"
    REGION_GROUNDING = "region_grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_QUESTION = "change_question"
    CROSS_MODAL_EXTRACTION = "cross_modal_extraction"
    CLAIM_INVESTIGATION = "claim_investigation"


# Which configurations each task can legitimately run on. Used to reject a
# contract that asks for change detection from a single image.
TASK_CONFIGURATIONS: dict[ContractTaskType, frozenset[InputConfiguration]] = {
    ContractTaskType.SCENE_DESCRIPTION: frozenset(
        {
            InputConfiguration.SINGLE,
            InputConfiguration.CROSS_MODAL_PAIR,
            InputConfiguration.BI_TEMPORAL_PAIR,
        }
    ),
    ContractTaskType.VISUAL_QUESTION: frozenset(
        {
            InputConfiguration.SINGLE,
            InputConfiguration.CROSS_MODAL_PAIR,
            InputConfiguration.BI_TEMPORAL_PAIR,
        }
    ),
    ContractTaskType.REGION_GROUNDING: frozenset(
        {
            InputConfiguration.SINGLE,
            InputConfiguration.CROSS_MODAL_PAIR,
            InputConfiguration.BI_TEMPORAL_PAIR,
        }
    ),
    ContractTaskType.CHANGE_DETECTION: frozenset(
        {InputConfiguration.BI_TEMPORAL_PAIR}
    ),
    ContractTaskType.CHANGE_QUESTION: frozenset({InputConfiguration.BI_TEMPORAL_PAIR}),
    ContractTaskType.CROSS_MODAL_EXTRACTION: frozenset(
        {InputConfiguration.CROSS_MODAL_PAIR}
    ),
    ContractTaskType.CLAIM_INVESTIGATION: frozenset(
        {
            InputConfiguration.SINGLE,
            InputConfiguration.CROSS_MODAL_PAIR,
            InputConfiguration.BI_TEMPORAL_PAIR,
        }
    ),
}


class ConfounderKind(str, Enum):
    """Alternative explanations the system will test before committing."""

    SEASONALITY = "seasonality"
    MISREGISTRATION = "misregistration"
    CLOUD_SHADOW = "cloud_shadow"
    RADIOMETRY = "radiometry"
    SENSOR_MISMATCH = "sensor_mismatch"
    SAR_SPECIFIC = "sar_specific"
    ILLUMINATION_TERRAIN = "illumination_terrain"


CONFOUNDER_LABEL: dict[ConfounderKind, str] = {
    ConfounderKind.SEASONALITY: "Seasonal phenology",
    ConfounderKind.MISREGISTRATION: "Misregistration",
    ConfounderKind.CLOUD_SHADOW: "Cloud and shadow",
    ConfounderKind.RADIOMETRY: "Radiometric difference",
    ConfounderKind.SENSOR_MISMATCH: "Sensor or resolution mismatch",
    ConfounderKind.SAR_SPECIFIC: "Radar geometry and speckle",
    ConfounderKind.ILLUMINATION_TERRAIN: "Illumination and terrain",
}

CONFOUNDER_QUESTION: dict[ConfounderKind, str] = {
    ConfounderKind.SEASONALITY: (
        "Could the annual growing cycle account for this instead of a real change?"
    ),
    ConfounderKind.MISREGISTRATION: (
        "Could the two images be slightly offset, so edges register as change?"
    ),
    ConfounderKind.CLOUD_SHADOW: (
        "Could cloud or its shadow be hiding the surface or mimicking a change?"
    ),
    ConfounderKind.RADIOMETRY: (
        "Could the two acquisitions simply be calibrated or illuminated differently?"
    ),
    ConfounderKind.SENSOR_MISMATCH: (
        "Could differing resolution or bands explain what looks like a difference?"
    ),
    ConfounderKind.SAR_SPECIFIC: (
        "Could speckle, layover, or radar shadow be producing this signal?"
    ),
    ConfounderKind.ILLUMINATION_TERRAIN: (
        "Could sun angle or terrain relief be producing this brightness difference?"
    ),
}


class ChangeDirection(str, Enum):
    """The direction a claim asserts, when it asserts one."""

    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"
    UNSPECIFIED = "unspecified"


class ContractSource(str, Enum):
    """Who drafted this contract.

    ``LANGUAGE_MODEL`` rather than a provider name, because the provider is
    configuration and the contract records which model actually answered in its
    own ``model`` field. ``GEMINI`` is kept only so contracts cached before the
    provider abstraction existed still deserialise; nothing produces it now.
    """

    LANGUAGE_MODEL = "language-model"
    GEMINI = "gemini"
    OFFLINE_RULE_ROUTER = "offline-rule-router"
    CACHE = "cache"


# ---------------------------------------------------------------------------
# Wire format: what the language model is allowed to return
# ---------------------------------------------------------------------------


class DraftParameter(BaseModel):
    """One proposed parameter, as a string pair.

    Values arrive as text and are coerced by the validator against the tool's
    declared type. Asking a language model for a typed union produces confident
    nonsense; asking for a string and converting it here does not.
    """

    tool: str
    name: str
    value: str


class DraftTool(BaseModel):
    name: str = Field(description="Must be an exact tool name from the registry")
    why: str = Field(description="One sentence on what this tool contributes")


class ContractDraft(BaseModel):
    """The language model's proposal. Primitives only.

    Field descriptions double as the instructions the model sees, so they are
    written to be read by it.
    """

    claim: str = Field(
        description=(
            "Restate the request as a single declarative sentence that can be "
            "supported or refuted. Do not answer it."
        )
    )
    task_type: ContractTaskType
    target_classes: list[str] = Field(
        default_factory=list,
        description=(
            "Land-cover or object classes the question is about, lowercase, "
            "for example water, built-up, vegetation. Empty if none apply."
        ),
    )
    change_direction: ChangeDirection = Field(
        default=ChangeDirection.UNSPECIFIED,
        description="Only if the request asserts a direction of change.",
    )
    metrics_requested: list[str] = Field(
        default_factory=list,
        description=(
            "Quantities the answer needs, for example area_km2, percentage_change, "
            "cluster_count."
        ),
    )
    confounders_to_test: list[ConfounderKind] = Field(
        default_factory=list,
        description="Alternative explanations that must be checked for this request.",
    )
    tools: list[DraftTool] = Field(
        default_factory=list,
        description=(
            "Registered tools to run, in order. Use only names given to you. "
            "Never invent a tool name."
        ),
    )
    parameters: list[DraftParameter] = Field(
        default_factory=list,
        description="Optional parameters, only those listed for each tool.",
    )
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="What the user should receive, for example a highlighted mask.",
    )
    interpretation_note: str = Field(
        default="",
        description=(
            "One sentence on how the request was read, especially any ambiguity "
            "resolved. Do not include numbers."
        ),
    )


# ---------------------------------------------------------------------------
# The validated contract
# ---------------------------------------------------------------------------


class ToolPlan(BaseModel):
    """A validated instruction to run one registered tool.

    The orchestrator consumes these directly, so nothing reaches it that did not
    survive registry resolution and parameter filtering.
    """

    tool: str
    version: str
    rationale: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class ContractRepair(BaseModel):
    """Something the validator changed, recorded rather than hidden."""

    field: str
    detail: str


class ContractRejection(BaseModel):
    """Something the validator refused outright."""

    field: str
    value: str
    reason: str


class ConfounderPlan(BaseModel):
    kind: ConfounderKind
    label: str
    question: str
    reason: str = Field(
        default="", description="Why this one applies to this particular input"
    )


class AnalysisContract(BaseModel):
    """What the system commits to doing, shown before it does it."""

    session_id: str
    query: str
    claim: str
    task_type: ContractTaskType
    configuration: InputConfiguration
    required_modalities: list[Modality] = Field(default_factory=list)
    acts_on: list[ImageRole] = Field(default_factory=list)

    target_classes: list[str] = Field(default_factory=list)
    change_direction: ChangeDirection = ChangeDirection.UNSPECIFIED
    metrics_requested: list[str] = Field(default_factory=list)

    confounders: list[ConfounderPlan] = Field(default_factory=list)
    tools: list[ToolPlan] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)

    # Provenance of the contract itself.
    source: ContractSource
    model: str | None = None
    interpretation_note: str = ""
    repairs: list[ContractRepair] = Field(default_factory=list)
    rejections: list[ContractRejection] = Field(default_factory=list)

    contract_hash: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    # Set when the language model was unavailable or its draft did not survive
    # validation, so the UI can say which path produced this.
    fallback_reason: str | None = None

    @property
    def tool_names(self) -> list[str]:
        return [plan.tool for plan in self.tools]

    @property
    def confounder_kinds(self) -> list[ConfounderKind]:
        return [plan.kind for plan in self.confounders]

    def summary_line(self) -> str:
        return (
            f"{self.task_type.value} on {self.configuration.value} using "
            f"{len(self.tools)} tool(s), testing {len(self.confounders)} confounder(s)"
        )


class ContractRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    force_offline: bool = Field(
        default=False,
        description="Skip the language model and use the rule router.",
    )
    refresh: bool = Field(default=False, description="Ignore any cached contract.")


__all__ = [
    "AnalysisContract",
    "CONFOUNDER_LABEL",
    "CONFOUNDER_QUESTION",
    "ChangeDirection",
    "ConfounderKind",
    "ConfounderPlan",
    "ContractDraft",
    "ContractRejection",
    "ContractRepair",
    "ContractRequest",
    "ContractSource",
    "ContractTaskType",
    "DraftParameter",
    "DraftTool",
    "TASK_CONFIGURATIONS",
    "ToolPlan",
]
