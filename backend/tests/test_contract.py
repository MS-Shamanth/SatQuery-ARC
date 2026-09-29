"""Analysis Contract tests.

The important ones are the rejections. A planner that quietly drops an invented
tool name is worse than one that refuses it, because the trace then claims a
capability the system does not have.

No test here calls Gemini. The model path is exercised by injecting drafts.
"""

from __future__ import annotations

import pytest

from app.core.contract import (
    ContractCache,
    build_system_instruction,
    compose_claim,
    contract_hash,
    detect_direction,
    detect_task_type,
    extract_targets,
    generate_contract,
    indices_for_targets,
    plan_confounders,
    plan_tools,
    route_offline,
    validate_draft,
)
from app.models.contract import (
    AnalysisContract,
    ChangeDirection,
    ConfounderKind,
    ContractDraft,
    ContractSource,
    ContractTaskType,
    DraftParameter,
    DraftTool,
)
from app.models.schemas import ImageRole, InputConfiguration, Modality
from tests.raster_fixtures import make_labels, make_optical_scene, make_sar_scene


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_context(make_context, tmp_path):
    scene = make_optical_scene(tmp_path / "single.tif", width=128, height=128)
    return make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)


@pytest.fixture
def bitemporal_context(make_context, tmp_path):
    labels = make_labels(128, 128)
    a = make_optical_scene(tmp_path / "a.tif", width=128, height=128, labels=labels)
    b = make_optical_scene(
        tmp_path / "b.tif", width=128, height=128, labels=labels, seed=31
    )
    return make_context(
        {ImageRole.DATE_A: a.path, ImageRole.DATE_B: b.path}, with_readiness=True
    )


@pytest.fixture
def cross_modal_context(make_context, tmp_path):
    labels = make_labels(128, 128)
    optical = make_optical_scene(tmp_path / "o.tif", width=128, height=128, labels=labels)
    sar = make_sar_scene(tmp_path / "s.tif", width=128, height=128, labels=labels)
    return make_context(
        {ImageRole.OPTICAL: optical.path, ImageRole.SAR: sar.path}, with_readiness=True
    )


@pytest.fixture
def cache(tmp_path) -> ContractCache:
    return ContractCache(tmp_path / "contracts")


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Highlight the water body referred to in the query", ["water"]),
        ("Find flooded built-up areas", ["water", "built-up"]),
        ("Has the built-up area increased since 2020?", ["built-up"]),
        ("Did vegetation decrease?", ["vegetation"]),
        ("Describe the land cover", []),
        ("Show me the reservoir and the settlement", ["water", "built-up"]),
    ],
)
def test_targets_are_extracted_in_order(query: str, expected: list[str]) -> None:
    assert extract_targets(query) == expected


def test_longest_phrase_wins_so_a_target_is_not_double_counted() -> None:
    """"built-up area" must not also register a bare "built up"."""
    assert extract_targets("the built-up area grew") == ["built-up"]
    assert extract_targets("water body") == ["water"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Has built-up area increased?", ChangeDirection.INCREASED),
        ("Did vegetation decrease?", ChangeDirection.DECREASED),
        ("Has the forest been lost?", ChangeDirection.DECREASED),
        ("Has it remained the same?", ChangeDirection.UNCHANGED),
        ("What changed between these dates?", ChangeDirection.UNSPECIFIED),
    ],
)
def test_direction_is_detected(query: str, expected: ChangeDirection) -> None:
    assert detect_direction(query) is expected


@pytest.mark.parametrize(
    ("query", "configuration", "expected"),
    [
        (
            "Highlight the water body referred to in the query",
            InputConfiguration.SINGLE,
            ContractTaskType.REGION_GROUNDING,
        ),
        (
            "Describe the land cover and major objects visible",
            InputConfiguration.SINGLE,
            ContractTaskType.SCENE_DESCRIPTION,
        ),
        (
            "How much of this scene is covered by water?",
            InputConfiguration.SINGLE,
            ContractTaskType.VISUAL_QUESTION,
        ),
        (
            "Has the built-up area increased, decreased, or remained unchanged?",
            InputConfiguration.BI_TEMPORAL_PAIR,
            ContractTaskType.CLAIM_INVESTIGATION,
        ),
        (
            "Did vegetation decrease?",
            InputConfiguration.BI_TEMPORAL_PAIR,
            ContractTaskType.CLAIM_INVESTIGATION,
        ),
        (
            "What changed between these two dates, and where?",
            InputConfiguration.BI_TEMPORAL_PAIR,
            ContractTaskType.CHANGE_QUESTION,
        ),
        (
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions",
            InputConfiguration.CROSS_MODAL_PAIR,
            ContractTaskType.CROSS_MODAL_EXTRACTION,
        ),
    ],
)
def test_task_type_is_classified(
    query: str, configuration: InputConfiguration, expected: ContractTaskType
) -> None:
    assert detect_task_type(query, configuration) is expected


def test_claims_are_declarative_and_testable() -> None:
    claim = compose_claim(
        "Has built-up area increased since 2020?",
        ContractTaskType.CLAIM_INVESTIGATION,
        ["built-up"],
        ChangeDirection.INCREASED,
    )
    assert claim == "Built-up area increased between the two acquisitions."
    assert not claim.endswith("?")

    grounding = compose_claim(
        "Highlight the water body",
        ContractTaskType.REGION_GROUNDING,
        ["water"],
        ChangeDirection.UNSPECIFIED,
    )
    assert "delineated" in grounding


# ---------------------------------------------------------------------------
# Index selection
# ---------------------------------------------------------------------------


def test_water_prefers_mndwi_when_swir_is_present(single_context) -> None:
    indices, notes = indices_for_targets(["water"], single_context)
    assert indices == ["MNDWI"]
    assert notes == []


def test_water_falls_back_to_ndwi_without_swir_and_says_so(
    make_context, tmp_path
) -> None:
    scene = make_optical_scene(
        tmp_path / "rgbn.tif",
        width=96,
        height=96,
        band_names=("blue", "green", "red", "nir"),
    )
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    indices, notes = indices_for_targets(["water"], context)

    assert indices == ["NDWI"]
    assert any("MNDWI is preferred" in note for note in notes)


def test_a_target_with_no_available_index_is_reported(make_context, tmp_path) -> None:
    scene = make_optical_scene(
        tmp_path / "rgb.tif",
        width=96,
        height=96,
        band_names=("blue", "green", "red"),
    )
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    indices, notes = indices_for_targets(["built-up"], context)

    assert indices == []
    assert any("no index for built-up" in note for note in notes)
    assert any("swir16" in note for note in notes)


# ---------------------------------------------------------------------------
# Tool planning
# ---------------------------------------------------------------------------


def test_planner_only_selects_registered_and_runnable_tools(
    tool_registry, single_context
) -> None:
    plans, notes = plan_tools(
        ContractTaskType.REGION_GROUNDING, ["water"], tool_registry, single_context
    )

    planned = [plan.tool for plan in plans]
    # Order matters: grounding locates the target, the index engine computes what
    # it thresholds, and the comparison of methods can only run after them.
    assert planned[:3] == [
        "grounding-engine",
        "spectral-index-engine",
        "gis-measure-engine",
    ]
    assert set(planned) - {
        "grounding-engine",
        "spectral-index-engine",
        "gis-measure-engine",
    } <= {"rs-landcover-probe", "evidence-disagreement-engine"}
    assert notes == [] or all("not part of this build" not in n for n in notes)
    # The grounding step must be told what to locate, or it has nothing to do.
    grounding = next(plan for plan in plans if plan.tool == "grounding-engine")
    assert grounding.parameters["target"] == "water"


def test_a_narrator_that_does_not_exist_is_reported_not_planned(
    tool_registry, single_context
) -> None:
    """Tools a task would want but this build lacks are named, not omitted."""
    _, notes = plan_tools(
        ContractTaskType.SCENE_DESCRIPTION, [], tool_registry, single_context
    )
    assert any("rsvlm-narrator is not part of this build yet" in note for note in notes)


def test_grounding_is_not_planned_without_a_target(
    tool_registry, single_context
) -> None:
    """Nothing to locate means nothing to locate, stated rather than guessed."""
    plans, notes = plan_tools(
        ContractTaskType.REGION_GROUNDING, [], tool_registry, single_context
    )
    assert all(plan.tool != "grounding-engine" for plan in plans)
    assert any("names no target to locate" in note for note in notes)


def test_grounding_plan_pins_the_published_threshold(
    tool_registry, single_context
) -> None:
    """Otsu merges classes on a multi-class scene, so grounding fixes the threshold."""
    plans, _ = plan_tools(
        ContractTaskType.REGION_GROUNDING, ["water"], tool_registry, single_context
    )
    index_plan = next(p for p in plans if p.tool == "spectral-index-engine")

    assert index_plan.parameters["indices"] == ["MNDWI"]
    assert index_plan.parameters["threshold_method"] == "fixed"
    assert index_plan.parameters["fixed_threshold"] == 0.0


def test_scene_description_plans_every_index(tool_registry, single_context) -> None:
    plans, _ = plan_tools(
        ContractTaskType.SCENE_DESCRIPTION, [], tool_registry, single_context
    )
    index_plan = next(p for p in plans if p.tool == "spectral-index-engine")
    assert index_plan.parameters == {}
    assert "all available indices" in index_plan.rationale


def test_planner_reports_a_tool_that_cannot_run_on_this_input(
    tool_registry, make_context, tmp_path
) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=96, height=96)
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    plans, notes = plan_tools(
        ContractTaskType.VISUAL_QUESTION, ["water"], tool_registry, context
    )

    assert [plan.tool for plan in plans] == ["gis-measure-engine"]
    assert any("spectral-index-engine cannot run here" in note for note in notes)
    assert any("optical" in note for note in notes)


# ---------------------------------------------------------------------------
# The offline router
# ---------------------------------------------------------------------------


def test_offline_router_produces_a_valid_draft(single_context) -> None:
    draft = route_offline("Highlight the water body", single_context)

    assert isinstance(draft, ContractDraft)
    assert draft.task_type is ContractTaskType.REGION_GROUNDING
    assert draft.target_classes == ["water"]
    assert draft.metrics_requested
    assert draft.expected_outputs
    assert "without a language model" in draft.interpretation_note


def test_offline_router_picks_confounders_from_the_configuration(
    bitemporal_context, single_context, cross_modal_context
) -> None:
    bi = route_offline("Has built-up area increased?", bitemporal_context)
    assert ConfounderKind.SEASONALITY in bi.confounders_to_test
    assert ConfounderKind.MISREGISTRATION in bi.confounders_to_test
    assert ConfounderKind.RADIOMETRY in bi.confounders_to_test

    single = route_offline("Describe the land cover", single_context)
    assert ConfounderKind.SEASONALITY not in single.confounders_to_test
    assert ConfounderKind.CLOUD_SHADOW in single.confounders_to_test

    cross = route_offline("Find flooded built-up areas", cross_modal_context)
    assert ConfounderKind.SENSOR_MISMATCH in cross.confounders_to_test
    assert ConfounderKind.SAR_SPECIFIC in cross.confounders_to_test


# ---------------------------------------------------------------------------
# Validation: the rejections that matter
# ---------------------------------------------------------------------------


def test_an_invented_tool_name_is_rejected_not_skipped(
    tool_registry, single_context
) -> None:
    """The guard against a hallucinated capability reaching the trace."""
    draft = ContractDraft(
        claim="Water is present and can be delineated.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[
            DraftTool(name="changeformer-large-v3", why="detects change"),
            DraftTool(name="spectral-index-engine", why="computes water indices"),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    assert "changeformer-large-v3" not in contract.tool_names
    assert "spectral-index-engine" in contract.tool_names
    rejected = {r.value: r for r in contract.rejections}
    assert "changeformer-large-v3" in rejected
    assert rejected["changeformer-large-v3"].reason == "not a registered tool"
    assert rejected["changeformer-large-v3"].field == "tools"


def test_every_invented_tool_is_rejected_and_the_plan_is_rebuilt(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[
            DraftTool(name="segment-anything-rs", why="segments"),
            DraftTool(name="croma-fusion-head", why="fuses"),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    assert len(contract.rejections) >= 2
    # Nothing survived, so the rule router supplied a real plan.
    assert contract.tool_names[:3] == [
        "grounding-engine",
        "spectral-index-engine",
        "gis-measure-engine",
    ]
    assert "segment-anything-rs" not in contract.tool_names
    assert "croma-fusion-head" not in contract.tool_names
    assert any("rebuilt from the rule router" in r.detail for r in contract.repairs)


def test_a_parameter_outside_the_whitelist_is_dropped_and_recorded(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(tool="spectral-index-engine", name="indices", value="MNDWI"),
            DraftParameter(
                tool="spectral-index-engine", name="learning_rate", value="0.001"
            ),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["indices"] == ["MNDWI"]
    assert "learning_rate" not in plan.parameters
    rejected = {r.value: r for r in contract.rejections}
    assert "spectral-index-engine.learning_rate" in rejected
    assert "not a permitted parameter" in rejected["spectral-index-engine.learning_rate"].reason


def test_a_parameter_value_outside_its_enum_is_rejected(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="threshold_method", value="kmeans"
            ),
            DraftParameter(
                tool="spectral-index-engine", name="indices", value="NDXX"
            ),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    reasons = " ".join(r.reason for r in contract.rejections)
    assert "kmeans" in reasons or "otsu" in reasons
    assert "NDXX" in reasons or "NDVI" in reasons


def test_parameter_types_are_coerced_from_strings(tool_registry, single_context) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="fixed_threshold", value="0.15"
            ),
            DraftParameter(
                tool="spectral-index-engine", name="exclude_cloud", value="false"
            ),
            DraftParameter(
                tool="spectral-index-engine", name="indices", value="MNDWI, NDVI"
            ),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["fixed_threshold"] == pytest.approx(0.15)
    assert plan.parameters["exclude_cloud"] is False
    assert plan.parameters["indices"] == ["MNDWI", "NDVI"]


def test_a_parameter_for_a_tool_not_in_the_plan_is_rejected(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        tools=[DraftTool(name="gis-measure-engine", why="measures")],
        parameters=[
            DraftParameter(tool="spectral-index-engine", name="indices", value="MNDWI")
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    reasons = {r.value: r.reason for r in contract.rejections}
    assert "spectral-index-engine.indices" in reasons
    assert "not in the plan" in reasons["spectral-index-engine.indices"]


def test_change_detection_on_a_single_image_is_refused_and_reclassified(
    tool_registry, single_context
) -> None:
    """A task its input cannot support must not be executed."""
    draft = ContractDraft(
        claim="Built-up area increased.",
        task_type=ContractTaskType.CHANGE_DETECTION,
        target_classes=["built-up"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Has built-up area increased?",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    assert contract.task_type is not ContractTaskType.CHANGE_DETECTION
    rejection = next(r for r in contract.rejections if r.field == "task_type")
    assert "requires bi_temporal_pair" in rejection.reason
    assert any(r.field == "task_type" for r in contract.repairs)


def test_a_directional_question_on_one_image_is_refused_not_dressed_up(
    tool_registry, single_context
) -> None:
    """One acquisition cannot support a claim about change.

    The promotion to claim_investigation is deliberately gated on a bi-temporal
    input, so this is refused and reclassified rather than presented as a claim
    the system could test.
    """
    draft = ContractDraft(
        claim="Built-up area increased.",
        task_type=ContractTaskType.CHANGE_DETECTION,
        target_classes=["built-up"],
        change_direction=ChangeDirection.INCREASED,
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Has the built-up area increased?",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert contract.task_type is not ContractTaskType.CLAIM_INVESTIGATION
    assert contract.task_type is not ContractTaskType.CHANGE_DETECTION
    rejection = next(r for r in contract.rejections if r.field == "task_type")
    assert "requires bi_temporal_pair" in rejection.reason


def test_an_asserted_direction_is_promoted_to_a_claim_investigation(
    tool_registry, bitemporal_context
) -> None:
    """A directional question must end in a verdict, not just a change map.

    Gemini classified "Did vegetation decrease?" as change_detection. Defensible
    in the abstract, but here the asserted direction is the claim under test, and
    the distinction decides whether the run produces a verdict at all. The
    validator enforces it rather than depending on the model's judgement.
    """
    draft = ContractDraft(
        claim="Vegetation decreased between the two acquisitions.",
        task_type=ContractTaskType.CHANGE_DETECTION,
        target_classes=["vegetation"],
        change_direction=ChangeDirection.UNSPECIFIED,
        tools=[DraftTool(name="spectral-index-engine", why="computes NDVI")],
    )

    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="Did vegetation decrease?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert contract.task_type is ContractTaskType.CLAIM_INVESTIGATION
    assert contract.change_direction is ChangeDirection.DECREASED
    promotion = next(
        r for r in contract.repairs if "promoted to claim_investigation" in r.detail
    )
    assert "decreased" in promotion.detail


def test_an_open_change_question_is_not_promoted(
    tool_registry, bitemporal_context
) -> None:
    """Only an asserted direction triggers the promotion."""
    draft = ContractDraft(
        claim="Measurable change occurred between the two acquisitions.",
        task_type=ContractTaskType.CHANGE_QUESTION,
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
    )

    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="What changed between these two dates, and where?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert contract.task_type is ContractTaskType.CHANGE_QUESTION
    assert contract.change_direction is ChangeDirection.UNSPECIFIED


def test_enum_parameters_match_case_insensitively(
    tool_registry, single_context
) -> None:
    """"Otsu" means otsu. Rejecting it over capitalisation discards a right answer.

    Uses a scene-description task so the coercion is observable: on a grounding
    task the threshold is deliberately overridden regardless of what was proposed.
    """
    draft = ContractDraft(
        claim="The scene can be characterised.",
        task_type=ContractTaskType.SCENE_DESCRIPTION,
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="threshold_method", value="Otsu"
            )
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Describe the land cover",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["threshold_method"] == "otsu"
    assert not any("threshold_method" in r.value for r in contract.rejections)


def test_array_enum_items_are_normalised_to_the_declared_spelling(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.VISUAL_QUESTION,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="indices", value="ndwi, mndwi"
            )
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="How much water is there?",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["indices"] == ["NDWI", "MNDWI"]
    assert not any("indices" in r.value for r in contract.rejections)


def test_grounding_overrides_a_proposed_otsu_threshold(
    tool_registry, single_context
) -> None:
    """Otsu is the wrong instrument for isolating one class, so it is replaced.

    Measured on the three-class fixture in Task 5: Otsu merges water with
    built-up and reports 1.64 km2 where the published threshold recovers the
    true 1.0. A proposal of 'otsu' for grounding is therefore overridden rather
    than deferred to, and the substitution is recorded.
    """
    draft = ContractDraft(
        claim="Water is present and can be delineated.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes water indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="threshold_method", value="otsu"
            )
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["threshold_method"] == "fixed"
    assert plan.parameters["fixed_threshold"] == 0.0
    repair = next(
        r for r in contract.repairs if "threshold_method changed from 'otsu'" in r.detail
    )
    assert "multi-class scene" in repair.detail


def test_a_non_grounding_task_keeps_its_otsu_threshold(
    tool_registry, single_context
) -> None:
    """The override is specific to isolating a named class."""
    draft = ContractDraft(
        claim="The scene can be characterised.",
        task_type=ContractTaskType.SCENE_DESCRIPTION,
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="threshold_method", value="otsu"
            )
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Describe the land cover",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert plan.parameters["threshold_method"] == "otsu"


def test_an_unknown_enum_value_is_still_rejected(tool_registry, single_context) -> None:
    """Case-insensitivity must not weaken the whitelist."""
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
        parameters=[
            DraftParameter(
                tool="spectral-index-engine", name="threshold_method", value="KMeans"
            )
        ],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert any("threshold_method" in r.value for r in contract.rejections)


def test_the_instruction_teaches_the_claim_versus_change_distinction(
    tool_registry, bitemporal_context
) -> None:
    instruction = build_system_instruction(tool_registry, bitemporal_context)
    assert "claim_investigation" in instruction
    assert "Did vegetation decrease?" in instruction
    assert "lower" in instruction and "'otsu' not 'Otsu'" in instruction


def test_a_tool_that_cannot_run_on_this_input_is_rejected(
    tool_registry, make_context, tmp_path
) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=96, height=96)
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        tools=[DraftTool(name="spectral-index-engine", why="computes water indices")],
    )

    contract = validate_draft(
        draft,
        session_id=context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    rejection = next(
        r for r in contract.rejections if r.value == "spectral-index-engine"
    )
    assert "cannot run on this input" in rejection.reason
    assert "optical" in rejection.reason


def test_a_confounder_needing_two_dates_is_refused_on_a_single_image(
    tool_registry, single_context
) -> None:
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        confounders_to_test=[
            ConfounderKind.SEASONALITY,
            ConfounderKind.CLOUD_SHADOW,
        ],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    assert ConfounderKind.SEASONALITY not in contract.confounder_kinds
    assert ConfounderKind.CLOUD_SHADOW in contract.confounder_kinds
    rejection = next(
        r for r in contract.rejections if r.value == ConfounderKind.SEASONALITY.value
    )
    assert "two acquisitions" in rejection.reason


def test_a_contract_never_commits_to_challenging_nothing(
    tool_registry, single_context
) -> None:
    """If every proposed confounder is rejected, the set is rebuilt.

    Observed against the live model: it proposed only confounders that need two
    acquisitions, all of which were refused on a single image, leaving a contract
    that challenged nothing at all.
    """
    draft = ContractDraft(
        claim="Water is present.",
        task_type=ContractTaskType.REGION_GROUNDING,
        target_classes=["water"],
        confounders_to_test=[
            ConfounderKind.SEASONALITY,
            ConfounderKind.MISREGISTRATION,
            ConfounderKind.RADIOMETRY,
        ],
        tools=[DraftTool(name="spectral-index-engine", why="computes indices")],
    )

    contract = validate_draft(
        draft,
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert len(contract.rejections) >= 3
    assert contract.confounders, "a contract must always challenge something"
    assert ConfounderKind.CLOUD_SHADOW in contract.confounder_kinds
    # However the list was completed, the completion is recorded. The three
    # proposals that need two acquisitions are still refused by name, so the
    # contract shows what it declined as well as what it settled on.
    assert any(r.field == "confounders_to_test" for r in contract.repairs)
    assert {r.value for r in contract.rejections} >= {
        ConfounderKind.SEASONALITY.value,
        ConfounderKind.MISREGISTRATION.value,
        ConfounderKind.RADIOMETRY.value,
    }


def test_confounders_carry_a_question_and_a_measured_reason(
    tool_registry, bitemporal_context
) -> None:
    draft = route_offline("Has built-up area increased?", bitemporal_context)
    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="Has built-up area increased?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.OFFLINE_RULE_ROUTER,
        model=None,
    )

    seasonality = next(
        plan for plan in contract.confounders if plan.kind is ConfounderKind.SEASONALITY
    )
    assert seasonality.label == "Seasonal phenology"
    assert seasonality.question.endswith("?")
    # The reason cites what was actually measured by the readiness gate.
    assert "month" in seasonality.reason

    misregistration = next(
        plan
        for plan in contract.confounders
        if plan.kind is ConfounderKind.MISREGISTRATION
    )
    assert "co-registration measured" in misregistration.reason


# ---------------------------------------------------------------------------
# Both paths produce the same shape
# ---------------------------------------------------------------------------


def test_offline_and_model_paths_produce_the_same_schema(
    tool_registry, single_context
) -> None:
    offline = validate_draft(
        route_offline("Highlight the water body", single_context),
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.OFFLINE_RULE_ROUTER,
        model=None,
    )
    from_model = validate_draft(
        ContractDraft(
            claim="Water is present in this scene and can be delineated.",
            task_type=ContractTaskType.REGION_GROUNDING,
            target_classes=["water"],
            metrics_requested=["area_km2"],
            confounders_to_test=[ConfounderKind.CLOUD_SHADOW],
            tools=[
                DraftTool(name="spectral-index-engine", why="computes water indices"),
                DraftTool(name="gis-measure-engine", why="measures the area"),
            ],
            expected_outputs=["a highlighted mask"],
            interpretation_note="Read as a request to outline water.",
        ),
        session_id=single_context.session.session_id,
        query="Highlight the water body",
        registry=tool_registry,
        context=single_context,
        source=ContractSource.GEMINI,
        model="gemini-2.5-flash",
    )

    assert isinstance(offline, AnalysisContract)
    assert isinstance(from_model, AnalysisContract)
    assert offline.task_type is from_model.task_type
    # The two paths need not choose identically: a valid model plan is respected
    # rather than overwritten. What must hold is that both plan only real tools.
    assert offline.tools and from_model.tools
    for contract in (offline, from_model):
        assert all(name in tool_registry for name in contract.tool_names)
    assert offline.source is ContractSource.OFFLINE_RULE_ROUTER
    assert from_model.source is ContractSource.GEMINI
    assert offline.model is None
    assert from_model.model == "gemini-2.5-flash"


def test_contract_records_what_it_will_act_on(tool_registry, cross_modal_context) -> None:
    contract = validate_draft(
        route_offline("Find flooded built-up areas", cross_modal_context),
        session_id=cross_modal_context.session.session_id,
        query="Find flooded built-up areas",
        registry=tool_registry,
        context=cross_modal_context,
        source=ContractSource.OFFLINE_RULE_ROUTER,
        model=None,
    )

    assert contract.configuration is InputConfiguration.CROSS_MODAL_PAIR
    assert set(contract.acts_on) == {ImageRole.OPTICAL, ImageRole.SAR}
    assert set(contract.required_modalities) == {Modality.OPTICAL, Modality.SAR}
    assert contract.target_classes == ["water", "built-up"]
    assert contract.summary_line()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_the_same_question_hashes_to_the_same_key(
    tool_registry, single_context
) -> None:
    first = contract_hash("Highlight the water body", single_context, tool_registry, None)
    second = contract_hash(
        "  HIGHLIGHT   the Water Body ", single_context, tool_registry, None
    )
    assert first == second
    assert len(first) == 32


def test_a_different_question_hashes_differently(tool_registry, single_context) -> None:
    assert contract_hash(
        "Highlight the water body", single_context, tool_registry, None
    ) != contract_hash("Describe the land cover", single_context, tool_registry, None)


def test_the_key_changes_when_the_available_tools_change(
    tool_registry, single_context
) -> None:
    """The same question deserves a different plan under different capabilities."""
    before = contract_hash("Describe the land cover", single_context, tool_registry, None)
    tool_registry.unregister("spectral-index-engine")
    after = contract_hash("Describe the land cover", single_context, tool_registry, None)
    assert before != after


def test_a_cached_contract_is_reused_and_labelled(
    tool_registry, single_context, cache, monkeypatch
) -> None:
    monkeypatch.setattr("app.core.contract.get_settings", _offline_settings)

    first = generate_contract(
        "Highlight the water body", single_context, tool_registry, cache=cache
    )
    second = generate_contract(
        "Highlight the water body", single_context, tool_registry, cache=cache
    )

    assert first.source is ContractSource.OFFLINE_RULE_ROUTER
    assert second.source is ContractSource.CACHE
    assert second.contract_hash == first.contract_hash
    assert second.claim == first.claim


def test_refresh_bypasses_the_cache(
    tool_registry, single_context, cache, monkeypatch
) -> None:
    monkeypatch.setattr("app.core.contract.get_settings", _offline_settings)

    generate_contract("Describe the land cover", single_context, tool_registry, cache=cache)
    refreshed = generate_contract(
        "Describe the land cover",
        single_context,
        tool_registry,
        cache=cache,
        refresh=True,
    )

    assert refreshed.source is ContractSource.OFFLINE_RULE_ROUTER


def test_an_unreadable_cache_entry_is_discarded(cache) -> None:
    cache.path_for("deadbeef").write_text("{not json", encoding="utf-8")
    assert cache.get("deadbeef") is None
    assert not cache.path_for("deadbeef").exists()


def _offline_settings():
    """Settings with no Gemini key, so generate_contract takes the rule path."""
    from app.config import Settings

    return Settings(
        _env_file=None,
        gemini_api_key="",
        cdse_username="",
        cdse_password="",
        satquery_env="test",
    )


def test_generate_records_why_it_fell_back(
    tool_registry, single_context, cache, monkeypatch
) -> None:
    monkeypatch.setattr("app.core.contract.get_settings", _offline_settings)

    contract = generate_contract(
        "Highlight the water body", single_context, tool_registry, cache=cache
    )

    assert contract.source is ContractSource.OFFLINE_RULE_ROUTER
    assert contract.fallback_reason == "no Gemini API key configured"
    assert contract.duration_ms >= 0


def test_force_offline_is_honoured_even_with_a_key(
    tool_registry, single_context, cache
) -> None:
    contract = generate_contract(
        "Highlight the water body",
        single_context,
        tool_registry,
        force_offline=True,
        cache=cache,
    )
    assert contract.source is ContractSource.OFFLINE_RULE_ROUTER
    assert contract.fallback_reason == "offline contract router requested"
    assert contract.model is None


# ---------------------------------------------------------------------------
# The prompt the model receives
# ---------------------------------------------------------------------------


def test_system_instruction_constrains_the_model_to_real_tools(
    tool_registry, single_context
) -> None:
    instruction = build_system_instruction(tool_registry, single_context)

    assert "never produce a measurement" in instruction
    assert "spectral-index-engine" in instruction
    assert "Never invent a tool name" in instruction
    # Parameter whitelists are stated, so a proposal can be legal by construction.
    assert "threshold_method" in instruction
    # Every confounder and task the schema allows is enumerated.
    for kind in ConfounderKind:
        assert kind.value in instruction
    for task in ContractTaskType:
        assert task.value in instruction


def test_system_instruction_lists_blocked_tools_with_reasons(
    tool_registry, make_context, tmp_path
) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=96, height=96)
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    instruction = build_system_instruction(tool_registry, context)

    assert "cannot run on this input, do not select them" in instruction
    assert "spectral-index-engine" in instruction
    assert "optical" in instruction


def test_system_instruction_describes_the_actual_input(
    tool_registry, bitemporal_context
) -> None:
    instruction = build_system_instruction(tool_registry, bitemporal_context)

    assert "bi_temporal_pair" in instruction
    assert "slot 'date_a'" in instruction
    assert "slot 'date_b'" in instruction
    assert "swir16" in instruction


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_contract_route_returns_a_plan(api, store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "s.tif", width=96, height=96)
    record = store.create()
    store.ingest(record.session_id, ImageRole.SINGLE, scene.path, "s.tif", move=False)

    response = api.post(
        f"/api/sessions/{record.session_id}/contract",
        json={"query": "Highlight the water body", "force_offline": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "region_grounding"
    assert body["claim"].endswith(".")
    assert body["target_classes"] == ["water"]
    assert body["source"] == "offline-rule-router"
    assert body["tools"]
    assert body["confounders"]
    assert body["contract_hash"]


def test_contract_route_refuses_an_incomplete_configuration(api, store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "s.tif", width=96, height=96)
    record = store.create()
    store.ingest(record.session_id, ImageRole.DATE_A, scene.path, "s.tif", move=False)

    response = api.post(
        f"/api/sessions/{record.session_id}/contract",
        json={"query": "What changed?", "force_offline": True},
    )

    assert response.status_code == 409
    assert "not complete" in response.json()["detail"]


def test_contract_route_404s_for_an_unknown_session(api) -> None:
    response = api.post(
        f"/api/sessions/{'0' * 32}/contract",
        json={"query": "Describe the land cover"},
    )
    assert response.status_code == 404


def test_contract_route_rejects_an_empty_query(api, store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "s.tif", width=96, height=96)
    record = store.create()
    store.ingest(record.session_id, ImageRole.SINGLE, scene.path, "s.tif", move=False)

    response = api.post(
        f"/api/sessions/{record.session_id}/contract", json={"query": ""}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# The plan is a property of the task, not of the planner
# ---------------------------------------------------------------------------
#
# These rules were enforced only by the offline rule router, so they held on the
# offline path and lapsed on the model path. Two demos went wrong as a result: an
# extraction acquired a verdict at 5% confidence, and the same claim over the same
# imagery came back with one fewer tool and a different verdict on consecutive
# runs. Both are tested here against a model draft, because that is the path where
# they failed.


def test_an_extraction_cannot_acquire_a_verdict_engine(
    tool_registry, cross_modal_context
) -> None:
    """An extraction has nothing to support or refute.

    The model planned a verdict engine for "identify flooded areas" and the run
    duly produced a verdict at 5% confidence, which is noise wearing the clothes
    of a conclusion. The task's tool list is enforced whoever proposed it.
    """
    draft = ContractDraft(
        claim="Flooded and built-up areas can be identified from both sensors.",
        task_type=ContractTaskType.CROSS_MODAL_EXTRACTION,
        target_classes=["water", "built-up"],
        tools=[
            DraftTool(name="spectral-index-engine", why="computes indices"),
            DraftTool(name="optical-sar-fusion", why="compares the sensors"),
            DraftTool(name="verdict-engine", why="concludes"),
        ],
    )

    contract = validate_draft(
        draft,
        session_id=cross_modal_context.session.session_id,
        query="Use the optical and SAR images together to identify flooded areas",
        registry=tool_registry,
        context=cross_modal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert "verdict-engine" not in {plan.tool for plan in contract.tools}
    rejection = next(
        r for r in contract.rejections if r.value == "verdict-engine"
    )
    assert "not admitted" in rejection.reason
    # Refused, never silently dropped: the contract has to show what it declined.
    assert "cross modal extraction" in rejection.reason


def test_a_tool_the_task_cannot_do_without_is_added_whoever_planned_it(
    tool_registry, bitemporal_context
) -> None:
    """A claim investigation that omits the change engine cannot reach a verdict."""
    draft = ContractDraft(
        claim="Vegetation decreased between the two acquisitions.",
        task_type=ContractTaskType.CLAIM_INVESTIGATION,
        target_classes=["vegetation"],
        change_direction=ChangeDirection.DECREASED,
        tools=[DraftTool(name="spectral-index-engine", why="computes NDVI")],
    )

    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="Did vegetation decrease?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    planned = {plan.tool for plan in contract.tools}
    assert {"change-cva-engine", "gis-measure-engine", "spectral-index-engine"} <= planned
    added = [r for r in contract.repairs if r.field == "tools"]
    assert added, "the additions have to be recorded as repairs, not made silently"
    assert any("not left to the planner" in r.detail for r in added)


def test_an_added_tool_is_configured_and_ordered_as_the_router_would(
    tool_registry, cross_modal_context
) -> None:
    """A tool added bare runs on its defaults, and one appended last runs last.

    Both were real failures. The index engine computed every index instead of the
    one the target needs, and the fusion engine was scheduled ahead of the index
    engine it depends on and skipped itself, so the comparison the scene exists to
    show silently did not happen.
    """
    draft = ContractDraft(
        claim="Flooded areas can be identified from both sensors.",
        task_type=ContractTaskType.CROSS_MODAL_EXTRACTION,
        target_classes=["water"],
        tools=[DraftTool(name="optical-sar-fusion", why="compares the sensors")],
    )

    contract = validate_draft(
        draft,
        session_id=cross_modal_context.session.session_id,
        query="Use both images to identify flooded areas",
        registry=tool_registry,
        context=cross_modal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    order = [plan.tool for plan in sorted(contract.tools, key=lambda p: p.order)]
    assert "spectral-index-engine" in order
    assert order.index("spectral-index-engine") < order.index("optical-sar-fusion")

    index_plan = next(p for p in contract.tools if p.tool == "spectral-index-engine")
    assert index_plan.parameters.get("indices"), (
        "an added tool has to carry the parameters the task implies, or it runs on "
        "defaults that do not match the question"
    )


def test_the_planner_cannot_drop_an_alternative_explanation_the_input_requires(
    tool_registry, bitemporal_context
) -> None:
    """Which explanations get tested is not the planner's choice.

    On consecutive runs of the same question the model proposed four confounders
    and then three, and the shorter list reached a different verdict. A finding
    whose thoroughness varies run to run is not reproducible.
    """
    draft = ContractDraft(
        claim="Vegetation decreased between the two acquisitions.",
        task_type=ContractTaskType.CLAIM_INVESTIGATION,
        target_classes=["vegetation"],
        change_direction=ChangeDirection.DECREASED,
        confounders_to_test=[ConfounderKind.SEASONALITY],
        tools=[DraftTool(name="change-cva-engine", why="measures change")],
    )

    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="Did vegetation decrease?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    kinds = {plan.kind for plan in contract.confounders}
    required = set(
        plan_confounders(
            ContractTaskType.CLAIM_INVESTIGATION,
            InputConfiguration.BI_TEMPORAL_PAIR,
            bitemporal_context,
        )
    )
    assert required <= kinds
    repair = next(
        r for r in contract.repairs if r.field == "confounders_to_test"
    )
    assert "not the planner's choice" in repair.detail


def test_an_explanation_the_planner_added_itself_is_kept(
    tool_registry, bitemporal_context
) -> None:
    """The rule is one-directional: the model may add, it may not subtract."""
    draft = ContractDraft(
        claim="Vegetation decreased between the two acquisitions.",
        task_type=ContractTaskType.CLAIM_INVESTIGATION,
        target_classes=["vegetation"],
        change_direction=ChangeDirection.DECREASED,
        confounders_to_test=[
            ConfounderKind.SEASONALITY,
            ConfounderKind.ILLUMINATION_TERRAIN,
        ],
        tools=[DraftTool(name="change-cva-engine", why="measures change")],
    )

    contract = validate_draft(
        draft,
        session_id=bitemporal_context.session.session_id,
        query="Did vegetation decrease?",
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    kinds = {plan.kind for plan in contract.confounders}
    assert ConfounderKind.ILLUMINATION_TERRAIN in kinds


def test_the_same_question_plans_the_same_investigation_either_way(
    tool_registry, bitemporal_context
) -> None:
    """The coverage of a run must not depend on which path drafted it.

    Wording, parameters and rationales may differ between the model and the rule
    router. What gets measured, what gets challenged, and whether a conclusion is
    reached may not.
    """
    query = "Did vegetation decrease?"
    offline = route_offline(query, bitemporal_context)
    from_router = validate_draft(
        offline,
        session_id=bitemporal_context.session.session_id,
        query=query,
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.OFFLINE_RULE_ROUTER,
        model=None,
    )

    # A deliberately threadbare model draft: one tool, one confounder.
    from_model = validate_draft(
        ContractDraft(
            claim="Vegetation decreased between the two acquisitions.",
            task_type=ContractTaskType.CLAIM_INVESTIGATION,
            target_classes=["vegetation"],
            change_direction=ChangeDirection.DECREASED,
            confounders_to_test=[ConfounderKind.SEASONALITY],
            tools=[DraftTool(name="spectral-index-engine", why="computes NDVI")],
        ),
        session_id=bitemporal_context.session.session_id,
        query=query,
        registry=tool_registry,
        context=bitemporal_context,
        source=ContractSource.GEMINI,
        model="gemini-3.7-flash",
    )

    assert {p.tool for p in from_model.tools} == {p.tool for p in from_router.tools}
    assert {c.kind for c in from_model.confounders} == {
        c.kind for c in from_router.confounders
    }
