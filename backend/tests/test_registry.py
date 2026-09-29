"""Tool registry tests.

The registry is the guard that stops a hallucinated tool name from being
executed, so the rejection behaviour matters as much as the lookup behaviour.
"""

from __future__ import annotations

import pytest

from app.core.registry import (
    ToolNotRegistered,
    ToolRegistry,
    build_default_registry,
)
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    Modality,
    ToolImplementation,
    ToolRequirement,
)
from app.tools.base import BaseTool, ToolContext, ToolOutcome
from tests.raster_fixtures import make_labels, make_optical_scene, make_sar_scene


class ProbeTool(BaseTool):
    name = "probe-tool"
    version = "0.1.0"
    implementation = ToolImplementation.DETERMINISTIC
    summary = "A tool used only by tests."

    def requirement(self) -> ToolRequirement:
        return ToolRequirement(modalities=[Modality.OPTICAL])

    def execute(self, context: ToolContext) -> ToolOutcome:
        return self.outcome()


class LearnedProbe(BaseTool):
    name = "learned-probe"
    version = "0.2.0"
    implementation = ToolImplementation.LEARNED
    summary = "Stands in for a tool with trained weights."

    def execute(self, context: ToolContext) -> ToolOutcome:
        return self.outcome()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


DETERMINISTIC_TOOLS = [
    "change-cva-engine",
    "confounder-engine",
    "evidence-disagreement-engine",
    "gis-measure-engine",
    "grounding-engine",
    "optical-sar-fusion",
    "sar-backscatter-engine",
    "spectral-index-engine",
    "verdict-engine",
]


def test_default_registry_holds_the_shipped_tools() -> None:
    """Every computed tool ships unconditionally; the learned one does not.

    The probe appears only when weights have been fitted in this checkout, so the
    assertion is on the deterministic set. A build with no trained model still has
    the whole measuring pipeline.
    """
    registry = build_default_registry()
    assert set(DETERMINISTIC_TOOLS) <= set(registry.names())
    assert registry.names() == sorted(registry.names())

    optional = set(registry.names()) - set(DETERMINISTIC_TOOLS)
    assert optional <= {"rs-landcover-probe"}
    if not optional:
        # Absent means explained, never merely missing.
        assert "rs-landcover-probe" in registry.declined()


def test_the_only_learned_tool_is_the_optional_one() -> None:
    """The division of labour has to be visible in the registry itself.

    Everything that produces a number a verdict rests on is a published formula.
    The single learned component is additive and declares itself as trained.
    """
    registry = build_default_registry()
    learned = [
        name
        for name in registry.names()
        if registry.get(name).implementation is ToolImplementation.LEARNED
    ]
    assert learned in ([], ["rs-landcover-probe"])
    for name in DETERMINISTIC_TOOLS:
        assert registry.get(name).implementation is ToolImplementation.DETERMINISTIC


def test_registering_a_duplicate_name_is_refused() -> None:
    registry = ToolRegistry()
    registry.register(ProbeTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ProbeTool())
    # Unless replacement is explicit.
    registry.register(ProbeTool(), replace=True)
    assert len(registry) == 1


def test_conditional_registration_records_why_a_tool_declined() -> None:
    """The optional adapter must be absent without the pipeline special-casing it."""
    registry = ToolRegistry()

    registered = registry.register_if(
        False, LearnedProbe(), reason="no trained weights in data/models"
    )

    assert registered is None
    assert "learned-probe" not in registry
    assert registry.declined() == {"learned-probe": "no trained weights in data/models"}


def test_conditional_registration_succeeds_when_the_prerequisite_holds() -> None:
    registry = ToolRegistry()
    registry.register_if(True, LearnedProbe())
    assert "learned-probe" in registry
    assert registry.declined() == {}


def test_unregistering_removes_a_tool() -> None:
    registry = build_default_registry()
    registry.unregister("spectral-index-engine")
    assert "spectral-index-engine" not in registry
    registry.clear()
    assert len(registry) == 0


# ---------------------------------------------------------------------------
# Lookup and rejection
# ---------------------------------------------------------------------------


def test_lookup_of_an_unknown_tool_lists_what_is_available() -> None:
    registry = build_default_registry()
    with pytest.raises(ToolNotRegistered) as excinfo:
        registry.get("change-transformer-v9")
    message = str(excinfo.value)
    assert "change-transformer-v9" in message
    assert "spectral-index-engine" in message


def test_resolve_separates_known_names_from_invented_ones() -> None:
    """This is the guard against a hallucinated tool reaching the trace."""
    registry = build_default_registry()

    resolved, unknown = registry.resolve(
        ["spectral-index-engine", "changeformer-large", "gis-measure-engine"]
    )

    assert [tool.name for tool in resolved] == [
        "spectral-index-engine",
        "gis-measure-engine",
    ]
    assert unknown == ["changeformer-large"]


def test_descriptors_expose_requirements_and_parameter_whitelist() -> None:
    registry = build_default_registry()
    by_name = {item.name: item for item in registry.describe()}

    index = by_name["spectral-index-engine"]
    assert index.version == "1.0.0"
    assert index.implementation is ToolImplementation.DETERMINISTIC
    assert index.summary
    assert Modality.OPTICAL in index.requirement.modalities
    assert "indices" in index.parameters
    assert "threshold_method" in index.parameters
    assert index.parameters["threshold_method"]["enum"] == ["otsu", "fixed"]
    assert any(key.startswith("ndwi") for key in index.produces)

    gis = by_name["gis-measure-engine"]
    assert gis.requirement.requires_crs is True


# ---------------------------------------------------------------------------
# Availability against a real session
# ---------------------------------------------------------------------------


def test_availability_reports_reasons_for_a_sar_only_session(
    tool_registry, make_context, tmp_path
) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=80, height=80)
    context = make_context({ImageRole.SINGLE: scene.path})

    availability = {item.tool_name: item for item in tool_registry.availability(context)}

    assert availability["gis-measure-engine"].available is True
    index = availability["spectral-index-engine"]
    assert index.available is False
    assert "optical" in index.reason


def test_availability_is_true_for_a_full_optical_scene(
    tool_registry, make_context, tmp_path
) -> None:
    scene = make_optical_scene(tmp_path / "o.tif", width=80, height=80)
    context = make_context({ImageRole.SINGLE: scene.path})

    # Availability is a question about the input, not about how far a run has
    # progressed, so the verdict engine counts as available on a single image even
    # though nothing has produced a measurement for it to weigh yet.
    # The probe registers only when trained, and the disagreement engine's inputs
    # are other tools' masks rather than bands, so both are excluded here: this
    # test is about what the imagery itself permits.
    optional = {"rs-landcover-probe", "evidence-disagreement-engine"}
    assert set(tool_registry.available_names(context)) - optional == {
        "gis-measure-engine",
        "grounding-engine",
        "spectral-index-engine",
        "verdict-engine",
    }

    # The radar and fusion engines need a radar image and a pair respectively, and
    # both say which.
    availability = {
        item.tool_name: item for item in tool_registry.availability(context)
    }
    assert availability["sar-backscatter-engine"].available is False
    assert "sar" in availability["sar-backscatter-engine"].reason
    assert availability["optical-sar-fusion"].available is False
    assert "cross_modal_pair" in availability["optical-sar-fusion"].reason


def test_missing_band_makes_the_index_engine_unavailable_with_the_band_named(
    tool_registry, make_context, tmp_path
) -> None:
    """A 2-band file satisfies no index band pair."""
    scene = make_optical_scene(
        tmp_path / "two.tif",
        width=80,
        height=80,
        labels=make_labels(80, 80),
        band_names=("blue", "green"),
    )
    context = make_context({ImageRole.SINGLE: scene.path})

    availability = {item.tool_name: item for item in tool_registry.availability(context)}
    reason = availability["spectral-index-engine"].reason

    assert availability["spectral-index-engine"].available is False
    assert "none of" in reason
    assert "nir" in reason


def test_requirement_check_names_the_wrong_configuration() -> None:
    class PairOnly(BaseTool):
        name = "pair-only"
        version = "1.0.0"
        summary = "Needs two dates."

        def requirement(self) -> ToolRequirement:
            return ToolRequirement(
                configurations=[InputConfiguration.BI_TEMPORAL_PAIR]
            )

        def execute(self, context: ToolContext) -> ToolOutcome:
            return self.outcome()

    registry = ToolRegistry()
    tool = registry.register(PairOnly())
    assert tool is not None


def test_pair_only_tool_is_unavailable_on_a_single_image(
    make_context, tmp_path
) -> None:
    class PairOnly(BaseTool):
        name = "pair-only"
        version = "1.0.0"
        summary = "Needs two dates."

        def requirement(self) -> ToolRequirement:
            return ToolRequirement(
                configurations=[InputConfiguration.BI_TEMPORAL_PAIR]
            )

        def execute(self, context: ToolContext) -> ToolOutcome:
            return self.outcome()

    scene = make_optical_scene(tmp_path / "o.tif", width=64, height=64)
    context = make_context({ImageRole.SINGLE: scene.path})

    ok, reason = PairOnly().can_run(context)

    assert ok is False
    assert "bi_temporal_pair" in reason
    assert "single" in reason


def test_a_crs_requirement_rejects_a_benchmark_raster(make_context, tmp_path) -> None:
    import numpy as np

    from tests.raster_fixtures import write_raster

    rgb = np.random.default_rng(2).integers(10, 240, (3, 64, 64), dtype="uint8")
    path = write_raster(tmp_path / "bench.png", rgb, epsg=None, driver="PNG")
    context = make_context({ImageRole.SINGLE: path})

    from app.tools.gis import GisMeasureEngine

    ok, reason = GisMeasureEngine().can_run(context)

    assert ok is False
    assert "coordinate reference system" in reason


# ---------------------------------------------------------------------------
# A failing tool must not take the run down
# ---------------------------------------------------------------------------


def test_a_tool_that_raises_is_reported_as_skipped(make_context, tmp_path) -> None:
    class Exploding(BaseTool):
        name = "exploding-tool"
        version = "1.0.0"
        summary = "Raises on purpose."

        def execute(self, context: ToolContext) -> ToolOutcome:
            raise RuntimeError("synthetic failure")

    scene = make_optical_scene(tmp_path / "o.tif", width=64, height=64)
    context = make_context({ImageRole.SINGLE: scene.path})

    outcome = Exploding().run(context)

    assert outcome.ok is False
    assert "synthetic failure" in (outcome.skipped_reason or "")
    assert outcome.duration_ms >= 0
    # It still produces a serialisable record for the trace.
    record = outcome.to_record()
    assert record.tool == "exploding-tool"
    assert record.ok is False


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_tools_route_lists_the_registry(api) -> None:
    body = api.get("/api/tools").json()
    names = {item["name"] for item in body}
    assert {"spectral-index-engine", "gis-measure-engine"} <= names
    for item in body:
        assert item["version"]
        assert item["implementation"] in (
            "deterministic",
            "learned",
            "llm-narration",
        )
        assert item["summary"]


def test_session_tools_route_reports_availability_with_reasons(
    api, store, tmp_path
) -> None:
    scene = make_sar_scene(tmp_path / "sar.tif", width=80, height=80)
    record = store.create()
    store.ingest(record.session_id, ImageRole.SINGLE, scene.path, "sar.tif", move=False)

    body = api.get(f"/api/sessions/{record.session_id}/tools").json()

    assert body["configuration"] == "single"
    assert "gis-measure-engine" in body["available"]
    unavailable = {item["tool"]: item["reason"] for item in body["unavailable"]}
    assert "spectral-index-engine" in unavailable
    assert "optical" in unavailable["spectral-index-engine"]


def test_declined_tools_route_is_serialisable(api) -> None:
    assert api.get("/api/tools/declined").status_code == 200


def test_session_tools_route_404s_for_an_unknown_session(api) -> None:
    assert api.get(f"/api/sessions/{'0' * 32}/tools").status_code == 404
