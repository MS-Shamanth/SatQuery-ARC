"""The evidence packet.

Two things are worth testing here and the rest is plumbing.

The first is attribution: the report must not print a number it cannot trace.
These tests check that the audit actually fails when a figure is unattributable,
because an audit that cannot fail is decoration.

The second is honesty about absence. A packet that omits the verdict because none
was reached has to say so, and a file that was not written must not be listed as
though it were.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from app.core.packet import (
    ARCHIVE_FILENAME,
    FINDINGS_FILENAME,
    MANIFEST_FILENAME,
    MEASUREMENTS_FILENAME,
    REPORT_FILENAME,
    TRACE_FILENAME,
    FigureLedger,
    build_packet,
    describe_omissions,
)
from app.models.confounders import ConfounderTest, ConfounderVerdict
from app.models.contract import ChangeDirection, ConfounderKind, ContractTaskType
from app.models.packet import PacketFileRole
from app.models.schemas import ImageRole, InputConfiguration, Measurement
from app.models.trace import RunStatus, RunTrace
from app.models.verdict import (
    ConfidenceComponent,
    ConsistencyCheck,
    EvidenceDirection,
    EvidenceItem,
    Verdict,
    VerdictLabel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def measurement(
    key: str = "grounded_area_km2",
    value: float = 9.6547,
    *,
    unit: str = "km2",
    precision: int = 4,
) -> Measurement:
    return Measurement(
        key=key,
        label=key.replace("_", " "),
        value=value,
        unit=unit,
        formula="pixel_count * pixel_area_m2 / 1e6",
        inputs={"pixel_count": 96547},
        source_tool="grounding-engine",
        source_version="1.0.0",
        method="conventional threshold",
        applies_to=[ImageRole.SINGLE],
        precision=precision,
    )


def verdict(
    label: VerdictLabel = VerdictLabel.SUPPORTED, confidence: float = 0.7123
) -> Verdict:
    return Verdict(
        label=label,
        claim="The water body shrank between the two dates.",
        asserted_direction=ChangeDirection.DECREASED,
        measured_direction=ChangeDirection.DECREASED,
        confidence=confidence,
        confidence_components=[
            ConfidenceComponent(
                name="measurement_quality",
                label="How little the answer depends on its threshold",
                measured=0.9912,
                contribution=0.2974,
                weight=0.30,
                best=0.30,
                rationale="Nudging the threshold barely moves the answer.",
                lever="A scene where the classes separate more cleanly.",
            ),
        ],
        evidence=[
            EvidenceItem(
                measurement_key="grounded_area_km2",
                label="grounded area",
                value=9.6547,
                unit="km2",
                display="9.6547 km2",
                direction=EvidenceDirection.SUPPORTS,
                relevance="It is the quantity the claim is about.",
                statement="The region covers 9.6547 km2.",
                source_tool="grounding-engine",
                source_version="1.0.0",
                formula="pixel_count * pixel_area_m2 / 1e6",
            )
        ],
        reasoning="The measured area is 9.6547 km2, which supports the claim.",
        what_would_change_it=["An acquisition from the same season."],
    )


def confounder(
    kind: ConfounderKind = ConfounderKind.SEASONALITY,
    result: ConfounderVerdict = ConfounderVerdict.RULED_OUT,
) -> ConfounderTest:
    return ConfounderTest(
        kind=kind,
        label="Seasonal phenology rather than land-cover change",
        question="Could the growing season explain this?",
        verdict=result,
        measured="0.031 of the scene shifted",
        measured_numeric=0.031,
        threshold="above 0.500 of the scene would be suspicious",
        formula="changed_pixels / observable_pixels",
        method="inside-versus-outside index shift",
        explanation="The change is localised, so phenology does not account for it.",
        requirement=(
            None
            if result is ConfounderVerdict.RULED_OUT
            else "An acquisition from the same part of the growing season."
        ),
    )


def make_trace(**overrides) -> RunTrace:
    payload: dict = dict(
        run_id="b" * 32,
        session_id="c" * 32,
        contract_hash="d" * 32,
        query="Did the reservoir shrink?",
        claim="The water body shrank between the two dates.",
        task_type=ContractTaskType.CLAIM_INVESTIGATION,
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        status=RunStatus.COMPLETED,
        measurements=[measurement()],
        confounders=[confounder()],
        verdict=verdict(),
    )
    payload.update(overrides)
    return RunTrace(**payload)


@pytest.fixture
def packet(tmp_path):
    return build_packet(make_trace(), {}, tmp_path)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_a_registered_figure_is_traced():
    ledger = FigureLedger()
    ledger.measured(measurement(), "measurements table")
    audit = ledger.audit("The area is 9.6547 km2.", {})
    assert audit.passed
    assert audit.traced == 1


def test_a_figure_nobody_registered_is_reported_as_untraceable():
    """The audit has to be able to fail, or it proves nothing."""
    ledger = FigureLedger()
    ledger.measured(measurement(), "measurements table")
    audit = ledger.audit("The area is 9.6547 km2, up from 4.3129 km2.", {})
    assert not audit.passed
    assert len(audit.untraceable) == 1
    # The report says where the figure was, not only what it was: a bare "4.3129"
    # is not enough to find it in a document.
    assert audit.untraceable[0].startswith("4.3129")
    assert "up from" in audit.untraceable[0]
    assert "4.3129" in audit.note


def test_a_measured_value_quoted_in_prose_is_traced_without_being_registered():
    """The report quotes the analysis's own sentences, which carry its figures."""
    ledger = FigureLedger()
    audit = ledger.audit(
        "The measured area is 9.6547 km2.", {"grounded_area_km2": measurement()}
    )
    assert audit.passed


def test_a_rounded_form_of_a_measured_value_is_traced():
    ledger = FigureLedger()
    audit = ledger.audit(
        "roughly 9.65 km2", {"grounded_area_km2": measurement()}
    )
    assert audit.passed


def test_precision_asserted_is_precision_required():
    """A figure claiming four decimals must match to four decimals."""
    ledger = FigureLedger()
    audit = ledger.audit(
        "exactly 9.6544 km2", {"grounded_area_km2": measurement()}
    )
    assert not audit.passed


def test_a_declared_threshold_is_attributed_to_its_source():
    ledger = FigureLedger()
    ledger.declared("0.500", "confounder: seasonality", "threshold declared in advance")
    audit = ledger.audit("compared against 0.500", {})
    assert audit.passed
    figure = ledger.figures[0]
    assert not figure.traced_to_measurement
    assert figure.source == "threshold declared in advance"


def test_a_measured_figure_carries_the_formula_behind_it():
    ledger = FigureLedger()
    ledger.measured(measurement(), "measurements table")
    figure = ledger.figures[0]
    assert figure.traced_to_measurement
    assert figure.measurement_key == "grounded_area_km2"
    assert figure.formula == "pixel_count * pixel_area_m2 / 1e6"
    assert figure.source_tool == "grounding-engine"
    assert figure.source_version == "1.0.0"


def test_the_report_attributes_every_figure_it_prints(packet):
    assert packet.audit.passed, packet.audit.untraceable
    assert packet.audit.checked > 0
    assert packet.audit.traced == packet.audit.checked


def test_every_figure_in_the_manifest_names_a_source(packet):
    for figure in packet.figures:
        assert figure.where
        assert figure.traced_to_measurement or figure.source, figure.display


# ---------------------------------------------------------------------------
# What gets written
# ---------------------------------------------------------------------------


def test_the_packet_contains_a_report_a_trace_and_the_measurements(packet, tmp_path):
    names = {item.filename for item in packet.files}
    assert {REPORT_FILENAME, TRACE_FILENAME, MEASUREMENTS_FILENAME} <= names
    for name in names:
        assert (tmp_path / name).exists()


def test_the_report_is_a_pdf_with_content(packet, tmp_path):
    report = tmp_path / REPORT_FILENAME
    data = report.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 2000


def test_every_file_carries_its_own_checksum_and_size(packet, tmp_path):
    import hashlib

    for item in packet.files:
        data = (tmp_path / item.filename).read_bytes()
        assert item.size_bytes == len(data)
        assert item.sha256 == hashlib.sha256(data).hexdigest()


def test_the_measurements_csv_carries_the_formula_beside_the_value(packet, tmp_path):
    rows = list(csv.DictReader((tmp_path / MEASUREMENTS_FILENAME).read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["key"] == "grounded_area_km2"
    assert rows[0]["formula"] == "pixel_count * pixel_area_m2 / 1e6"
    assert rows[0]["source_tool"] == "grounding-engine"
    # The full precision, not the display rounding: a value in a table is for
    # recomputing with, not for reading.
    assert float(rows[0]["value"]) == pytest.approx(9.6547)


def test_the_trace_json_round_trips(packet, tmp_path):
    restored = RunTrace.model_validate_json(
        (tmp_path / TRACE_FILENAME).read_text(encoding="utf-8")
    )
    assert restored.run_id == "b" * 32
    assert restored.verdict is not None
    assert restored.measurements[0].key == "grounded_area_km2"


def test_the_archive_holds_every_file_plus_a_readme(packet, tmp_path):
    assert packet.archive is not None
    assert packet.archive.filename == ARCHIVE_FILENAME
    with zipfile.ZipFile(tmp_path / ARCHIVE_FILENAME) as archive:
        held = set(archive.namelist())
    assert "README.txt" in held
    assert MANIFEST_FILENAME in held
    for item in packet.files:
        assert item.filename in held


def test_the_readme_names_the_verdict_and_the_checksums(packet, tmp_path):
    with zipfile.ZipFile(tmp_path / ARCHIVE_FILENAME) as archive:
        readme = archive.read("README.txt").decode("utf-8")
    assert "Supported" in readme
    assert packet.files[0].sha256 in readme
    assert "does not contain" in readme


def test_the_manifest_does_not_claim_to_contain_its_own_checksum(packet, tmp_path):
    """A file cannot record its own hash, so the manifest is not listed in itself."""
    assert MANIFEST_FILENAME not in {item.filename for item in packet.files}
    stored = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert MANIFEST_FILENAME not in {item["filename"] for item in stored["files"]}
    # It does name the archive it is inside, which is written after it.
    assert stored["archive"]["filename"] == ARCHIVE_FILENAME


def test_the_manifest_restores_as_the_packet_it_describes(packet, tmp_path):
    from app.models.packet import EvidencePacket

    restored = EvidencePacket.model_validate_json(
        (tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert restored.run_id == packet.run_id
    assert restored.audit.passed
    assert len(restored.figures) == len(packet.figures)


def test_the_packet_repeats_the_verdict_so_it_reads_on_its_own(packet):
    assert packet.verdict_label == "supported"
    assert "Supported" in packet.verdict_text
    assert packet.confidence == pytest.approx(0.7123)


def test_no_findings_file_is_listed_when_no_mask_could_be_vectorised(packet):
    """A file that was not written must not appear as though it were."""
    assert FINDINGS_FILENAME not in {item.filename for item in packet.files}


# ---------------------------------------------------------------------------
# Honesty about absence
# ---------------------------------------------------------------------------


def test_the_packet_says_it_does_not_redistribute_the_imagery(packet):
    assert any("source imagery" in omission for omission in packet.omissions)


def test_a_run_without_a_verdict_says_so_rather_than_implying_one(tmp_path):
    result = build_packet(make_trace(verdict=None), {}, tmp_path)
    assert result.verdict_label is None
    assert result.verdict_text == ""
    assert any("no conclusion is asserted" in item for item in result.omissions)
    assert result.audit.passed, result.audit.untraceable


def test_a_run_that_tested_nothing_says_the_finding_is_unchallenged(tmp_path):
    result = build_packet(make_trace(confounders=[]), {}, tmp_path)
    assert any("no alternative explanation was tested" in item for item in result.omissions)


def test_an_untestable_alternative_explanation_is_named_as_such(tmp_path):
    trace = make_trace(
        confounders=[
            confounder(ConfounderKind.SAR_SPECIFIC, ConfounderVerdict.NOT_TESTED)
        ]
    )
    result = build_packet(trace, {}, tmp_path)
    assert any("sar_specific" in item for item in result.omissions)


def test_a_missing_figure_is_declared_as_missing():
    omissions = describe_omissions(make_trace(), has_figure=False)
    assert any("composed map figure" in item for item in omissions)
    assert not any(
        "composed map figure" in item
        for item in describe_omissions(make_trace(), has_figure=True)
    )


def test_a_report_with_no_measurements_still_writes(tmp_path):
    """An empty run is still a record, and an empty record must not crash."""
    result = build_packet(make_trace(measurements=[], verdict=None), {}, tmp_path)
    assert (tmp_path / REPORT_FILENAME).exists()
    assert result.audit.passed, result.audit.untraceable
    rows = (tmp_path / MEASUREMENTS_FILENAME).read_text().strip().splitlines()
    assert len(rows) == 1  # the header, and nothing claimed beneath it


def test_a_cross_check_that_disagrees_is_reported_as_disagreeing(tmp_path):
    decided = verdict()
    decided.consistency = [
        ConsistencyCheck(
            quantity="gained area",
            label="gain measured twice",
            first_key="gain_km2",
            first_value=1.2500,
            second_key="corroborated_gain_km2",
            second_value=0.4000,
            unit="km2",
            relative_difference=0.68,
            tolerance=0.25,
            agrees=False,
            method="NDBI gain against NDVI loss over the same pixels",
            explanation=(
                "The two estimates disagree, and an average of them is a number "
                "neither method supports."
            ),
        )
    ]
    result = build_packet(make_trace(verdict=decided), {}, tmp_path)
    assert result.audit.passed, result.audit.untraceable
    assert any("cross-check" in figure.where for figure in result.figures)


def test_a_surviving_confounder_puts_its_requirement_in_the_report(tmp_path):
    trace = make_trace(
        confounders=[confounder(result=ConfounderVerdict.LIKELY)],
        verdict=verdict(VerdictLabel.INCONCLUSIVE, 0.4200),
    )
    result = build_packet(trace, {}, tmp_path)
    assert result.audit.passed, result.audit.untraceable
    assert result.verdict_label == "inconclusive"


# ---------------------------------------------------------------------------
# Integration: the packet of a real run
# ---------------------------------------------------------------------------


def test_a_real_run_writes_a_packet_the_trace_names(tmp_path, make_context, tool_registry):
    """The stage has to record the export, or nobody can vouch for the files."""
    from app.core.orchestrator import execute_run
    from app.models.trace import StageId, StageStatus
    from tests.raster_fixtures import make_optical_scene
    from tests.test_orchestrator import build_contract

    scene = make_optical_scene(tmp_path / "single.tif")
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)
    contract = build_contract(context)

    directory = tmp_path / "packet"
    trace, _ = execute_run(
        contract,
        context,
        tool_registry,
        compose=lambda t, o: build_packet(
            t,
            o,
            directory,
            session=context.session,
            readiness=context.readiness,
        ),
    )

    assert trace.packet is not None
    assert trace.packet.audit.passed, trace.packet.audit.untraceable
    stage = trace.stage(StageId.COMPOSE_PACKET)
    assert stage is not None
    assert stage.status is StageStatus.OK
    for item in trace.packet.files:
        assert (directory / item.filename).exists()
    # The report quotes the real readiness gate and real measurements, which is
    # where an attribution bug would actually show up.
    assert any(
        figure.traced_to_measurement for figure in trace.packet.figures
    )
    assert any("readiness" in figure.where for figure in trace.packet.figures)


def test_a_run_with_no_composer_says_no_packet_was_written(
    tmp_path, make_context, tool_registry
):
    from app.core.orchestrator import execute_run
    from app.models.trace import StageId, StageStatus
    from tests.raster_fixtures import make_optical_scene
    from tests.test_orchestrator import build_contract

    scene = make_optical_scene(tmp_path / "single.tif")
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)
    trace, _ = execute_run(build_contract(context), context, tool_registry)

    stage = trace.stage(StageId.COMPOSE_PACKET)
    assert stage is not None
    assert stage.status is StageStatus.NOT_BUILT
    assert stage.unavailable_note
    assert "task" not in stage.unavailable_note.lower()
    assert trace.packet is None


def test_a_composer_that_raises_does_not_fail_the_run(
    tmp_path, make_context, tool_registry
):
    """An export is not load-bearing: the measurements outrank the paperwork."""
    from app.core.orchestrator import execute_run
    from app.models.trace import StageId, StageStatus
    from tests.raster_fixtures import make_optical_scene
    from tests.test_orchestrator import build_contract

    scene = make_optical_scene(tmp_path / "single.tif")
    context = make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)

    def explode(trace, outcomes):
        raise OSError("the disk is full")

    trace, _ = execute_run(
        build_contract(context), context, tool_registry, compose=explode
    )

    assert trace.status is RunStatus.COMPLETED
    stage = trace.stage(StageId.COMPOSE_PACKET)
    assert stage is not None
    assert stage.status is StageStatus.FAILED
    assert "the disk is full" in stage.steps[0].reason
    assert trace.measurements, "the measurements must survive a failed export"


def test_the_overlays_a_run_rendered_are_included_in_its_packet(tmp_path):
    """The map layers and the packet are the same files, not two copies."""
    from app.core.packet import collect_overlay_files

    directory = tmp_path
    (directory / "grounding__water.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    (directory / "grounding__water.geojson").write_text('{"type":"FeatureCollection"}')

    collected = collect_overlay_files(
        directory,
        [
            {
                "kind": "mask",
                "label": "Water",
                "description": "NDWI above 0",
                "png_filename": "grounding__water.png",
                "geojson_filename": "grounding__water.geojson",
                "colour": "#22d3ee",
                "area_km2": 9.6547,
            }
        ],
    )
    assert {item.role for item in collected} == {
        PacketFileRole.OVERLAY,
        PacketFileRole.FINDINGS,
    }
    assert all(item.sha256 for item in collected)


def test_a_layer_whose_file_was_never_written_is_not_listed(tmp_path):
    from app.core.packet import collect_overlay_files

    collected = collect_overlay_files(
        tmp_path,
        [{"kind": "mask", "label": "Water", "png_filename": "absent.png"}],
    )
    assert collected == []
