"""Assembling the evidence packet.

The packet is the run turned into files that outlive the browser tab: a PDF that
reads on its own, the trace as JSON, the measurements as CSV, the footprints as
GeoJSON, a composed map figure, and a zip of all of it.

The rule that shapes the code is that the report may not print a number the
packet cannot attribute. Every figure goes through :class:`FigureLedger`, which
records the measurement or the declared source behind it, and the finished text is
then audited against that ledger plus the same value-matching used on the phrased
explanation. A report that fails says so on its own front page rather than being
quietly shipped, because an unattributable figure in an exported document is
worse than one on a screen: the screen has the trace next to it.

Writing the packet is not load-bearing. Every writer here is wrapped by the
caller, and a missing PDF costs the run nothing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from app.models.confounders import VERDICT_LABEL as CONFOUNDER_VERDICT_LABEL
from app.models.packet import (
    ROLE_LABEL,
    EvidencePacket,
    PacketAudit,
    PacketFigure,
    PacketFile,
    PacketFileRole,
)
from app.models.schemas import (
    CheckStatus,
    Measurement,
    ReadinessReport,
    SessionRecord,
)
from app.models.trace import RunTrace
from app.models.verdict import VERDICT_LABEL_TEXT, VerdictLabel

logger = logging.getLogger(__name__)

REPORT_FILENAME = "report.pdf"
TRACE_FILENAME = "trace.json"
MEASUREMENTS_FILENAME = "measurements.csv"
FINDINGS_FILENAME = "findings.geojson"
FIGURE_FILENAME = "map-figure.png"
MANIFEST_FILENAME = "packet.json"
ARCHIVE_FILENAME = "satquery-evidence.zip"

# The composed figure is a reading aid, not a data product; the GeoTIFF-derived
# overlays beside it are the record. Capping it keeps the PDF openable by email.
MAX_FIGURE_EDGE = 1200
# Overlays drawn onto the figure, on top of the base imagery. Beyond a handful the
# figure stops being legible and the separate overlay files serve better.
MAX_FIGURE_MASKS = 4

# Numeric tokens that are not claims about the world: they are part of how a
# figure is written, not a measured quantity.
AUDIT_IGNORE = frozenset({"0", "1", "2", "100"})

A4_USABLE_MM = 174.0


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


class FigureLedger:
    """Every number the report prints, with where it came from.

    Two ways in. ``measured`` attributes a figure to a tool's measurement, which
    carries the formula. ``declared`` attributes one to something that is
    traceable but not measured: a threshold the analysis committed to in advance,
    a count of records, or a value read from the raster's own header. Both appear
    in the packet manifest, so a reader can check any figure in the report against
    its source without taking the report's word for it.
    """

    def __init__(self) -> None:
        self.figures: list[PacketFigure] = []
        self._allowed: set[str] = set()

    def measured(
        self,
        measurement: Measurement,
        where: str,
        *,
        display: str | None = None,
    ) -> str:
        text = display if display is not None else measurement.render()
        return self._register(
            PacketFigure(
                display=text,
                where=where,
                measurement_key=measurement.key,
                source_tool=measurement.source_tool,
                source_version=measurement.source_version,
                formula=measurement.formula,
            )
        )

    def declared(self, display: str, where: str, source: str) -> str:
        return self._register(
            PacketFigure(display=display, where=where, source=source)
        )

    def quoted(self, text: str, where: str, *, source: str) -> str:
        """A sentence the report repeats rather than composes.

        The question as the user asked it, and the explanations the analysis wrote,
        carry their own figures. Those figures were already held to the ledger
        where they were produced, so quoting the sentence verbatim is more
        faithful than paraphrasing it. What matters is that the manifest says the
        report is quoting rather than asserting.
        """
        return self.definition(text, where, source=source)

    def definition(self, text: str, where: str, *, source: str) -> str:
        """A constant that is part of a formula rather than a result.

        ``/ 1e6`` in a published area formula is not a claim about the imagery, it
        is the definition of the unit. Attributing it to the formula it belongs to
        keeps the audit total without pretending the constant was measured.
        """
        if not _numbers_in(text):
            return text
        return self._register(
            PacketFigure(display=text, where=where, source=source)
        )

    def _register(self, figure: PacketFigure) -> str:
        self.figures.append(figure)
        for token in _numbers_in(figure.display):
            self._allowed.add(token)
            # A figure written as 0.91 also licences 0.91 appearing without its
            # trailing zero, and vice versa, since both are the same number.
            self._allowed.add(_canonical(token))
        return figure.display

    def audit(self, text: str, measurements: dict[str, Measurement]) -> PacketAudit:
        """Check every number in the report against the ledger.

        A token passes if the report registered it, or if it matches a measured
        value under the same tolerance rule the phrased explanation is held to.
        The second route exists because the report quotes the analysis's own
        sentences, and those sentences already carry audited figures.
        """
        from app.tools.verdict import admissible_figures, figure_matches, tolerance_for

        audit = PacketAudit()
        known = admissible_figures(measurements)

        for line in text.splitlines():
            for token in _numbers_in(line):
                if token.lstrip("-") in AUDIT_IGNORE:
                    continue
                audit.checked += 1
                if token in self._allowed or _canonical(token) in self._allowed:
                    audit.traced += 1
                    continue
                try:
                    value = float(token)
                except ValueError:  # pragma: no cover - the pattern matches numbers
                    continue
                if figure_matches(value, known, tolerance_for(token)):
                    audit.traced += 1
                else:
                    # The line is carried with the token: a bare "16" in a failure
                    # report is not enough to find where it came from.
                    audit.untraceable.append(f"{token} (in \u201c{line[:70].strip()}\u201d)")

        audit.passed = not audit.untraceable
        audit.note = (
            f"All {audit.traced} figure(s) in this report are attributed to a "
            "measurement, a declared threshold, or the imagery's own metadata."
            if audit.passed
            else (
                "These figures could not be attributed and should not be relied "
                f"on: {', '.join(audit.untraceable[:6])}."
            )
        )
        return audit


def _numbers_in(text: str) -> list[str]:
    from app.tools.verdict import NUMBER_PATTERN

    return NUMBER_PATTERN.findall(text)


def _canonical(token: str) -> str:
    """A number's shortest exact spelling, so 0.910 and 0.91 compare equal."""
    try:
        return repr(float(token))
    except ValueError:  # pragma: no cover
        return token


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def _describe(
    path: Path,
    role: PacketFileRole,
    *,
    description: str,
    media_type: str,
    label: str | None = None,
) -> PacketFile:
    data = path.read_bytes()
    return PacketFile(
        filename=path.name,
        role=role,
        label=label or ROLE_LABEL[role],
        description=description,
        media_type=media_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def write_trace_json(trace: RunTrace, directory: Path) -> PacketFile:
    """The whole trace, unabridged.

    The PDF is a reading of the run; this is the run. Anything the report
    summarises can be checked against it, including the stages and steps the
    report does not have room for.
    """
    path = directory / TRACE_FILENAME
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return _describe(
        path,
        PacketFileRole.TRACE,
        description=(
            "Every stage, step, tool, parameter and measurement of this run, "
            "exactly as the system recorded it."
        ),
        media_type="application/json",
    )


MEASUREMENT_COLUMNS = (
    "key",
    "label",
    "value",
    "unit",
    "precision",
    "formula",
    "method",
    "source_tool",
    "source_version",
    "applies_to",
    "inputs",
)


def write_measurements_csv(trace: RunTrace, directory: Path) -> PacketFile:
    """Every measurement as a row, with the formula that produced it.

    A spreadsheet is where a reviewer will actually want these. The formula and
    the tool version travel in the same row as the value, so a figure quoted from
    this file cannot be separated from how it was obtained.
    """
    path = directory / MEASUREMENTS_FILENAME
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MEASUREMENT_COLUMNS)
        for item in trace.measurements:
            writer.writerow(
                [
                    item.key,
                    item.label,
                    repr(item.value),
                    item.unit,
                    item.precision,
                    item.formula,
                    item.method or "",
                    item.source_tool,
                    item.source_version,
                    " ".join(role.value for role in item.applies_to),
                    json.dumps(item.inputs, default=str, sort_keys=True),
                ]
            )
    return _describe(
        path,
        PacketFileRole.MEASUREMENTS,
        description=(
            f"{len(trace.measurements)} measurement(s), each with its formula, "
            "method, and the tool and version that produced it."
        ),
        media_type="text/csv",
    )


def write_findings_geojson(
    trace: RunTrace,
    outcomes: dict[str, Any],
    directory: Path,
) -> PacketFile | None:
    """Every mask footprint in one FeatureCollection, in EPSG:4326.

    The per-layer GeoJSON files are already written for the map. This combines
    them so the whole finding opens as a single layer in QGIS, with each feature
    carrying the tool and mask it came from.
    """
    from app.tools.gis import mask_to_geojson

    features: list[dict] = []
    for tool, outcome in outcomes.items():
        if not getattr(outcome, "ok", False):
            continue
        for layer in getattr(outcome, "masks", []):
            if layer.crs is None:
                continue
            try:
                collection = mask_to_geojson(
                    layer.array,
                    layer.transform,
                    layer.crs,
                    properties={
                        "layer": layer.key,
                        "label": layer.label,
                        "tool": tool,
                        "run_id": trace.run_id,
                        "threshold": layer.threshold,
                        "threshold_method": layer.threshold_method,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - one bad mask is not fatal
                logger.warning("packet vector export failed for %s: %s", layer.key, exc)
                continue
            features.extend(collection.get("features", []))

    if not features:
        return None

    # Feature ids are per-collection, so they repeat once the collections are
    # merged. Renumbering keeps them unique across the packet.
    for index, feature in enumerate(features):
        feature["id"] = index

    path = directory / FINDINGS_FILENAME
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return _describe(
        path,
        PacketFileRole.FINDINGS,
        description=(
            f"{len(features)} polygon(s) in EPSG:4326, each tagged with the mask "
            "and tool it came from and the threshold that defined it."
        ),
        media_type="application/geo+json",
    )


@dataclass(frozen=True)
class FigureEntry:
    """One layer drawn on the composed figure, for the legend beside it."""

    label: str
    colour: str
    area_km2: float | None


def compose_figure(
    directory: Path, layers: Sequence[Any]
) -> tuple[PacketFile, list[FigureEntry]] | None:
    """Flatten the base imagery and its overlays into one PNG for the report.

    The overlays are already reprojected to EPSG:4326 on a common grid, so
    stacking them is an alpha composite rather than a second reprojection. Each
    drawn layer is returned so the legend states what the colours mean instead of
    leaving the reader to guess.
    """
    from PIL import Image

    base = next((item for item in layers if _layer_kind(item) == "base"), None)
    if base is None:
        return None

    base_path = directory / _layer_field(base, "png_filename")
    if not base_path.exists():
        return None

    canvas = Image.open(base_path).convert("RGBA")
    if max(canvas.size) > MAX_FIGURE_EDGE:
        scale = MAX_FIGURE_EDGE / max(canvas.size)
        canvas = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            Image.LANCZOS,
        )

    drawn: list[FigureEntry] = []
    for item in layers:
        if _layer_kind(item) != "mask" or len(drawn) >= MAX_FIGURE_MASKS:
            continue
        overlay_path = directory / _layer_field(item, "png_filename")
        if not overlay_path.exists():
            continue
        try:
            overlay = Image.open(overlay_path).convert("RGBA")
            if overlay.size != canvas.size:
                overlay = overlay.resize(canvas.size, Image.NEAREST)
            canvas = Image.alpha_composite(canvas, overlay)
        except Exception as exc:  # noqa: BLE001 - a figure is not load-bearing
            logger.warning("figure composite failed for %s: %s", overlay_path.name, exc)
            continue
        drawn.append(
            FigureEntry(
                label=_layer_field(item, "label"),
                colour=_layer_field(item, "colour") or "#22d3ee",
                area_km2=_layer_field(item, "area_km2"),
            )
        )

    path = directory / FIGURE_FILENAME
    canvas.convert("RGB").save(path, format="PNG", optimize=True)
    described = _describe(
        path,
        PacketFileRole.FIGURE,
        description=(
            "The scene with the detected regions drawn over it, in EPSG:4326. "
            "A reading aid: the overlay and vector files are the record."
        ),
        media_type="image/png",
    )
    return described, drawn


def _layer_kind(layer: Any) -> str:
    return str(_layer_field(layer, "kind") or "")


def _layer_field(layer: Any, field: str) -> Any:
    if isinstance(layer, dict):
        return layer.get(field)
    return getattr(layer, field, None)


def collect_overlay_files(directory: Path, layers: Sequence[Any]) -> list[PacketFile]:
    """The overlay PNGs and per-layer vectors the map already wrote."""
    collected: list[PacketFile] = []
    for item in layers:
        for field, role, media, what in (
            ("png_filename", PacketFileRole.OVERLAY, "image/png", "overlay"),
            (
                "geojson_filename",
                PacketFileRole.FINDINGS,
                "application/geo+json",
                "footprint",
            ),
        ):
            name = _layer_field(item, field)
            if not name:
                continue
            path = directory / str(name)
            if not path.exists():
                continue
            label = _layer_field(item, "label") or str(name)
            collected.append(
                _describe(
                    path,
                    role,
                    label=f"{label} ({what})",
                    description=str(_layer_field(item, "description") or ""),
                    media_type=media,
                )
            )
    return collected


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

_VERDICT_COLOUR: dict[VerdictLabel, str] = {
    VerdictLabel.SUPPORTED: "#047857",
    VerdictLabel.REFUTED: "#be123c",
    VerdictLabel.INCONCLUSIVE: "#b45309",
    VerdictLabel.UNANSWERABLE: "#475569",
}

_CHECK_COLOUR: dict[CheckStatus, str] = {
    CheckStatus.PASS: "#047857",
    CheckStatus.WARN: "#b45309",
    CheckStatus.FAIL: "#be123c",
    CheckStatus.NOT_APPLICABLE: "#64748b",
}


class ReportBuilder:
    """Builds the PDF, registering every figure it prints as it goes."""

    def __init__(
        self,
        trace: RunTrace,
        *,
        session: SessionRecord | None,
        readiness: ReadinessReport | None,
        ledger: FigureLedger,
        figure: tuple[Path, list[FigureEntry]] | None = None,
        omissions: Sequence[str] = (),
    ) -> None:
        self.trace = trace
        self.session = session
        self.readiness = readiness
        self.ledger = ledger
        self.figure = figure
        self.omissions = list(omissions)
        self.measurements = {item.key: item for item in trace.measurements}
        self.story: list[Any] = []
        # Everything written, for the audit. Kept as text rather than re-parsing
        # the finished PDF, because the audit is about what the report claims and
        # not about how reportlab lays it out.
        self._audited: list[str] = []
        self._styles = _build_styles()

    # -- text helpers ------------------------------------------------------
    def _para(self, text: str, style: str = "body") -> None:
        from reportlab.platypus import Paragraph

        self._audited.append(text)
        self.story.append(Paragraph(escape(text), self._styles[style]))

    def _prose(self, text: str, where: str, *, source: str, style: str = "body") -> None:
        """A paragraph the analysis wrote, registered as quoted rather than asserted."""
        self._para(self.ledger.quoted(text, where, source=source), style)

    def _raw(self, markup: str, style: str = "body", *, audit: str = "") -> None:
        """A paragraph with inline markup already escaped by the caller."""
        from reportlab.platypus import Paragraph

        self._audited.append(audit or markup)
        self.story.append(Paragraph(markup, self._styles[style]))

    def _space(self, height: float = 4.0) -> None:
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        self.story.append(Spacer(1, height * mm))

    def _heading(self, text: str) -> None:
        self._space(5)
        self._para(text, "heading")

    def _table(
        self,
        header: Sequence[str],
        rows: Sequence[Sequence[str]],
        widths: Sequence[float],
        *,
        colours: dict[int, str] | None = None,
    ) -> None:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Table, TableStyle

        for row in rows:
            self._audited.extend(str(cell) for cell in row)
        # A column heading is the report's own wording, so any digits in it are
        # part of the label rather than a figure about the imagery.
        for cell in header:
            self._audited.append(
                self.ledger.definition(
                    str(cell), "table heading", source="column heading"
                )
            )

        data = [
            [Paragraph(escape(str(cell)), self._styles["th"]) for cell in header]
        ]
        for row in rows:
            data.append(
                [Paragraph(escape(str(cell)), self._styles["td"]) for cell in row]
            )

        table = Table(
            data,
            colWidths=[width * mm for width in widths],
            hAlign="LEFT",
            repeatRows=1,
        )
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for index in range(1, len(data)):
            if index % 2 == 0:
                commands.append(
                    ("BACKGROUND", (0, index), (-1, index), colors.HexColor("#f1f5f9"))
                )
        for row_index, colour in (colours or {}).items():
            commands.append(
                (
                    "TEXTCOLOR",
                    (0, row_index + 1),
                    (0, row_index + 1),
                    colors.HexColor(colour),
                )
            )
        table.setStyle(TableStyle(commands))
        self.story.append(table)

    # -- sections ----------------------------------------------------------
    def build(self) -> None:
        self._front_matter()
        self._verdict_section()
        self._confidence_section()
        self._confounder_section()
        self._figure_section()
        self._measurement_section()
        self._consistency_section()
        self._input_section()
        self._readiness_section()
        self._pipeline_section()
        self._omissions_section()

    def audit(self) -> PacketAudit:
        """Audit what has been written so far."""
        return self.ledger.audit("\n".join(self._audited), self.measurements)

    def traceability(self, audit: PacketAudit) -> None:
        """Print the audit inside the document it audits.

        A report that cannot attribute one of its own figures says so where the
        figure is read, not in a log nobody opens.
        """
        self._space(5)
        self._para("Traceability", "heading")
        self._para(audit.note, "small")
        self._para(
            "Every figure above is listed in "
            f"{MANIFEST_FILENAME} with the measurement or declared source behind it.",
            "small",
        )

    def _front_matter(self) -> None:
        trace = self.trace
        self._para("SatQuery evidence packet", "title")
        self._para(
            "The record of one analysis: what was asked, what was measured, what "
            "was concluded, and what could still overturn it.",
            "lede",
        )
        self._space(3)

        generated = datetime.now(timezone.utc)
        asked = "the question as it was asked"
        rows = [
            ["Question", self.ledger.quoted(trace.query, "front matter", source=asked)],
            [
                "Claim under test",
                self.ledger.quoted(
                    trace.claim,
                    "front matter",
                    source="the claim derived from the question, as approved",
                ),
            ],
            ["Task", trace.task_type.value.replace("_", " ")],
            ["Input configuration", trace.configuration.value.replace("_", " ")],
            [
                "Run",
                self.ledger.declared(
                    trace.run_id, "front matter", "run identifier"
                ),
            ],
            [
                "Contract",
                self.ledger.declared(
                    trace.contract_hash[:16], "front matter", "contract hash"
                ),
            ],
            [
                "Started",
                self.ledger.declared(
                    _stamp(trace.started_at), "front matter", "run clock"
                ),
            ],
            [
                "Generated",
                self.ledger.declared(
                    _stamp(generated), "front matter", "packet clock"
                ),
            ],
            ["Outcome of the run", trace.status.value],
        ]
        self._table(["Field", "Value"], rows, [38.0, A4_USABLE_MM - 38.0])

    def _verdict_section(self) -> None:
        verdict = self.trace.verdict
        self._heading("The conclusion")
        if verdict is None:
            self._para(
                "No verdict was issued for this run. Nothing here weighed the "
                "measurements against a claim, so the measurements below stand on "
                "their own and no conclusion is asserted.",
            )
            return

        label = VERDICT_LABEL_TEXT[verdict.label]
        confidence = self.ledger.declared(
            f"{verdict.confidence * 100:.0f}%",
            "verdict",
            "sum of the confidence components listed below",
        )
        self._raw(
            f'<font color="{_VERDICT_COLOUR[verdict.label]}"><b>{escape(label)}</b>'
            f"</font> at {escape(confidence)} confidence",
            "verdict",
            audit=f"{label} at {confidence} confidence",
        )
        self._space(2)
        self._prose(
            verdict.reasoning,
            "verdict reasoning",
            source=(
                "written by the verdict engine from the evidence ledger, whose "
                f"figures are listed in {MEASUREMENTS_FILENAME}"
            ),
        )

        if verdict.asserted_direction.value != "unspecified":
            self._space(2)
            self._para(
                f"The claim asserts the quantity {verdict.asserted_direction.value}. "
                f"The imagery measures it {verdict.measured_direction.value}.",
                "small",
            )

        if verdict.narrative:
            self._space(3)
            self._prose(
                f"Phrased by {verdict.narrative_source}, with every figure checked "
                "against the ledger before it was accepted:",
                "phrased explanation",
                # Model names carry version numbers, and a version is an
                # identifier rather than a figure about the imagery.
                source="the name of the model that phrased it",
                style="small",
            )
            self._prose(
                verdict.narrative,
                "phrased explanation",
                source=(
                    "phrased by a language model and accepted only after every "
                    "figure in it was matched against the ledger"
                ),
                style="quote",
            )
        elif verdict.narrative_audit is not None and not verdict.narrative_audit.passed:
            self._space(3)
            self._prose(
                verdict.narrative_audit.note,
                "phrased explanation",
                source="the numeric audit that rejected the phrasing",
                style="small",
            )

        if verdict.what_would_change_it:
            self._space(3)
            self._para("What would change this answer", "subheading")
            for lever in verdict.what_would_change_it:
                self._prose(
                    f"\u2022 {lever}",
                    "what would change the answer",
                    source="the confidence component this lever would move",
                    style="small",
                )

    def _confidence_section(self) -> None:
        verdict = self.trace.verdict
        if verdict is None or not verdict.confidence_components:
            return

        self._heading("Confidence, by component")
        self._para(
            "A single percentage cannot be argued with, so it is reported as the "
            "parts that produced it. A component contributing nothing is where "
            "this finding is weak.",
            "small",
        )
        self._space(2)

        rows = []
        for component in verdict.confidence_components:
            measured = self.ledger.declared(
                f"{component.measured:.3f}",
                "confidence breakdown",
                f"{component.name} component, measured",
            )
            contribution = self.ledger.declared(
                f"{component.contribution:+.3f} of {component.weight:.2f}",
                "confidence breakdown",
                f"{component.name} component, weighted contribution",
            )
            rows.append(
                [
                    component.label,
                    measured,
                    contribution,
                    self.ledger.quoted(
                        component.rationale,
                        f"confidence: {component.name}",
                        source="written by the verdict engine from this component",
                    ),
                ]
            )
        self._table(
            ["Component", "Measured", "Contributes", "Why"],
            rows,
            [40.0, 18.0, 30.0, A4_USABLE_MM - 88.0],
        )

    def _confounder_section(self) -> None:
        tests = self.trace.confounders
        self._heading("Alternative explanations, tested")
        if not tests:
            self._para(
                "No alternative explanation was tested in this run. An untested "
                "finding is not a confirmed one.",
            )
            return

        ruled_out = sum(1 for test in tests if not test.survived)
        summary = self.ledger.declared(
            f"{ruled_out} of {len(tests)}",
            "alternative explanations",
            "count of confounder tests recorded in this run",
        )
        self._para(
            f"{summary} alternative explanations were ruled out. Each was given a "
            "threshold before it was measured, so the bar was not moved to fit the "
            "result.",
            "small",
        )
        self._space(2)

        rows: list[list[str]] = []
        colours: dict[int, str] = {}
        for index, test in enumerate(tests):
            measured = self.ledger.declared(
                test.measured,
                f"confounder: {test.kind.value}",
                f"measured by {test.method or 'the confounder engine'}",
            )
            threshold = self.ledger.declared(
                test.threshold,
                f"confounder: {test.kind.value}",
                "threshold declared before the measurement",
            )
            rows.append(
                [
                    CONFOUNDER_VERDICT_LABEL[test.verdict],
                    self.ledger.quoted(
                        test.label,
                        f"confounder: {test.kind.value}",
                        source="the alternative explanation as the contract named it",
                    ),
                    measured,
                    threshold,
                    self.ledger.quoted(
                        test.requirement or "\u2014",
                        f"confounder: {test.kind.value}",
                        source="the acquisition the test says would settle it",
                    ),
                ]
            )
            colours[index] = (
                "#047857"
                if not test.survived
                else ("#be123c" if test.verdict.value == "likely" else "#b45309")
            )
        self._table(
            ["Verdict", "Explanation", "Measured", "Threshold", "What would settle it"],
            rows,
            [24.0, 36.0, 26.0, 26.0, A4_USABLE_MM - 112.0],
            colours=colours,
        )

    def _figure_section(self) -> None:
        if self.figure is None:
            return
        from reportlab.lib.units import mm
        from reportlab.platypus import Image as PdfImage

        path, entries = self.figure
        self._heading("What was found, drawn")
        try:
            from PIL import Image as PilImage

            with PilImage.open(path) as probe:
                width, height = probe.size
        except Exception as exc:  # noqa: BLE001 - the figure is optional
            logger.warning("could not measure figure %s: %s", path.name, exc)
            return

        draw_width = min(A4_USABLE_MM, 150.0)
        draw_height = draw_width * height / width
        self.story.append(
            PdfImage(str(path), width=draw_width * mm, height=draw_height * mm)
        )
        self._space(2)
        self._prose(
            "The scene in EPSG:4326 with the detected regions composited over it. "
            "Areas are measured from the pixels, not from this rendering.",
            "map figure caption",
            source="the coordinate reference system the overlays are written in",
            style="small",
        )
        if entries:
            rows = []
            for entry in entries:
                area = (
                    self.ledger.declared(
                        f"{entry.area_km2:.4f} km\u00b2",
                        "map figure legend",
                        "area of the drawn mask, counted from its pixels",
                    )
                    if entry.area_km2 is not None
                    else "\u2014"
                )
                rows.append([entry.label, area])
            self._space(2)
            self._table(
                ["Drawn layer", "Area"], rows, [A4_USABLE_MM - 34.0, 34.0]
            )

    def _measurement_section(self) -> None:
        self._heading("Every measurement this rests on")
        if not self.trace.measurements:
            self._para("No tool produced a measurement in this run.")
            return

        count = self.ledger.declared(
            str(len(self.trace.measurements)),
            "measurements",
            "count of measurements recorded in the trace",
        )
        self._para(
            f"{count} measurements, each with the formula that produced it and the "
            "tool and version that ran it. This table is the spine of the packet: "
            "every figure elsewhere in this report resolves to a row here or to a "
            "threshold declared in advance.",
            "small",
        )
        self._space(2)

        rows = []
        for item in self.trace.measurements:
            rows.append(
                [
                    item.label,
                    self.ledger.measured(item, "measurements table"),
                    self.ledger.definition(
                        item.formula,
                        f"formula for {item.key}",
                        source="constant in the published formula",
                    ),
                    self.ledger.definition(
                        f"{item.source_tool} v{item.source_version}",
                        f"tool for {item.key}",
                        source="version of the tool that computed it",
                    ),
                ]
            )
        self._table(
            ["Quantity", "Value", "Formula", "Computed by"],
            rows,
            [44.0, 28.0, A4_USABLE_MM - 116.0, 44.0],
        )

    def _consistency_section(self) -> None:
        verdict = self.trace.verdict
        checks = verdict.consistency if verdict else []
        if not checks:
            return

        self._heading("Measured twice")
        self._para(
            "Quantities estimated by two independent routes, compared rather than "
            "averaged. An average of two disagreeing estimates is a number neither "
            "method supports.",
            "small",
        )
        self._space(2)

        rows = []
        for check in checks:
            pair = self.ledger.declared(
                f"{check.first_value:.3f} vs {check.second_value:.3f} {check.unit}",
                "cross-check",
                f"{check.first_key} against {check.second_key}",
            )
            apart = self.ledger.declared(
                f"{check.relative_difference * 100:.0f}%",
                "cross-check",
                "relative difference, |a - b| / max(|a|, |b|)",
            )
            rows.append(
                [
                    check.quantity,
                    pair,
                    apart,
                    "agree" if check.agrees else "beyond tolerance",
                    self.ledger.quoted(
                        check.explanation,
                        "cross-check",
                        source="written by the verdict engine from the two estimates",
                    ),
                ]
            )
        self._table(
            ["Quantity", "Two estimates", "Apart", "Reading", "Note"],
            rows,
            [30.0, 40.0, 14.0, 24.0, A4_USABLE_MM - 108.0],
        )

    def _input_section(self) -> None:
        self._heading("The imagery this was measured from")
        if self.session is None or not self.session.images:
            self._para("The input record was not available when this packet was made.")
            return

        rows = []
        for role, image in self.session.images.items():
            metadata = image.metadata
            geo = metadata.geo
            acquired = (
                self.ledger.declared(
                    _stamp(metadata.acquisition_date),
                    f"input: {role.value}",
                    f"acquisition date, {metadata.date_source.value}",
                )
                if metadata.acquisition_date
                else "not stated in the file"
            )
            grid = self.ledger.declared(
                f"{metadata.width}\u00d7{metadata.height} px"
                + (f", {geo.gsd_m:.2f} m" if geo.gsd_m else "")
                + (f", EPSG:{geo.epsg}" if geo.epsg else ""),
                f"input: {role.value}",
                "read from the raster header",
            )
            checksum = self.ledger.declared(
                image.sha256[:16],
                f"input: {role.value}",
                "SHA-256 of the file as ingested",
            )
            rows.append(
                [
                    role.value.replace("_", " "),
                    self.ledger.definition(
                        image.original_filename,
                        f"input: {role.value}",
                        source="the filename as supplied",
                    ),
                    acquired,
                    grid,
                    checksum,
                ]
            )
        self._table(
            ["Slot", "File", "Acquired", "Grid", "SHA-256 (first 16)"],
            rows,
            [20.0, 38.0, 30.0, 46.0, A4_USABLE_MM - 134.0],
        )
        self._space(2)
        self._para(
            "Band roles, per-band statistics and the full header are in "
            f"{TRACE_FILENAME}. The checksums are of the files as ingested, so the "
            "same inputs can be identified again.",
            "small",
        )

    def _readiness_section(self) -> None:
        report = self.readiness
        if report is None:
            return

        self._heading("Whether the imagery could carry the question")
        counts = {
            status: len(report.by_status(status))
            for status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL)
        }
        tally = self.ledger.declared(
            f"{counts[CheckStatus.PASS]} passed, {counts[CheckStatus.WARN]} warned, "
            f"{counts[CheckStatus.FAIL]} failed",
            "readiness gate",
            "count of readiness checks by status",
        )
        self._para(
            f"The gate ran before any analysis: {tally}. Verdict: "
            f"{report.verdict.value.replace('_', ' ')}.",
            "small",
        )
        self._space(2)

        shown = [
            check
            for check in report.checks
            if check.status is not CheckStatus.NOT_APPLICABLE
        ]
        rows: list[list[str]] = []
        colours: dict[int, str] = {}
        for index, check in enumerate(shown):
            rows.append(
                [
                    check.status.value,
                    self.ledger.definition(
                        check.label,
                        f"readiness: {check.id}",
                        source="the check as the gate names it",
                    ),
                    self.ledger.declared(
                        check.measured,
                        f"readiness: {check.id}",
                        f"measured by {check.method or 'the readiness gate'}",
                    ),
                    self.ledger.declared(
                        check.threshold or "\u2014",
                        f"readiness: {check.id}",
                        "threshold declared by the gate",
                    ),
                ]
            )
            colours[index] = _CHECK_COLOUR[check.status]
        self._table(
            ["Status", "Check", "Measured", "Threshold"],
            rows,
            [18.0, 52.0, 44.0, A4_USABLE_MM - 114.0],
            colours=colours,
        )

    def _pipeline_section(self) -> None:
        self._heading("How the run proceeded")
        rows = []
        for stage in self.trace.stages:
            detail = self.ledger.definition(
                stage.unavailable_note
                or "; ".join(step.label for step in stage.steps[:4]),
                f"stage: {stage.id.value}",
                source="the steps the run recorded for this stage",
            )
            duration = (
                self.ledger.declared(
                    f"{stage.duration_ms:.0f} ms",
                    f"stage: {stage.id.value}",
                    "wall-clock time the stage took",
                )
                if stage.duration_ms
                else "\u2014"
            )
            rows.append([stage.status.value, stage.label, duration, detail])
        self._table(
            ["Status", "Stage", "Took", "What happened"],
            rows,
            [18.0, 32.0, 18.0, A4_USABLE_MM - 68.0],
        )

    def _omissions_section(self) -> None:
        self._heading("What this packet does not contain")
        for omission in self.omissions:
            self._prose(
                f"\u2022 {omission}",
                "omissions",
                source="what the packet leaves out, and why",
                style="small",
            )


def _build_styles() -> dict[str, Any]:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm

    ink = colors.HexColor("#0f172a")
    dim = colors.HexColor("#475569")
    base = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=8.6,
        leading=11.4,
        textColor=ink,
        spaceAfter=2,
    )
    return {
        "title": ParagraphStyle(
            "title",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            spaceAfter=3,
        ),
        "lede": ParagraphStyle(
            "lede", parent=base, fontSize=9.4, leading=12.6, textColor=dim
        ),
        "heading": ParagraphStyle(
            "heading",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=2,
            spaceAfter=3,
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=base, fontName="Helvetica-Bold", fontSize=9.2
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=base, fontSize=13, leading=16, spaceAfter=2
        ),
        "body": base,
        "small": ParagraphStyle(
            "small", parent=base, fontSize=7.6, leading=10, textColor=dim
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base,
            leftIndent=4 * mm,
            borderPadding=2,
            textColor=colors.HexColor("#5b21b6"),
        ),
        "th": ParagraphStyle(
            "th", parent=base, fontName="Helvetica-Bold", fontSize=7.4, leading=9.4,
            textColor=colors.white, spaceAfter=0,
        ),
        "td": ParagraphStyle(
            "td", parent=base, fontSize=7.4, leading=9.4, spaceAfter=0
        ),
    }


def _stamp(moment: datetime | None) -> str:
    if moment is None:
        return "\u2014"
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def write_report_pdf(
    trace: RunTrace,
    directory: Path,
    *,
    session: SessionRecord | None,
    readiness: ReadinessReport | None,
    ledger: FigureLedger,
    figure: tuple[Path, list[FigureEntry]] | None,
    omissions: list[str],
) -> tuple[PacketFile, PacketAudit]:
    """Write the readable report and audit every figure in it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate

    builder = ReportBuilder(
        trace,
        session=session,
        readiness=readiness,
        ledger=ledger,
        figure=figure,
        omissions=omissions,
    )
    builder.build()
    audit = builder.audit()
    builder.traceability(audit)

    path = directory / REPORT_FILENAME
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"SatQuery evidence packet {trace.run_id[:8]}",
        author="SatQuery",
        subject=trace.claim or trace.query,
    )
    document.build(
        builder.story, onFirstPage=_page_footer, onLaterPages=_page_footer
    )

    described = _describe(
        path,
        PacketFileRole.REPORT,
        description=(
            "The finding, the confidence breakdown, the alternative explanations "
            "tested, and every measurement behind them."
        ),
        media_type="application/pdf",
    )
    return described, audit


def _page_footer(canvas, document) -> None:
    from reportlab.lib.units import mm

    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillGray(0.45)
    canvas.drawString(
        18 * mm,
        10 * mm,
        "SatQuery \u2014 every figure in this report traces to a measurement or a "
        "declared threshold.",
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def describe_omissions(trace: RunTrace, *, has_figure: bool) -> list[str]:
    """What the packet leaves out, said plainly.

    A reader who cannot tell "not measured" from "measured and omitted" has to
    assume the worst, so the difference is stated.
    """
    omissions = [
        "The source imagery itself. The packet records which files were used, "
        "with their checksums, rather than redistributing licensed data.",
        "Intermediate rasters. The index arrays and change magnitudes are "
        "recomputable from the inputs and the parameters in "
        f"{TRACE_FILENAME}.",
    ]
    if not has_figure:
        omissions.append(
            "A composed map figure, because no georeferenced overlay was "
            "rendered for this run."
        )
    if trace.verdict is None:
        omissions.append(
            "A verdict. Nothing in this run weighed the measurements against a "
            "claim, so no conclusion is asserted."
        )
    if not trace.confounders:
        omissions.append(
            "Confounder test results, because no alternative explanation was "
            "tested in this run."
        )
    untested = [
        test.kind.value for test in trace.confounders if test.verdict.value == "not_tested"
    ]
    if untested:
        omissions.append(
            "Measured evidence for these named alternative explanations, which "
            f"this build cannot test: {', '.join(sorted(set(untested)))}."
        )
    return omissions


def write_manifest(packet: EvidencePacket, directory: Path) -> Path:
    """Write the manifest: what is in the packet, and the source of every figure.

    The manifest is deliberately not listed among ``packet.files``. A file cannot
    record its own checksum, and a manifest that claimed to would be wrong the
    moment it was written.
    """
    path = directory / MANIFEST_FILENAME
    path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_archive(packet: EvidencePacket, directory: Path) -> PacketFile:
    """One download containing the whole packet."""
    path = directory / ARCHIVE_FILENAME
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in packet.files:
            source = directory / item.filename
            if source.exists():
                archive.write(source, arcname=item.filename)
        manifest = directory / MANIFEST_FILENAME
        if manifest.exists():
            archive.write(manifest, arcname=MANIFEST_FILENAME)
        archive.writestr("README.txt", _readme(packet))
    path.write_bytes(buffer.getvalue())
    return _describe(
        path,
        PacketFileRole.ARCHIVE,
        description=(
            f"{len(packet.files)} file(s), the manifest, and a plain-text guide to "
            "what each one is."
        ),
        media_type="application/zip",
    )


def _readme(packet: EvidencePacket) -> str:
    lines = [
        "SatQuery evidence packet",
        "=" * 24,
        "",
        f"Question : {packet.query}",
        f"Claim    : {packet.claim}",
        f"Verdict  : {packet.verdict_text or 'none issued'}",
        f"Run      : {packet.run_id}",
        f"Written  : {_stamp(packet.generated_at)}",
        "",
        "Contents",
        "--------",
    ]
    for item in packet.files:
        lines.append(f"{item.filename}")
        lines.append(f"    {item.description}")
        lines.append(f"    sha256 {item.sha256}")
    lines += [
        "",
        "Traceability",
        "------------",
        packet.audit.note,
        "",
        "What this packet does not contain",
        "---------------------------------",
    ]
    lines.extend(f"- {omission}" for omission in packet.omissions)
    return "\n".join(lines) + "\n"


def build_packet(
    trace: RunTrace,
    outcomes: dict[str, Any],
    directory: Path,
    *,
    session: SessionRecord | None = None,
    readiness: ReadinessReport | None = None,
    layers: Sequence[Any] = (),
) -> EvidencePacket:
    """Write every packet file into ``directory`` and describe what was written.

    The order matters: the figure is composed before the report because the report
    embeds it, and the manifest is written after the report because it records the
    report's checksum. The archive comes last for the same reason.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ledger = FigureLedger()

    files: list[PacketFile] = []

    figure_result = None
    try:
        figure_result = compose_figure(directory, layers)
    except Exception as exc:  # noqa: BLE001 - a figure is never load-bearing
        logger.warning("packet figure failed for run %s: %s", trace.run_id, exc)

    omissions = describe_omissions(trace, has_figure=figure_result is not None)

    report, audit = write_report_pdf(
        trace,
        directory,
        session=session,
        readiness=readiness,
        ledger=ledger,
        figure=(
            (directory / figure_result[0].filename, figure_result[1])
            if figure_result
            else None
        ),
        omissions=omissions,
    )
    files.append(report)
    files.append(write_trace_json(trace, directory))
    files.append(write_measurements_csv(trace, directory))

    findings = write_findings_geojson(trace, outcomes, directory)
    if findings is not None:
        files.append(findings)
    if figure_result is not None:
        files.append(figure_result[0])
    files.extend(collect_overlay_files(directory, layers))

    verdict = trace.verdict
    packet = EvidencePacket(
        run_id=trace.run_id,
        session_id=trace.session_id,
        query=trace.query,
        claim=trace.claim,
        verdict_label=verdict.label.value if verdict else None,
        verdict_text=(
            f"{VERDICT_LABEL_TEXT[verdict.label]} at {verdict.confidence:.0%} "
            "confidence"
            if verdict
            else ""
        ),
        confidence=verdict.confidence if verdict else None,
        files=files,
        figures=ledger.figures,
        audit=audit,
        omissions=omissions,
        notes=[
            "Every number in the report resolves to a row in "
            f"{MEASUREMENTS_FILENAME} or to a threshold the analysis declared "
            "before measuring.",
            "Checksums are of the files as written, so a recipient can show the "
            "packet is intact.",
        ],
    )

    # Manifest first so the archive can carry it, then the archive, then the
    # manifest again so the copy left on disk names the archive it is inside.
    write_manifest(packet, directory)
    packet.archive = write_archive(packet, directory)
    write_manifest(packet, directory)
    return packet


__all__ = [
    "ARCHIVE_FILENAME",
    "FINDINGS_FILENAME",
    "FIGURE_FILENAME",
    "MANIFEST_FILENAME",
    "MEASUREMENTS_FILENAME",
    "REPORT_FILENAME",
    "TRACE_FILENAME",
    "FigureLedger",
    "ReportBuilder",
    "build_packet",
    "collect_overlay_files",
    "compose_figure",
    "describe_omissions",
    "write_archive",
    "write_findings_geojson",
    "write_manifest",
    "write_measurements_csv",
    "write_report_pdf",
    "write_trace_json",
]
