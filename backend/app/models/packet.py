"""The evidence packet: the run, made portable.

An answer that exists only inside a web page cannot be filed, forwarded, or
challenged six months later by someone who was not in the room. The packet is
the same finding as a set of files: a PDF that can be read on its own, the
machine-readable trace, the measurements as a table, the footprints as GeoJSON,
and the overlays as images.

Two rules shape it.

**Every figure is attributed.** ``PacketFigure`` records, for each number printed
in the report, the measurement it came from and the formula behind it. The report
is not allowed to contain a number that is not in that list, and
``PacketAudit`` is the check that says so. This is the Never-Guess Rule applied
to the export rather than only to the phrased explanation.

**What is missing is listed.** ``omissions`` names what the packet does not
contain and why. A reader who cannot tell the difference between "this was not
measured" and "this was measured and omitted" has to assume the worst, so the
packet says which it is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class PacketFileRole(str, Enum):
    """What each file in the packet is for."""

    REPORT = "report"
    TRACE = "trace"
    MEASUREMENTS = "measurements"
    FINDINGS = "findings"
    OVERLAY = "overlay"
    FIGURE = "figure"
    MANIFEST = "manifest"
    ARCHIVE = "archive"


ROLE_LABEL: dict[PacketFileRole, str] = {
    PacketFileRole.REPORT: "Readable report",
    PacketFileRole.TRACE: "Machine-readable trace",
    PacketFileRole.MEASUREMENTS: "Measurements table",
    PacketFileRole.FINDINGS: "Footprints as vectors",
    PacketFileRole.OVERLAY: "Map overlay",
    PacketFileRole.FIGURE: "Composed map figure",
    PacketFileRole.MANIFEST: "Packet manifest",
    PacketFileRole.ARCHIVE: "Everything, zipped",
}


class PacketFile(BaseModel):
    """One file in the packet, with a checksum so it can be shown to be intact."""

    filename: str
    role: PacketFileRole
    label: str
    description: str
    media_type: str
    size_bytes: int
    # Present so a recipient can prove the file they hold is the file that was
    # written, without trusting the channel it arrived over.
    sha256: str


class PacketFigure(BaseModel):
    """A number printed in the report, and where it came from.

    ``measurement_key`` is empty only when ``source`` explains a different
    provenance: a threshold the analysis declared in advance, a count of records,
    or a value read from the file's own metadata. Those are traceable too, just
    not to a tool's measurement.
    """

    display: str
    where: str
    measurement_key: str = ""
    source_tool: str = ""
    source_version: str = ""
    formula: str = ""
    # Stated when the figure is not a tool measurement, e.g. "declared threshold"
    # or "read from the raster header".
    source: str = ""

    @property
    def traced_to_measurement(self) -> bool:
        return bool(self.measurement_key)


class PacketAudit(BaseModel):
    """Whether every number in the report is attributable.

    The report is assembled from a figure ledger, so in a correct build this
    passes by construction. It is checked anyway and reported on the packet's own
    front page, because an invariant that is never tested is an assumption.
    """

    checked: int = 0
    traced: int = 0
    untraceable: list[str] = Field(default_factory=list)
    passed: bool = True
    note: str = ""


class EvidencePacket(BaseModel):
    """The reproducible record of one run."""

    run_id: str
    session_id: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    query: str
    claim: str
    # Repeated here so the packet reads on its own, without the trace beside it.
    verdict_label: str | None = None
    verdict_text: str = ""
    confidence: float | None = None

    files: list[PacketFile] = Field(default_factory=list)
    figures: list[PacketFigure] = Field(default_factory=list)
    audit: PacketAudit = Field(default_factory=PacketAudit)

    # The single download. Listed separately because it contains the others.
    archive: PacketFile | None = None

    notes: list[str] = Field(default_factory=list)
    # What this packet does not contain, and why.
    omissions: list[str] = Field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    def by_role(self, role: PacketFileRole) -> list[PacketFile]:
        return [item for item in self.files if item.role is role]

    def file(self, filename: str) -> PacketFile | None:
        if self.archive is not None and self.archive.filename == filename:
            return self.archive
        for item in self.files:
            if item.filename == filename:
                return item
        return None

    def summary_line(self) -> str:
        return (
            f"{len(self.files)} file(s), {self.total_bytes / 1024:.0f} kB, "
            f"{len(self.figures)} figure(s) attributed"
        )


__all__ = [
    "ROLE_LABEL",
    "EvidencePacket",
    "PacketAudit",
    "PacketFigure",
    "PacketFile",
    "PacketFileRole",
]
