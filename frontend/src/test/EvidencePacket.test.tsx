import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EvidencePacketCard } from "../components/EvidencePacketCard";
import type {
  EvidencePacket,
  PacketFigure,
  PacketFile,
  PacketFileRole,
} from "../lib/types";

function file(
  filename: string,
  role: PacketFileRole,
  overrides: Partial<PacketFile> = {},
): PacketFile {
  return {
    filename,
    role,
    label: role,
    description: `the ${role}`,
    media_type: "application/octet-stream",
    size_bytes: 12_345,
    sha256: "a".repeat(64),
    ...overrides,
  };
}

function figure(overrides: Partial<PacketFigure> = {}): PacketFigure {
  return {
    display: "9.6547 km2",
    where: "measurements table",
    measurement_key: "grounded_area_km2",
    source_tool: "grounding-engine",
    source_version: "1.0.0",
    formula: "pixel_count * pixel_area_m2 / 1e6",
    source: "",
    ...overrides,
  };
}

function packet(overrides: Partial<EvidencePacket> = {}): EvidencePacket {
  return {
    run_id: "b".repeat(32),
    session_id: "c".repeat(32),
    generated_at: "2026-09-21T20:00:00Z",
    query: "Did the reservoir shrink?",
    claim: "The water body shrank between the two dates.",
    verdict_label: "supported",
    verdict_text: "Supported at 71% confidence",
    confidence: 0.7123,
    files: [
      file("report.pdf", "report", { size_bytes: 478_236 }),
      file("trace.json", "trace"),
      file("measurements.csv", "measurements"),
      file("findings.geojson", "findings"),
    ],
    figures: [figure()],
    audit: {
      checked: 355,
      traced: 355,
      untraceable: [],
      passed: true,
      note: "All 355 figure(s) in this report are attributed to a measurement.",
    },
    archive: file("satquery-evidence.zip", "archive", { size_bytes: 3_120_489 }),
    notes: [],
    omissions: [
      "The source imagery itself. The packet records which files were used.",
    ],
    ...overrides,
  };
}

describe("EvidencePacketCard", () => {
  it("renders nothing when no packet was written", () => {
    const { container } = render(
      <EvidencePacketCard packet={null} sessionId={"c".repeat(32)} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("offers the whole packet as one download", () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    const link = screen.getByRole("link", { name: /Download everything/ });
    expect(link).toHaveAttribute("download");
    expect(link.getAttribute("href")).toContain(
      "/packet/satquery-evidence.zip",
    );
  });

  it("offers the report on its own, since that is what gets read", () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    const link = screen.getByRole("link", { name: /^Report/ });
    expect(link.getAttribute("href")).toContain("/packet/report.pdf");
  });

  it("states how many of the report's figures are attributed", () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    expect(
      screen.getByText(/Every figure in the report is attributed/),
    ).toBeInTheDocument();
    expect(screen.getByText(/355 of 355 figures trace/)).toBeInTheDocument();
  });

  it("says so loudly when a figure could not be attributed", () => {
    render(
      <EvidencePacketCard
        packet={packet({
          audit: {
            checked: 90,
            traced: 88,
            untraceable: ['16 (in "SHA-256")', '-256 (in "SHA-256")'],
            passed: false,
            note: "These figures could not be attributed.",
          },
        })}
        sessionId={"c".repeat(32)}
      />,
    );
    expect(
      screen.getByText(/Some figures could not be attributed/),
    ).toBeInTheDocument();
    expect(screen.getByText(/88 of 90 figures trace/)).toBeInTheDocument();
    expect(screen.getByText('16 (in "SHA-256")')).toBeInTheDocument();
  });

  it("shows where each measured figure came from on request", async () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    expect(screen.queryByText(/grounded_area_km2/)).toBeNull();

    await userEvent.click(
      screen.getByRole("button", { name: /where each of the 1 measured/ }),
    );
    expect(screen.getByText("9.6547 km2")).toBeInTheDocument();
    expect(
      screen.getByText(/grounded_area_km2.*grounding-engine.*pixel_count/),
    ).toBeInTheDocument();
  });

  it("does not list a declared threshold among the measured figures", async () => {
    render(
      <EvidencePacketCard
        packet={packet({
          figures: [
            figure(),
            figure({
              display: "0.500",
              measurement_key: "",
              source_tool: "",
              source: "threshold declared in advance",
            }),
          ],
        })}
        sessionId={"c".repeat(32)}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: /where each of the 1 measured/ }),
    );
    expect(screen.queryByText("0.500")).toBeNull();
  });

  it("lists every file with its checksum on request", async () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    await userEvent.click(
      screen.getByRole("button", { name: /all 4 files with their checksums/ }),
    );
    expect(screen.getByRole("link", { name: "trace.json" })).toBeInTheDocument();
    expect(screen.getAllByText(/sha256 aaaa/)).toHaveLength(4);
  });

  it("says what the packet does not contain", () => {
    render(<EvidencePacketCard packet={packet()} sessionId={"c".repeat(32)} />);
    expect(
      screen.getByText(/What this packet does not contain/),
    ).toBeInTheDocument();
    expect(screen.getByText(/The source imagery itself/)).toBeInTheDocument();
  });

  it("counts several files of the same kind rather than hiding them", () => {
    render(
      <EvidencePacketCard
        packet={packet({
          files: [
            file("report.pdf", "report"),
            file("a.png", "overlay"),
            file("b.png", "overlay"),
            file("c.png", "overlay"),
          ],
        })}
        sessionId={"c".repeat(32)}
      />,
    );
    expect(screen.getByText(/1 of 3/)).toBeInTheDocument();
  });

  it("still renders when no archive was written", () => {
    render(
      <EvidencePacketCard
        packet={packet({ archive: null })}
        sessionId={"c".repeat(32)}
      />,
    );
    expect(screen.queryByRole("link", { name: /Download everything/ })).toBeNull();
    expect(screen.getByRole("link", { name: /^Report/ })).toBeInTheDocument();
  });
});
