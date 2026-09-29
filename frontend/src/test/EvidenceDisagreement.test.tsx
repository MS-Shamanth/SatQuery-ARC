/**
 * Evidence disagreement panel.
 *
 * The panel exists to make the system's own uncertainty the headline, so the
 * tests are about what it refuses to say as much as what it shows: no winner is
 * named, a method that could not see is not presented as dissenting, and the
 * percentage on screen is the one the backend measured rather than one the
 * component recomputed from areas it happens to have.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { EvidenceDisagreementPanel } from "../components/EvidenceDisagreementPanel";
import type {
  ConflictRegion,
  DisagreementReport,
  MapLayer,
  MethodOpinion,
} from "../lib/types";

function opinion(overrides: Partial<MethodOpinion> = {}): MethodOpinion {
  return {
    method: "Spectral index (NDBI)",
    verdict: "BUILT-UP",
    source_tool: "spectral-index-engine",
    source_version: "1.0.0",
    basis: "NDBI thresholded by Otsu at 0.04",
    confidence: 0.32,
    could_not_see: false,
    ...overrides,
  };
}

function region(overrides: Partial<ConflictRegion> = {}): ConflictRegion {
  return {
    region_id: 17,
    agreement: "disagree",
    area_km2: 1.24,
    pixel_count: 12_400,
    centroid_wgs84: [73.5842, 21.2451],
    bounds_wgs84: [],
    opinions: [
      opinion(),
      opinion({
        method: "Learned probe",
        verdict: "not built-up",
        source_tool: "rs-landcover-probe",
        source_version: "0.1.0",
        basis: "logistic probe over six bands",
        confidence: null,
      }),
    ],
    reason:
      "Low spectral separability: Spectral index (NDBI) splits this scene at only 0.32, so its boundary here is a line through one population rather than between two.",
    reason_measurement_key: "class_separability",
    reason_value: 0.321,
    action: "Human review recommended.",
    ...overrides,
  };
}

function report(overrides: Partial<DisagreementReport> = {}): DisagreementReport {
  return {
    candidate_area_km2: 25.13,
    agree_area_km2: 11.88,
    disagree_area_km2: 13.25,
    uncertain_area_km2: 0,
    conflicting_fraction: 0.5273,
    methods: ["Learned probe", "Spectral index (NDBI)"],
    regions: [region()],
    applies_to: ["single"],
    agree_layer: "evidence_agree",
    disagree_layer: "evidence_disagree",
    uncertain_layer: "evidence_uncertain",
    notes: [
      "A method that could not see a pixel is excluded from the comparison rather than counted as disagreeing. Cloud and missing data are not dissent.",
      "No method is treated as correct. There is no labelled reference for this scene, so a conflict is reported as ground for review rather than resolved by preferring one instrument.",
    ],
    ...overrides,
  };
}

const evidenceLayer = (key: string): MapLayer => ({
  key,
  label: key,
  description: key,
  kind: "mask",
  png_url: `/api/sessions/s/runs/r/layers/${key}.png`,
  geojson_url: `/api/sessions/s/runs/r/layers/${key}.geojson`,
  bounds_wgs84: [73.5, 21.2, 73.6, 21.3],
  colour: "#ef4444",
  area_km2: 1,
  pixel_count: 1000,
  applies_to: ["single"],
});

describe("evidence disagreement", () => {
  it("shows nothing at all when the comparison did not run", () => {
    // Fewer than two methods produced a class mask. An empty panel claiming 0%
    // contested would be asserting agreement that was never measured.
    const { container } = render(
      <EvidenceDisagreementPanel disagreement={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("leads with the contested share the backend measured", () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);

    const panel = screen.getByLabelText("Evidence disagreement");
    expect(within(panel).getByText("52.7%")).toBeInTheDocument();
    expect(
      within(panel).getByText(/conflicting evidence across/i),
    ).toHaveTextContent("2 methods");
  });

  it("says the red is not an error", () => {
    // The whole framing of the panel. Without this line a reader sees a map of
    // red patches and concludes the system got something wrong.
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    expect(
      screen.getByText(/These are not errors\./i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the ground where the methods do not agree/i),
    ).toBeInTheDocument();
  });

  it("names the three states and the area in each", () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    const panel = screen.getByLabelText("Evidence disagreement");

    expect(within(panel).getAllByText("Methods agree").length).toBeGreaterThan(0);
    expect(within(panel).getAllByText("Methods disagree").length).toBeGreaterThan(0);
    expect(
      within(panel).getAllByText(/Not enough confidence to say/i).length,
    ).toBeGreaterThan(0);
    // Both areas appear, so the bar is readable as numbers and not only as colour.
    expect(within(panel).getByText(/11\.88/)).toBeInTheDocument();
    expect(within(panel).getByText(/13\.25/)).toBeInTheDocument();
  });

  it("lists the candidate area and every method compared", () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    // Anchored to the start of the line: "candidate area" also appears in the
    // headline sentence and in the bar's accessible name.
    const footer = screen.getByText(/^Candidate area/i);
    expect(footer).toHaveTextContent("25.13");
    expect(footer).toHaveTextContent("Learned probe");
    expect(footer).toHaveTextContent("Spectral index (NDBI)");
  });

  it("opens a region to show each method's verdict, the reason, and the action", async () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);

    const row = screen.getByRole("button", { name: /Region #17/ });
    expect(row).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(row);
    expect(row).toHaveAttribute("aria-expanded", "true");

    // Per-method verdicts, in each method's own terms.
    expect(screen.getByText("Spectral index (NDBI)")).toBeInTheDocument();
    expect(screen.getByText(/→ BUILT-UP/)).toBeInTheDocument();
    expect(screen.getByText("Learned probe")).toBeInTheDocument();
    expect(screen.getByText(/→ not built-up/)).toBeInTheDocument();

    expect(screen.getByText("Reason")).toBeInTheDocument();
    expect(screen.getByText(/Low spectral separability/)).toBeInTheDocument();
    expect(screen.getByText("Action")).toBeInTheDocument();
    expect(screen.getByText("Human review recommended.")).toBeInTheDocument();
  });

  it("attributes the reason to the measurement it came from", async () => {
    // "Low spectral separability" is only allowed on screen where the
    // separability was actually computed, and the figure has to be visible.
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    await userEvent.click(screen.getByRole("button", { name: /Region #17/ }));

    expect(screen.getByText(/from class_separability = 0\.321/)).toBeInTheDocument();
  });

  it("attributes every opinion to a tool and a version", async () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    await userEvent.click(screen.getByRole("button", { name: /Region #17/ }));

    expect(screen.getByText(/spectral-index-engine v1\.0\.0/)).toBeInTheDocument();
    expect(screen.getByText(/rs-landcover-probe v0\.1\.0/)).toBeInTheDocument();
  });

  it("shows a method that could not see as abstaining, not dissenting", async () => {
    // Cloud is not an opinion. Listing it as a dissent would invent a conflict.
    render(
      <EvidenceDisagreementPanel
        disagreement={report({
          regions: [
            region({
              opinions: [
                opinion(),
                opinion({
                  method: "Radar backscatter",
                  verdict: "could not see it",
                  source_tool: "sar-backscatter-engine",
                  could_not_see: true,
                  confidence: null,
                }),
              ],
            }),
          ],
        })}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /Region #17/ }));

    expect(screen.getByText(/→ could not see it/)).toBeInTheDocument();
  });

  it("declares no winner anywhere on screen", () => {
    render(<EvidenceDisagreementPanel disagreement={report()} />);
    const text = screen.getByLabelText("Evidence disagreement").textContent ?? "";

    for (const word of ["ground truth", "more accurate", "is correct", "is wrong"]) {
      expect(text.toLowerCase()).not.toContain(word);
    }
    expect(text).toContain("No method is treated as correct");
  });

  it("keeps regions the methods agree about out of the review queue", () => {
    render(
      <EvidenceDisagreementPanel
        disagreement={report({
          regions: [region(), region({ region_id: 18, agreement: "agree" })],
        })}
      />,
    );

    expect(screen.getByText(/1 contested region/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Region #18/ })).toBeNull();
  });

  it("counts uncertain regions as contested, since they are not agreement", () => {
    render(
      <EvidenceDisagreementPanel
        disagreement={report({
          regions: [region(), region({ region_id: 19, agreement: "uncertain" })],
        })}
      />,
    );
    expect(screen.getByText(/2 contested regions/)).toBeInTheDocument();
  });

  it("points at the map overlays only when they were actually drawn", () => {
    const { rerender } = render(
      <EvidenceDisagreementPanel disagreement={report()} layers={[]} />,
    );
    expect(screen.queryByText(/drawn on the map above/i)).toBeNull();

    rerender(
      <EvidenceDisagreementPanel
        disagreement={report()}
        layers={[evidenceLayer("evidence_disagree")]}
      />,
    );
    expect(screen.getByText(/drawn on the map above/i)).toBeInTheDocument();
  });

  it("survives a report with no contested regions at all", () => {
    render(
      <EvidenceDisagreementPanel
        disagreement={report({
          disagree_area_km2: 0,
          uncertain_area_km2: 0,
          agree_area_km2: 25.13,
          conflicting_fraction: 0,
          regions: [],
        })}
      />,
    );
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.queryByText(/contested region/)).toBeNull();
  });
});
