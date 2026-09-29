import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { VerdictCard } from "../components/VerdictCard";
import type {
  ConfidenceComponent,
  EvidenceItem,
  Verdict,
} from "../lib/types";

function component(
  overrides: Partial<ConfidenceComponent> = {},
): ConfidenceComponent {
  return {
    name: "measurement_quality",
    label: "How cleanly the index separates the class",
    measured: 0.85,
    contribution: 0.255,
    weight: 0.3,
    best: 0.3,
    rationale: "The index divides this scene into two clear populations.",
    lever: "A trained classifier",
    ...overrides,
  };
}

function evidence(overrides: Partial<EvidenceItem> = {}): EvidenceItem {
  return {
    measurement_key: "gain_km2",
    label: "built-up gained",
    value: 4.1,
    unit: "km2",
    display: "4.100 km2",
    direction: "supports",
    relevance: "area that crossed into the class",
    statement: "4.100 km2 was gained",
    source_tool: "change-cva-engine",
    source_version: "1.0.0",
    formula: "area = 41000 px x 100.00 m2 / 1e6",
    independent: false,
    weight: 1,
    ...overrides,
  };
}

function verdict(overrides: Partial<Verdict> = {}): Verdict {
  return {
    label: "supported",
    claim: "Built-up area increased between the two acquisitions.",
    asserted_direction: "increased",
    measured_direction: "increased",
    confidence: 0.72,
    confidence_components: [
      component(),
      component({
        name: "alternative_explanations",
        label: "Alternative explanations still standing",
        measured: 0,
        contribution: -0,
        weight: 0.6,
        best: 0,
        rationale: "All 3 alternative explanations tested were ruled out.",
        lever: "The acquisitions named under what would settle it",
      }),
    ],
    evidence: [evidence()],
    consistency: [
      {
        quantity: "area gained",
        label: "Two estimates of area gained",
        first_key: "gain_km2",
        first_value: 4.1,
        second_key: "corroborated_gain_km2",
        second_value: 3.9,
        unit: "km2",
        relative_difference: 0.0488,
        tolerance: 0.25,
        agrees: true,
        method: "the target's own index against an independent index",
        explanation:
          "Area gained measured two ways: 4.100 and 3.900 km2, a relative difference of 5%. Within the 25% tolerance, so the two routes agree.",
      },
    ],
    surviving_confounders: [],
    requirements: [],
    reasoning:
      "The claim asserts the quantity increased, and the measurements agree: area_change_km2 is +4.0000 km2.",
    narrative: null,
    narrative_source: "template",
    narrative_audit: null,
    what_would_change_it: [],
    ...overrides,
  };
}

describe("VerdictCard", () => {
  it("states the conclusion and the confidence together", () => {
    render(<VerdictCard verdict={verdict()} />);
    expect(screen.getByText("Supported")).toBeInTheDocument();
    expect(screen.getByText("72% confidence")).toBeInTheDocument();
    expect(
      screen.getByText(/Built-up area increased between the two acquisitions/),
    ).toBeInTheDocument();
  });

  it("shows the asserted direction against the measured one", () => {
    render(<VerdictCard verdict={verdict()} />);
    const region = screen.getByRole("region", { name: /verdict/i });
    expect(within(region).getByText(/asserted/)).toHaveTextContent(
      /measured/,
    );
  });

  it("distinguishes unanswerable from inconclusive", () => {
    render(<VerdictCard verdict={verdict({ label: "unanswerable" })} />);
    expect(
      screen.getByText("Unanswerable with this data"),
    ).toBeInTheDocument();
  });

  /* A lone percentage cannot be argued with; the components can. */
  it("breaks confidence into named components with their rationale", () => {
    render(<VerdictCard verdict={verdict()} />);
    expect(
      screen.getByText("How cleanly the index separates the class"),
    ).toBeInTheDocument();
    expect(screen.getByText("+0.26")).toBeInTheDocument();
    expect(screen.getByText("of 0.30")).toBeInTheDocument();
    expect(
      screen.getByText(/divides this scene into two clear populations/),
    ).toBeInTheDocument();
  });

  it("shows a penalty component as negative", () => {
    render(
      <VerdictCard
        verdict={verdict({
          confidence_components: [
            component({
              name: "alternative_explanations",
              label: "Alternative explanations still standing",
              measured: 0.7,
              contribution: -0.42,
              weight: 0.6,
              best: 0,
              rationale: "Seasonal phenology is the likely explanation.",
            }),
          ],
        })}
      />,
    );
    expect(screen.getByText("-0.42")).toBeInTheDocument();
  });

  it("reports two estimates of one quantity side by side", () => {
    render(<VerdictCard verdict={verdict()} />);
    expect(screen.getByText(/4.100 vs 3.900/)).toBeInTheDocument();
    expect(screen.getByText(/5% apart .* agree/)).toBeInTheDocument();
    expect(
      screen.getByText(/Within the 25% tolerance, so the two routes agree/),
    ).toBeInTheDocument();
  });

  it("marks disagreeing estimates as beyond tolerance", () => {
    const current = verdict();
    render(
      <VerdictCard
        verdict={verdict({
          consistency: [
            {
              ...current.consistency[0],
              second_value: 0.5,
              relative_difference: 0.878,
              agrees: false,
              explanation: "They are reported separately rather than averaged.",
            },
          ],
        })}
      />,
    );
    expect(screen.getByText(/beyond tolerance/)).toBeInTheDocument();
    expect(
      screen.getByText(/separately rather than averaged/),
    ).toBeInTheDocument();
  });

  it("names what would change the answer", () => {
    render(
      <VerdictCard
        verdict={verdict({
          label: "inconclusive",
          what_would_change_it: [
            "An acquisition from the same part of the growing season as date_a",
          ],
        })}
      />,
    );
    expect(
      screen.getByText(/What would change this answer/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/same part of the growing season/),
    ).toBeInTheDocument();
  });

  /* Every displayed number has to be two clicks from its derivation. */
  it("traces each admitted measurement back to its tool and formula", async () => {
    render(<VerdictCard verdict={verdict()} />);
    expect(screen.queryByText(/41000 px/)).not.toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /Show the 1 measurements/i }),
    );
    expect(screen.getByText("4.100 km2")).toBeInTheDocument();
    expect(
      screen.getByText(/gain_km2 .* change-cva-engine v1.0.0 .* 41000 px/),
    ).toBeInTheDocument();
  });

  it("labels a phrased explanation as such and reports its audit", () => {
    render(
      <VerdictCard
        verdict={verdict({
          narrative: "Built-up area grew by 4.100 km2.",
          narrative_source: "gemini-3.7-flash",
          narrative_audit: {
            checked: 3,
            traced: 3,
            untraceable: [],
            passed: true,
            note: "All 3 figures trace to a measurement in the ledger.",
          },
        })}
      />,
    );
    expect(
      screen.getByText(/Phrased by a language model/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/3\/3 figures traced/)).toBeInTheDocument();
    expect(screen.getByText(/grew by 4.100 km2/)).toBeInTheDocument();
  });

  it("says so when a phrased explanation was discarded", () => {
    render(
      <VerdictCard
        verdict={verdict({
          narrative: null,
          narrative_audit: {
            checked: 4,
            traced: 3,
            untraceable: ["61"],
            passed: false,
            note:
              "The phrased explanation was discarded because 61 does not appear in the evidence ledger.",
          },
        })}
      />,
    );
    expect(screen.getByText(/was discarded because 61/)).toBeInTheDocument();
    expect(
      screen.queryByText(/Phrased by a language model/i),
    ).not.toBeInTheDocument();
    // The measured reasoning survives, so the finding is not lost with it.
    expect(screen.getByText(/the measurements agree/)).toBeInTheDocument();
  });
});
