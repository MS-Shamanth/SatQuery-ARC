import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ContractCard } from "../components/ContractCard";
import type { AnalysisContract } from "../lib/types";

function contract(overrides: Partial<AnalysisContract> = {}): AnalysisContract {
  return {
    session_id: "a".repeat(32),
    query: "Has the built-up area increased since 2020?",
    claim: "Built-up area increased between the two acquisitions.",
    task_type: "claim_investigation",
    configuration: "bi_temporal_pair",
    required_modalities: ["optical"],
    acts_on: ["date_a", "date_b"],
    target_classes: ["built-up"],
    change_direction: "increased",
    metrics_requested: ["area_km2_before", "area_km2_after", "percentage_change"],
    confounders: [
      {
        kind: "seasonality",
        label: "Seasonal phenology",
        question:
          "Could the annual growing cycle account for this instead of a real change?",
        reason: "the acquisitions sit about 0 month(s) apart in the annual cycle",
      },
      {
        kind: "misregistration",
        label: "Misregistration",
        question:
          "Could the two images be slightly offset, so edges register as change?",
        reason: "co-registration measured 0.46 px (4.61 m)",
      },
    ],
    tools: [
      {
        tool: "spectral-index-engine",
        version: "1.0.0",
        rationale: "compute NDBI as independent evidence",
        parameters: { indices: ["NDBI"] },
        order: 0,
      },
      {
        tool: "gis-measure-engine",
        version: "1.0.0",
        rationale: "measure the areas",
        parameters: {},
        order: 1,
      },
    ],
    expected_outputs: ["a change map", "before and after measurements", "a verdict"],
    source: "gemini",
    model: "gemini-3.7-flash",
    interpretation_note: "Read as a claim about built-up expansion.",
    repairs: [],
    rejections: [],
    contract_hash: "c".repeat(32),
    generated_at: "2026-09-21T10:00:00Z",
    duration_ms: 1842,
    fallback_reason: null,
    ...overrides,
  };
}

const noop = vi.fn();

/** The claim types itself out, so the body is gated behind that finishing. */
async function renderSettled(node: React.ReactElement) {
  render(node);
  await waitFor(
    () => expect(screen.getByText(/Tools to run/i)).toBeInTheDocument(),
    { timeout: 4000 },
  );
}

describe("ContractCard", () => {
  it("shows the claim under test as a declarative statement", async () => {
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByText(/Claim under test/i)).toBeInTheDocument();
    expect(
      screen.getByText("Built-up area increased between the two acquisitions."),
    ).toBeInTheDocument();
  });

  it("lists the planned tools with versions and parameters", async () => {
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByText("spectral-index-engine")).toBeInTheDocument();
    expect(screen.getAllByText("v1.0.0")).toHaveLength(2);
    expect(screen.getByText(/indices=\["NDBI"\]/)).toBeInTheDocument();
  });

  it("names what could make the answer wrong, with measured reasons", async () => {
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByText(/What could make this wrong/i)).toBeInTheDocument();
    expect(screen.getByText("Seasonal phenology")).toBeInTheDocument();
    expect(screen.getByText(/co-registration measured 0.46 px/)).toBeInTheDocument();
  });

  it("says which path drafted the contract", async () => {
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByText(/drafted by the language model/i)).toBeInTheDocument();
    expect(screen.getByText(/gemini-3.7-flash/)).toBeInTheDocument();
  });

  it("says when it fell back to the offline router, and why", async () => {
    await renderSettled(
      <ContractCard
        contract={contract({
          source: "offline-rule-router",
          model: null,
          fallback_reason: "no Gemini model produced a contract",
        })}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(
      screen.getByText(/drafted by the offline rule router/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/no Gemini model produced a contract/i)).toBeInTheDocument();
  });

  it("hides validation changes behind a disclosure and reveals them on demand", async () => {
    await renderSettled(
      <ContractCard
        contract={contract({
          rejections: [
            {
              field: "tools",
              value: "changeformer-large",
              reason: "not a registered tool",
            },
          ],
          repairs: [
            {
              field: "task_type",
              detail: "promoted to claim_investigation",
            },
          ],
        })}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );

    expect(screen.queryByText(/not a registered tool/i)).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: /Show 2 validation changes/i }),
    );
    expect(screen.getByText(/not a registered tool/i)).toBeInTheDocument();
    expect(screen.getByText(/promoted to claim_investigation/i)).toBeInTheDocument();
  });

  it("requires approval before anything runs", async () => {
    const onApprove = vi.fn();
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={onApprove}
        onDiscard={noop}
      />,
    );

    const button = screen.getByRole("button", { name: /Approve and run/i });
    expect(onApprove).not.toHaveBeenCalled();
    await userEvent.click(button);
    expect(onApprove).toHaveBeenCalledTimes(1);
  });

  it("can be discarded", async () => {
    const onDiscard = vi.fn();
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={onDiscard}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /Discard/i }));
    expect(onDiscard).toHaveBeenCalledTimes(1);
  });

  it("shows a drafting state", () => {
    render(
      <ContractCard
        contract={null}
        loading
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText(/Drafting the analysis contract/i)).toBeInTheDocument();
  });

  it("surfaces a drafting failure", () => {
    render(
      <ContractCard
        contract={null}
        loading={false}
        error="Could not draft an analysis contract"
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/Could not draft/i);
  });

  it("renders nothing before a question is asked", () => {
    const { container } = render(
      <ContractCard
        contract={null}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("reports the asserted direction alongside the task", async () => {
    await renderSettled(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );
    const region = screen.getByRole("region", { name: /analysis contract/i });
    expect(within(region).getByText("Claim investigation")).toBeInTheDocument();
    expect(within(region).getByText(/asserted increased/i)).toBeInTheDocument();
  });

  /*
    The suite otherwise runs with reduced motion preferred, which skips the
    typewriter outright. This is the path the demo recording actually shows, and
    it is where the reveal gate can go wrong: completing the reveal re-renders
    the parent, and if that feeds back into the reveal the claim retypes forever
    and the body below it never opens.
  */
  it("opens the body once the claim finishes typing, and leaves it typed", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent: () => false,
    }));

    const claim = contract().claim;
    render(
      <ContractCard
        contract={contract()}
        loading={false}
        error={null}
        onApprove={noop}
        onDiscard={noop}
      />,
    );

    // Mid-reveal the gated body is genuinely absent, not merely transparent.
    expect(screen.queryByText(/Tools to run/i)).not.toBeInTheDocument();

    await waitFor(() => expect(screen.getByText(claim)).toBeInTheDocument(), {
      timeout: 4000,
    });
    await waitFor(() => expect(screen.getByText(/Tools to run/i)).toBeInTheDocument());

    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(screen.getByText(claim)).toBeInTheDocument();
  });
});
