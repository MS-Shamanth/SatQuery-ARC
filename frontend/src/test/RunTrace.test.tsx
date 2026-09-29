import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AgentGraph } from "../components/AgentGraph";
import { RunTracePanel } from "../components/RunTracePanel";
import type {
  Measurement,
  RunTrace,
  StageId,
  StageStatus,
  TraceStage,
  TraceStep,
} from "../lib/types";

function step(overrides: Partial<TraceStep> = {}): TraceStep {
  return {
    id: "run_specialists:grounding-engine",
    stage: "run_specialists",
    label: "grounding-engine",
    detail: "locate 'water'",
    status: "ok",
    tool: "grounding-engine",
    version: "1.0.0",
    implementation: "deterministic",
    parameters: { target: "water" },
    applies_to: ["single"],
    measurement_keys: ["grounded_area_km2"],
    mask_keys: ["grounded_water"],
    notes: [],
    reason: null,
    started_at: "2026-09-21T10:00:00Z",
    finished_at: "2026-09-21T10:00:01Z",
    duration_ms: 604,
    ...overrides,
  };
}

function stage(
  id: StageId,
  status: StageStatus,
  overrides: Partial<TraceStage> = {},
): TraceStage {
  return {
    id,
    label: id.replace(/_/g, " "),
    purpose: `purpose of ${id}`,
    status,
    steps: [],
    notes: [],
    unavailable_note: "",
    started_at: null,
    finished_at: null,
    duration_ms: 0,
    ...overrides,
  };
}

const measurement: Measurement = {
  key: "grounded_area_km2",
  label: "water region area",
  value: 9.6547,
  unit: "km2",
  formula: "area = 96547 px x 100.00 m2 / 1e6",
  inputs: { pixel_count: 96547, pixel_area_m2: 100 },
  source_tool: "grounding-engine",
  source_version: "1.0.0",
  method: "pixel count x pixel ground area",
  applies_to: ["single"],
  precision: 2,
};

function trace(overrides: Partial<RunTrace> = {}): RunTrace {
  return {
    run_id: "r".repeat(32),
    session_id: "s".repeat(32),
    contract_hash: "c".repeat(32),
    query: "Highlight the water body",
    claim: "The water body is visible and can be highlighted.",
    task_type: "region_grounding",
    configuration: "single",
    status: "completed",
    stages: [
      stage("verify_inputs", "ok", {
        steps: [
          step({
            id: "verify_inputs:gate",
            stage: "verify_inputs",
            label: "Readiness gate",
            detail: "7 passed, 0 warned, 0 failed",
            tool: null,
            version: null,
            implementation: null,
            parameters: {},
            measurement_keys: [],
            mask_keys: [],
          }),
        ],
      }),
      stage("run_specialists", "ok", { steps: [step()], duration_ms: 604 }),
      stage("resolve_verdict", "not_built", {
        unavailable_note:
          "No verdict is issued in this build. The system will not state a conclusion it cannot yet defend with a confidence breakdown.",
      }),
    ],
    tool_runs: [],
    measurements: [measurement],
    masks: [],
    confounders: [],
    verdict: null,
    ledger: null,
    remedies: null,
    disagreement: null,
    packet: null,
    started_at: "2026-09-21T10:00:00Z",
    finished_at: "2026-09-21T10:00:01Z",
    duration_ms: 922,
    error: null,
    refusal_reasons: [],
    ...overrides,
  };
}

describe("AgentGraph", () => {
  it("draws every declared stage, including the ones not built", () => {
    const current = trace();
    render(
      <AgentGraph
        stages={current.stages}
        status="completed"
        durationMs={current.duration_ms}
      />,
    );
    expect(screen.getByLabelText(/verify inputs: done/i)).toBeInTheDocument();
    expect(
      screen.getByLabelText(/resolve verdict: not in this build/i),
    ).toBeInTheDocument();
  });

  it("reports how far through the pipeline the run is", () => {
    render(
      <AgentGraph stages={trace().stages} status="running" durationMs={120} />,
    );
    expect(screen.getByText("3 / 3 stages")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("distinguishes a refusal from a failure", () => {
    render(<AgentGraph stages={[]} status="refused" durationMs={0} />);
    expect(screen.getByText("Refused")).toBeInTheDocument();
  });
});

describe("RunTracePanel", () => {
  const noop = vi.fn();

  const renderPanel = (current: RunTrace = trace(), streaming = false) =>
    render(
      <RunTracePanel
        trace={current}
        log={[]}
        streaming={streaming}
        onCancel={noop}
      />,
    );

  /**
   * Render with every stage open.
   *
   * The trace collapses by default, because printed in full it ran to thousands
   * of pixels and the reader scrolled past all of it. The assertions below are
   * about what the trace can show, not about what it shows first, so they open it
   * the way a reader would.
   */
  async function renderOpen(current: RunTrace = trace()) {
    const result = renderPanel(current);
    await userEvent.click(screen.getByRole("button", { name: "Show all" }));
    return result;
  }

  it("names every stage without being opened", () => {
    // The headings are the scannable layer and are never folded away.
    renderPanel();
    expect(screen.getByText(/run specialists/i)).toBeInTheDocument();
    expect(screen.getByText(/verify inputs/i)).toBeInTheDocument();
    expect(screen.getByText(/resolve verdict/i)).toBeInTheDocument();
  });

  it("collapses the detail until asked, and counts what is inside", () => {
    renderPanel();
    // The step, its parameters and the stage's purpose are all behind a click.
    expect(screen.queryByText("grounding-engine")).not.toBeInTheDocument();
    expect(screen.queryByText(/target="water"/)).not.toBeInTheDocument();
    expect(screen.queryByText(/purpose of run_specialists/i)).not.toBeInTheDocument();
    // But the reader can see there is something to open. Two stages carry one
    // step each in this fixture, so the count is per stage rather than global.
    expect(screen.getAllByText("1 step")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Show all" })).toBeInTheDocument();
  });

  it("opens one stage without opening the rest", async () => {
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /run specialists/i }));

    expect(screen.getByText("grounding-engine")).toBeInTheDocument();
    // The not_built stage's note belongs to a different stage and stays shut.
    expect(
      screen.queryByText(/will not state a conclusion it cannot yet defend/i),
    ).not.toBeInTheDocument();
  });

  it("show all opens everything and collapse all puts it back", async () => {
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: "Show all" }));
    expect(screen.getByText("grounding-engine")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Collapse all" }));
    expect(screen.queryByText("grounding-engine")).not.toBeInTheDocument();
  });

  it("keeps the stage in flight open, whatever the collapse state", () => {
    // Collapsing the trace must not take away the live view of the current stage.
    const running = trace({
      status: "running",
      stages: [
        stage("verify_inputs", "ok"),
        stage("run_specialists", "running", { steps: [step()] }),
      ],
    });
    renderPanel(running, true);
    expect(screen.getByText("grounding-engine")).toBeInTheDocument();
  });

  it("names the tool, its version, and how it arrives at its output", async () => {
    await renderOpen();
    expect(screen.getByText("grounding-engine")).toBeInTheDocument();
    expect(screen.getByText("v1.0.0")).toBeInTheDocument();
    expect(screen.getByText("computed")).toBeInTheDocument();
  });

  it("states what each stage is for", async () => {
    await renderOpen();
    expect(screen.getByText(/purpose of run_specialists/i)).toBeInTheDocument();
  });

  it("shows the parameters a tool was actually given", async () => {
    await renderOpen();
    expect(screen.getByText(/target="water"/)).toBeInTheDocument();
  });

  /* The Never-Guess Rule is only real if it is checkable from the interface. */
  it("traces a displayed number back to its formula and inputs", async () => {
    await renderOpen();

    expect(screen.queryByText(/96547 px x 100.00 m2/)).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: /Show 1 measurements/i }),
    );
    expect(screen.getByText("9.65")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /water region area/i }),
    );
    expect(screen.getByText(/area = 96547 px x 100.00 m2/)).toBeInTheDocument();
    expect(screen.getByText(/pixel_count=96547/)).toBeInTheDocument();
    expect(screen.getByText(/grounding-engine v1.0.0/)).toBeInTheDocument();
  });

  it("says why a stage that was not built produced nothing", async () => {
    await renderOpen();
    expect(
      screen.getByText(/will not state a conclusion it cannot yet defend/i),
    ).toBeInTheDocument();
  });

  it("gives a skipped tool's reason rather than omitting the row", async () => {
    const skipped = trace({
      stages: [
        stage("run_specialists", "skipped", {
          steps: [
            step({
              status: "skipped",
              reason: "'single' is missing band(s) swir16",
              measurement_keys: [],
            }),
          ],
        }),
      ],
    });
    await renderOpen(skipped);
    expect(screen.getByText(/missing band\(s\) swir16/)).toBeInTheDocument();
  });

  it("surfaces a gate refusal as the reason nothing ran", () => {
    const refused = trace({
      status: "refused",
      refusal_reasons: ["73.2% of the scene is cloud"],
    });
    render(
      <RunTracePanel trace={refused} log={[]} streaming={false} onCancel={noop} />,
    );
    const banner = screen.getByText(/The gate stopped this run/i).parentElement;
    expect(banner).not.toBeNull();
    expect(within(banner as HTMLElement).getByText(/73.2% of the scene is cloud/))
      .toBeInTheDocument();
  });

  it("streams the log and offers a stop while a run is in flight", async () => {
    const onCancel = vi.fn();
    render(
      <RunTracePanel
        trace={trace({ status: "running" })}
        log={[
          { seq: 1, at: "2026-09-21T10:00:00Z", text: "grounding-engine v1.0.0" },
        ]}
        streaming
        onCancel={onCancel}
      />,
    );
    expect(screen.getByRole("log", { name: /run log/i })).toHaveTextContent(
      /grounding-engine v1.0.0/,
    );
    await userEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("offers no stop once the run has ended", () => {
    render(
      <RunTracePanel
        trace={trace()}
        log={[]}
        streaming={false}
        onCancel={noop}
      />,
    );
    expect(screen.queryByRole("button", { name: /stop/i })).not.toBeInTheDocument();
  });
});
