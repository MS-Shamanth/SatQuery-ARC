import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ReadinessPanel } from "../components/ReadinessPanel";
import type { ReadinessCheck, ReadinessReport } from "../lib/types";

function check(overrides: Partial<ReadinessCheck>): ReadinessCheck {
  return {
    id: "crs.single",
    label: "Coordinate reference system",
    status: "pass",
    measured: "EPSG:32643",
    measured_numeric: 32643,
    unit: null,
    threshold: "a resolvable CRS",
    message: "WGS 84 / UTM zone 43N resolved from the file.",
    method: "rasterio dataset CRS",
    applies_to: ["single"],
    ...overrides,
  };
}

function report(overrides: Partial<ReadinessReport> = {}): ReadinessReport {
  return {
    session_id: "a".repeat(32),
    configuration: "single",
    verdict: "ready",
    checks: [check({})],
    refusal_reasons: [],
    requirements: [],
    computed_ms: 184.2,
    common_epsg: 32643,
    overlap_bounds_native: null,
    overlap_bounds_wgs84: null,
    overlap_fraction: null,
    day_delta: null,
    month_of_year_delta: null,
    seasonal_risk: false,
    ...overrides,
  };
}

describe("ReadinessPanel", () => {
  it("shows a radar sweep while the gate is running", () => {
    render(<ReadinessPanel report={null} loading error={null} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText(/Checking data readiness/i)).toBeInTheDocument();
  });

  it("reports a ready verdict with the pass count and elapsed time", () => {
    render(<ReadinessPanel report={report()} loading={false} error={null} />);
    const panel = screen.getByRole("region", { name: /data readiness/i });
    expect(within(panel).getByText("Ready")).toBeInTheDocument();
    expect(within(panel).getByText("1 pass")).toBeInTheDocument();
    expect(within(panel).getByText("184 ms")).toBeInTheDocument();
  });

  it("puts the measured value next to every check", () => {
    render(<ReadinessPanel report={report()} loading={false} error={null} />);
    expect(screen.getByText("EPSG:32643")).toBeInTheDocument();
  });

  it("reveals the threshold and measurement method on expand", async () => {
    render(<ReadinessPanel report={report()} loading={false} error={null} />);

    // Collapsed by default: the number is visible, the justification is not.
    expect(screen.queryByText(/a resolvable CRS/i)).not.toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /Coordinate reference system/i }),
    );

    expect(screen.getByText(/a resolvable CRS/i)).toBeInTheDocument();
    expect(screen.getByText(/rasterio dataset CRS/i)).toBeInTheDocument();
    expect(screen.getByText(/UTM zone 43N resolved/i)).toBeInTheDocument();
  });

  it("explains a refusal and states what would be needed", () => {
    render(
      <ReadinessPanel
        loading={false}
        error={null}
        report={report({
          verdict: "refused",
          checks: [
            check({
              id: "pair.coregistration",
              label: "Co-registration",
              status: "fail",
              measured: "7.21 px (72.10 m)",
              measured_numeric: 7.21,
              unit: "pixels",
              threshold: "pass at or below 1.0 px, fail above 2.0 px",
              message: "The images are offset by more than two pixels.",
              method: "phase cross-correlation on intensity",
            }),
          ],
          refusal_reasons: [
            "Co-registration: The images are offset by more than two pixels.",
          ],
          requirements: [
            {
              what: "A co-registered pair with sub-pixel alignment",
              why: "Beyond two pixels of offset, a change map shows edge artefacts rather than change.",
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("Refused")).toBeInTheDocument();
    expect(screen.getByText("1 fail")).toBeInTheDocument();
    expect(screen.getByText(/Why this was refused/i)).toBeInTheDocument();
    expect(screen.getByText(/What would be needed/i)).toBeInTheDocument();
    expect(
      screen.getByText(/A co-registered pair with sub-pixel alignment/i),
    ).toBeInTheDocument();
    // The measured offset is on screen, not just the word "refused".
    expect(screen.getByText("7.21 px (72.10 m)")).toBeInTheDocument();
  });

  it("collapses inapplicable checks behind a disclosure", async () => {
    render(
      <ReadinessPanel
        loading={false}
        error={null}
        report={report({
          checks: [
            check({}),
            check({
              id: "pair.overlap",
              label: "Spatial overlap",
              status: "not_applicable",
              measured: "not applicable",
              message: "Single-image input, so no pair check applies.",
              threshold: null,
            }),
          ],
        })}
      />,
    );

    expect(screen.queryByText("Spatial overlap")).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", {
      name: /Show 1 check that do not apply/i,
    });
    await userEvent.click(toggle);
    expect(screen.getByText("Spatial overlap")).toBeInTheDocument();
  });

  it("carries a seasonal offset forward to the confounder stage", () => {
    render(
      <ReadinessPanel
        loading={false}
        error={null}
        report={report({
          verdict: "ready_with_warnings",
          seasonal_risk: true,
          month_of_year_delta: 7,
          day_delta: 222,
        })}
      />,
    );

    expect(screen.getByText("Ready, with caveats")).toBeInTheDocument();
    expect(screen.getByText(/Carried forward/i)).toBeInTheDocument();
    expect(screen.getByText(/7 months/i)).toBeInTheDocument();
    expect(screen.getByText(/seasonality confounder test will run/i)).toBeInTheDocument();
  });

  it("surfaces a gate failure rather than rendering nothing", () => {
    render(
      <ReadinessPanel report={null} loading={false} error="Backend unreachable" />,
    );
    expect(screen.getByText("Backend unreachable")).toBeInTheDocument();
  });
});
