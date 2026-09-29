import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DisagreementPanel } from "../components/DisagreementPanel";
import type { MapLayer, Measurement } from "../lib/types";

function measurement(
  key: string,
  value: number,
  unit: string,
  inputs: Record<string, unknown> = {},
): Measurement {
  return {
    key,
    label: key.replace(/_/g, " "),
    value,
    unit,
    formula: `${key} = measured`,
    inputs,
    source_tool: "optical-sar-fusion",
    source_version: "1.0.0",
    method: null,
    applies_to: ["optical", "sar"],
    precision: 3,
  };
}

const AGREEING: Measurement[] = [
  measurement("agreement_iou", 0.8144, "ratio"),
  measurement("agreement_kappa", 0.8787, "kappa", {
    interpretation: "almost perfect",
  }),
  measurement("agreed_water_km2", 3.7064, "km2"),
  measurement("sar_only_km2", 0.1338, "km2"),
  measurement("optical_only_km2", 0.711, "km2"),
];

function layer(key: string, area: number): MapLayer {
  return {
    key,
    label: key,
    description: "",
    kind: "mask",
    png_url: `/api/x/${key}.png`,
    geojson_url: null,
    bounds_wgs84: [85.11, 25.59, 85.16, 25.64],
    colour: "#34d399",
    area_km2: area,
    pixel_count: 1000,
    applies_to: ["optical", "sar"],
  };
}

describe("DisagreementPanel", () => {
  it("reports both agreement figures, because they answer different questions", () => {
    render(
      <DisagreementPanel measurements={AGREEING} layers={[]} notes={[]} />,
    );
    expect(screen.getByText("Overlap")).toBeInTheDocument();
    expect(screen.getByText("0.81")).toBeInTheDocument();
    expect(screen.getByText("Kappa")).toBeInTheDocument();
    expect(screen.getByText("0.88")).toBeInTheDocument();
    expect(screen.getByText("almost perfect")).toBeInTheDocument();
  });

  it("calls strong agreement what it is", () => {
    render(
      <DisagreementPanel measurements={AGREEING} layers={[]} notes={[]} />,
    );
    expect(screen.getByText("Strong agreement")).toBeInTheDocument();
  });

  it("does not claim agreement when the sensors conflict", () => {
    render(
      <DisagreementPanel
        measurements={[
          measurement("agreement_iou", 0.12, "ratio"),
          measurement("agreement_kappa", 0.09, "kappa", {
            interpretation: "slight",
          }),
          measurement("agreed_water_km2", 0.2, "km2"),
          measurement("sar_only_km2", 1.4, "km2"),
          measurement("optical_only_km2", 1.1, "km2"),
        ]}
        layers={[]}
        notes={[]}
      />,
    );
    expect(screen.getByText("They disagree")).toBeInTheDocument();
  });

  /* Who saw what is the whole content of the disagreement map. */
  it("separates both-sensors, radar-only, and optical-only with their areas", () => {
    render(
      <DisagreementPanel measurements={AGREEING} layers={[]} notes={[]} />,
    );
    expect(screen.getByText("Both sensors")).toBeInTheDocument();
    expect(screen.getByText("3.71 km²")).toBeInTheDocument();
    expect(screen.getByText("Radar only")).toBeInTheDocument();
    expect(screen.getByText("0.134 km²")).toBeInTheDocument();
    expect(screen.getByText("Optical only")).toBeInTheDocument();
    expect(screen.getByText("0.711 km²")).toBeInTheDocument();
  });

  it("explains why each kind of disagreement happens", () => {
    render(
      <DisagreementPanel measurements={AGREEING} layers={[]} notes={[]} />,
    );
    expect(
      screen.getByText(/Radar sees water the optical result does not/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/independently place water here/),
    ).toBeInTheDocument();
  });

  /* A sensor that was blocked made no claim, so it cannot have disagreed. */
  it("keeps radar-under-cloud out of the agreement figures and says so", () => {
    render(
      <DisagreementPanel
        measurements={AGREEING}
        layers={[layer("fusion_sar_under_cloud", 0.0964)]}
        notes={[]}
      />,
    );
    expect(
      screen.getByText(/Radar water where optical could not see/),
    ).toBeInTheDocument();
    expect(screen.getByText("0.096 km²")).toBeInTheDocument();
    expect(
      screen.getByText(/cannot be said to disagree/),
    ).toBeInTheDocument();
  });

  it("reports how much of the radar-only water sits under cloud", () => {
    render(
      <DisagreementPanel
        measurements={[
          ...AGREEING,
          measurement("sar_only_under_cloud_share", 0.0635, "fraction"),
        ]}
        layers={[]}
        notes={[]}
      />,
    );
    expect(screen.getByText("6%")).toBeInTheDocument();
  });

  it("surfaces the engine's own reading of the comparison", () => {
    render(
      <DisagreementPanel
        measurements={AGREEING}
        layers={[]}
        notes={[
          "Raw overlap 0.81, kappa 0.88 (almost perfect). Two instruments measuring unrelated physics placing water in the same place is the strongest evidence available from this input.",
          "Only 6% of the water only radar found sits near optical cloud.",
        ]}
      />,
    );
    expect(screen.getByText(/unrelated physics/)).toBeInTheDocument();
  });

  it("renders nothing when no fusion ran", () => {
    const { container } = render(
      <DisagreementPanel measurements={[]} layers={[]} notes={[]} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
