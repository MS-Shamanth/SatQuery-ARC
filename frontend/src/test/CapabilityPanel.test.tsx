import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { CapabilityPanel } from "../components/CapabilityPanel";
import type { SessionToolAvailability, ToolDescriptor } from "../lib/types";

const indexEngine: ToolDescriptor = {
  name: "spectral-index-engine",
  version: "1.0.0",
  implementation: "deterministic",
  summary: "NDVI, NDWI, MNDWI, NDBI, and NDMI from reflectance.",
  requirement: {
    band_roles: [],
    any_of_band_roles: [["nir", "red"]],
    modalities: ["optical"],
    configurations: ["single"],
    requires_crs: false,
    description: "An optical image with at least one index's band pair.",
  },
  parameters: { indices: { default: ["NDVI"] } },
  produces: ["ndwi_area_km2"],
  registered: true,
};

const gisEngine: ToolDescriptor = {
  ...indexEngine,
  name: "gis-measure-engine",
  summary: "Areas from pixel counts, components, centroids, IoU, kappa.",
  requirement: { ...indexEngine.requirement, description: "Any georeferenced raster." },
};

const narrator: ToolDescriptor = {
  ...indexEngine,
  name: "rsvlm-narrator",
  implementation: "llm-narration",
  summary: "Phrases the explanation from computed evidence.",
};

const availability: SessionToolAvailability = {
  session_id: "a".repeat(32),
  configuration: "single",
  available: ["gis-measure-engine"],
  unavailable: [
    {
      tool: "spectral-index-engine",
      reason: "no optical image is present in this session",
    },
  ],
  declined_registration: {
    "rs-landcover-probe": "no trained weights in data/models",
  },
};

describe("CapabilityPanel", () => {
  it("lists every registered tool", () => {
    render(
      <CapabilityPanel
        tools={[indexEngine, gisEngine]}
        availability={null}
        loading={false}
        error={null}
      />,
    );
    const panel = screen.getByRole("region", { name: /specialist tools/i });
    expect(within(panel).getByText("spectral-index-engine")).toBeInTheDocument();
    expect(within(panel).getByText("gis-measure-engine")).toBeInTheDocument();
  });

  it("labels how each tool arrives at its output", () => {
    render(
      <CapabilityPanel
        tools={[indexEngine, narrator]}
        availability={null}
        loading={false}
        error={null}
      />,
    );
    expect(screen.getByText("computed")).toBeInTheDocument();
    expect(screen.getByText("wording only")).toBeInTheDocument();
  });

  it("states the Never-Guess division of labour", () => {
    render(
      <CapabilityPanel tools={[indexEngine]} availability={null} loading={false} error={null} />,
    );
    expect(screen.getByText(/never produces a number/i)).toBeInTheDocument();
  });

  it("gives the reason a tool cannot run on this input", () => {
    render(
      <CapabilityPanel
        tools={[indexEngine, gisEngine]}
        availability={availability}
        loading={false}
        error={null}
      />,
    );
    expect(
      screen.getByText(/no optical image is present in this session/i),
    ).toBeInTheDocument();
    expect(screen.getByText("1 of 2 usable here")).toBeInTheDocument();
  });

  it("explains a tool that never registered", () => {
    render(
      <CapabilityPanel
        tools={[gisEngine]}
        availability={availability}
        loading={false}
        error={null}
      />,
    );
    expect(screen.getByText(/Not loaded/i)).toBeInTheDocument();
    expect(screen.getByText("rs-landcover-probe")).toBeInTheDocument();
    expect(screen.getByText(/no trained weights/i)).toBeInTheDocument();
  });

  it("reveals the summary, version, and requirements on expand", async () => {
    render(
      <CapabilityPanel tools={[indexEngine]} availability={null} loading={false} error={null} />,
    );
    expect(screen.queryByText(/version 1.0.0/i)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /spectral-index-engine/i }));

    expect(screen.getByText(/NDVI, NDWI, MNDWI/i)).toBeInTheDocument();
    expect(screen.getByText(/version 1.0.0/i)).toBeInTheDocument();
    expect(screen.getByText(/at least one index's band pair/i)).toBeInTheDocument();
  });

  it("surfaces a registry error", () => {
    render(
      <CapabilityPanel
        tools={[]}
        availability={null}
        loading={false}
        error="Could not read the tool registry"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/Could not read/i);
  });
});
