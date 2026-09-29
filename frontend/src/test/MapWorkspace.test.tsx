/**
 * The map, mounted the way the browser mounts it.
 *
 * The map is the one panel that appears only after a run starts, which made it
 * the prime suspect for the blank page: everything was fine until approve-and-run,
 * and then the whole workspace disappeared. Leaflet creates its map imperatively
 * against a container element, and React 19's StrictMode mounts every effect,
 * unmounts it, and mounts it again. A second initialisation against the same
 * container throws, React unmounts the tree, and the page goes white.
 *
 * So this renders inside StrictMode on purpose. None of the other component tests
 * do, which is why none of them caught it.
 */

import { StrictMode } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MapWorkspace } from "../components/MapWorkspace";
import type { MapLayer } from "../lib/types";

function layer(overrides: Partial<MapLayer> = {}): MapLayer {
  return {
    key: "grounded_water",
    label: "Water",
    description: "MNDWI above 0",
    kind: "mask",
    png_url: "/api/sessions/a/runs/b/layers/g.png",
    geojson_url: null,
    bounds_wgs84: [73.5611, 21.2217, 73.6108, 21.2683],
    colour: "#22d3ee",
    area_km2: 9.6547,
    pixel_count: 96547,
    applies_to: ["single"],
    ...overrides,
  };
}

const base = layer({
  key: "base.single",
  label: "Single imagery",
  kind: "base",
  png_url: "/api/sessions/a/runs/b/layers/base.png",
  area_km2: null,
  pixel_count: null,
});

describe("MapWorkspace under StrictMode", () => {
  it("mounts without throwing when React double-invokes the effects", () => {
    // If Leaflet is initialised twice against one container this throws, and in
    // the browser that takes the entire page with it.
    expect(() =>
      render(
        <StrictMode>
          <MapWorkspace layers={[base, layer()]} loading={false} error={null} />
        </StrictMode>,
      ),
    ).not.toThrow();
  });

  it("survives being remounted, as it is on every new run", () => {
    const { unmount } = render(
      <StrictMode>
        <MapWorkspace layers={[base, layer()]} loading={false} error={null} />
      </StrictMode>,
    );
    unmount();

    expect(() =>
      render(
        <StrictMode>
          <MapWorkspace layers={[base, layer()]} loading={false} error={null} />
        </StrictMode>,
      ),
    ).not.toThrow();
  });

  it("renders the layer list with its measured areas", async () => {
    render(
      <StrictMode>
        <MapWorkspace layers={[base, layer()]} loading={false} error={null} />
      </StrictMode>,
    );
    expect(await screen.findByText("Water")).toBeInTheDocument();
  });

  it("does not attempt a map when no layer has been rendered yet", () => {
    expect(() =>
      render(
        <StrictMode>
          <MapWorkspace layers={[]} loading={true} error={null} />
        </StrictMode>,
      ),
    ).not.toThrow();
  });

  it("tolerates a layer whose bounds are degenerate", () => {
    // Bounds come from the server, and a zero-area footprint is a Leaflet error
    // rather than a sensible viewport.
    expect(() =>
      render(
        <StrictMode>
          <MapWorkspace
            layers={[layer({ bounds_wgs84: [0, 0, 0, 0] })]}
            loading={false}
            error={null}
          />
        </StrictMode>,
      ),
    ).not.toThrow();
  });
});
