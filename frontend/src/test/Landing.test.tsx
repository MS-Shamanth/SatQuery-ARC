import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Landing } from "../pages/Landing";
import type { HealthResponse } from "../lib/types";

const healthy: HealthResponse = {
  status: "ok",
  app: "SatQuery AI",
  version: "0.1.0",
  environment: "test",
  components: {
    raster_stack: {
      component: "raster_stack",
      status: "ok",
      detail: "rasterio 1.5.1 on GDAL 3.12.4",
      checked_at: 1,
    },
    gemini: {
      component: "gemini",
      status: "ok",
      detail: "Key accepted.",
      checked_at: 1,
    },
    copernicus: {
      component: "copernicus",
      status: "ok",
      detail: "Credentials accepted.",
      checked_at: 1,
    },
    earth_search: {
      component: "earth_search",
      status: "ok",
      detail: "Reachable.",
      checked_at: 1,
    },
  },
  capabilities: {
    contract_source: "language-model",
    narration_available: true,
    llm_provider: "mistral",
    llm_models: ["mistral/open-mistral-nemo"],
    sar_source: "copernicus-sentinel-1-grd",
    optical_source: "earth-search-sentinel-2-l2a",
    geospatial_compute: true,
  },
};

function mockHealth(body: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok,
        status: ok ? 200 : 500,
        statusText: ok ? "OK" : "Server Error",
        json: () => Promise.resolve(body),
        text: () => Promise.resolve(JSON.stringify(body)),
      } as Response),
    ),
  );
}

const renderLanding = () =>
  render(
    <MemoryRouter>
      <Landing />
    </MemoryRouter>,
  );

describe("Landing", () => {
  beforeEach(() => mockHealth(healthy));

  it("renders the positioning statement", async () => {
    renderLanding();
    expect(await screen.findByText(/Investigates claims/i)).toBeInTheDocument();
    expect(screen.getByText(/Not just answers questions/i)).toBeInTheDocument();
  });

  it("shows all three verdict states, so no forced yes/no is implied", async () => {
    renderLanding();
    expect(await screen.findByText("Supported")).toBeInTheDocument();
    for (const verdict of ["Inconclusive", "Refuted"]) {
      expect(screen.getByText(verdict)).toBeInTheDocument();
    }
  });

  it("renders all eight pipeline stages in order", async () => {
    renderLanding();
    const list = await screen.findByLabelText("Analysis pipeline");
    const labels = Array.from(list.querySelectorAll("li")).map((li) =>
      li.textContent?.slice(0, 20),
    );
    expect(labels).toHaveLength(8);
    expect(labels[0]).toContain("Readiness");
    expect(labels[7]).toContain("Packet");
  });

  it("cites the BigEarthNet.txt adaptation dataset with a working link", async () => {
    renderLanding();
    const link = await screen.findByRole("link", { name: /BigEarthNet.txt/i });
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/2603.29630");
  });

  it("surfaces nominal system status once health resolves", async () => {
    renderLanding();
    await waitFor(() =>
      expect(screen.getByText(/All systems nominal/i)).toBeInTheDocument(),
    );
  });

  it("surfaces an unreachable backend instead of failing silently", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("refused"))));
    renderLanding();
    await waitFor(() =>
      expect(screen.getByText(/Backend unreachable/i)).toBeInTheDocument(),
    );
  });

  it("renders the decorative hero without requiring canvas support", async () => {
    renderLanding();
    expect(await screen.findByTestId("starfield")).toBeInTheDocument();
    expect(screen.getByTestId("orbital-scene")).toBeInTheDocument();
  });
});
