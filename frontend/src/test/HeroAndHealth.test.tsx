/**
 * The hero graphic and the status panel, both of which were saying the wrong thing.
 *
 * The orbits drew flattened, tilted ellipses while the platforms were positioned by
 * a CSS rotation, which traces a circle. The dot met its own orbit at two points
 * and spent the rest of the loop out in open space. The fix was to make the track
 * and the route the same path, so the test worth having is the one that asserts
 * they still are: a platform whose motion references a path that is not on screen
 * is a platform that is not on the line.
 *
 * The status panel listed every probe with its full detail, so it opened with a
 * rasterio build number and a paragraph about a provider that was not in use. The
 * tests here are about what it leads with and what it leaves out.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HealthPill } from "../components/HealthPill";
import { OrbitalScene } from "../components/OrbitalScene";
import { installMatchMedia } from "./setup";
import type { HealthResponse } from "../lib/types";

describe("OrbitalScene", () => {
  // The suite forces reduced motion by default, which is the branch that renders
  // no animation at all. The moving platforms are the thing under test here, so
  // this is the one place that needs motion on. The global afterEach restores it.
  beforeEach(() => installMatchMedia(false));

  it("routes every platform along a track that is actually drawn", () => {
    const { container } = render(<OrbitalScene size={420} />);

    const motions = container.querySelectorAll("animateMotion");
    expect(motions.length).toBeGreaterThan(0);

    for (const motion of Array.from(motions)) {
      const mpath = motion.querySelector("mpath");
      expect(mpath, "a platform with no route").not.toBeNull();

      const href =
        mpath?.getAttribute("href") ?? mpath?.getAttribute("xlink:href") ?? "";
      expect(href.startsWith("#")).toBe(true);

      // The invariant. If this resolves to nothing, the dot is travelling a path
      // the viewer cannot see, which is the bug this replaced.
      const target = container.querySelector(href);
      expect(target, `no path matches ${href}`).not.toBeNull();
      expect(target?.tagName.toLowerCase()).toBe("path");
      // And the path has to be stroked, or it is invisible even if it is correct.
      expect(target?.getAttribute("stroke")).toBeTruthy();
    }
  });

  it("gives each track its own path rather than sharing one", () => {
    const { container } = render(<OrbitalScene size={420} />);
    const ids = Array.from(container.querySelectorAll("path[id]")).map(
      (path) => path.id,
    );
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBeGreaterThanOrEqual(2);
  });

  it("puts the platform in the same tilted group as its track", () => {
    // The tilt is applied once, to a group containing both. Two copies of the
    // rotation is how they drifted apart the first time.
    const { container } = render(<OrbitalScene size={420} />);
    const tilted = Array.from(container.querySelectorAll("g[transform]")).filter(
      (group) => (group.getAttribute("transform") ?? "").startsWith("rotate("),
    );
    expect(tilted.length).toBeGreaterThanOrEqual(2);
  });

  it("scales without leaving the platforms behind", () => {
    const { container } = render(<OrbitalScene size={260} />);
    for (const mpath of Array.from(container.querySelectorAll("mpath"))) {
      const href = mpath.getAttribute("href") ?? "";
      expect(container.querySelector(href)).not.toBeNull();
    }
  });

  it("parks the platforms on their tracks when motion is not wanted", () => {
    installMatchMedia(true);
    const size = 420;
    const { container } = render(<OrbitalScene size={size} />);

    // No animation, and no platform stranded at the origin either: a parked dot
    // still has to sit on the line.
    expect(container.querySelectorAll("animateMotion")).toHaveLength(0);
    const parked = Array.from(container.querySelectorAll("g[transform]")).filter(
      (group) =>
        (group.getAttribute("transform") ?? "").startsWith("translate("),
    );
    expect(parked.length).toBeGreaterThanOrEqual(2);
    for (const group of parked) {
      const [x, y] = (group.getAttribute("transform") ?? "")
        .replace(/translate\(|\)/g, "")
        .split(/[\s,]+/)
        .map(Number);
      // On the horizontal extreme of its own ellipse: y at the centre, x outside
      // the globe.
      expect(y).toBeCloseTo(size / 2, 5);
      expect(x).toBeGreaterThan(size / 2);
    }
  });
});

const health: HealthResponse = {
  status: "ok",
  app: "SatQuery AI",
  version: "0.1.0",
  environment: "development",
  components: {
    raster_stack: {
      component: "raster_stack",
      status: "ok",
      detail: "rasterio 1.5.1 on GDAL 3.12.4; GeoTIFF reads available.",
      checked_at: 1,
    },
    llm_provider: {
      component: "llm_provider",
      status: "ok",
      detail:
        "mistral: 1 model(s) generated: open-mistral-nemo. Unavailable: mistral-small-latest: HTTP 429 Rate limit exceeded | openrouter: The free allowance on this account is spent",
      checked_at: 1,
    },
    gemini: {
      component: "gemini",
      status: "skipped",
      detail: "Not the selected provider. Set LLM_PROVIDER=gemini to use it.",
      checked_at: 1,
    },
    copernicus: {
      component: "copernicus",
      status: "ok",
      detail: "Credentials accepted; access token issued for Sentinel-1 retrieval.",
      checked_at: 1,
    },
    earth_search: {
      component: "earth_search",
      status: "ok",
      detail: "Sentinel-2 L2A collection reachable without credentials.",
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

const openPanel = async (payload: HealthResponse | null = health) => {
  render(
    <HealthPill health={payload} error={null} loading={false} onRefresh={vi.fn()} />,
  );
  await userEvent.click(screen.getByRole("button", { name: /System status/i }));
};

describe("HealthPill", () => {
  it("leads with what is in use, naming the model that answered", async () => {
    await openPanel();

    expect(screen.getByText("In use")).toBeInTheDocument();
    expect(screen.getByText("Language model")).toBeInTheDocument();
    // Provider and model, with the chain's provider prefix stripped off the model
    // because the provider is already named beside it.
    expect(screen.getByText("mistral · open-mistral-nemo")).toBeInTheDocument();
    expect(screen.getByText("Sentinel-1 GRD (real)")).toBeInTheDocument();
    expect(screen.getByText("Sentinel-2 L2A (real)")).toBeInTheDocument();
  });

  it("leaves out the diagnostics nobody opened it to read", async () => {
    await openPanel();

    // A build number and a paragraph of quota prose about a provider that is not
    // being used are not the answer to "what is running".
    expect(screen.queryByText(/GDAL 3\.12\.4/)).toBeNull();
    expect(screen.queryByText(/Rate limit exceeded/)).toBeNull();
    expect(screen.queryByText(/access token issued/)).toBeNull();
  });

  it("does not list a provider that is deliberately on standby", async () => {
    await openPanel();
    expect(screen.queryByText(/Gemini/)).toBeNull();
    expect(screen.queryByText("Needs attention")).toBeNull();
  });

  it("keeps the full probe text one click away", async () => {
    await openPanel();
    await userEvent.click(screen.getByRole("button", { name: "Probe detail" }));

    expect(screen.getByText(/GDAL 3\.12\.4/)).toBeInTheDocument();
    // Including the standby row, which belongs in diagnostics rather than up front.
    expect(screen.getByText(/Gemini \(standby\)/)).toBeInTheDocument();
  });

  it("still shows a real failure rather than hiding it to stay tidy", async () => {
    await openPanel({
      ...health,
      status: "degraded",
      components: {
        ...health.components,
        copernicus: {
          component: "copernicus",
          status: "invalid",
          detail: "Credentials rejected.",
          checked_at: 1,
        },
      },
    });

    expect(screen.getByText("Needs attention")).toBeInTheDocument();
    const section = screen.getByText("Needs attention").parentElement;
    expect(
      within(section as HTMLElement).getByText(/Copernicus \(SAR\)/),
    ).toBeInTheDocument();
  });

  it("says so plainly when no model is configured at all", async () => {
    await openPanel({
      ...health,
      capabilities: {
        ...health.capabilities,
        llm_provider: "none",
        llm_models: [],
        contract_source: "offline-rule-router",
      },
    });

    expect(
      screen.getByText(/none, using the offline rule router/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Offline rule router")).toBeInTheDocument();
  });

  it("reports the backend being unreachable without pretending otherwise", async () => {
    render(
      <HealthPill
        health={null}
        error="Backend unreachable"
        loading={false}
        onRefresh={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /System status/i }));
    // The phrase is both the pill's own label and the body of the panel, so the
    // assertion is on the count rather than on a single node.
    await waitFor(() =>
      expect(screen.getAllByText(/Backend unreachable/).length).toBeGreaterThan(0),
    );
    expect(screen.queryByText("In use")).toBeNull();
    expect(screen.getByText(/uvicorn app.main:app/)).toBeInTheDocument();
  });
});
