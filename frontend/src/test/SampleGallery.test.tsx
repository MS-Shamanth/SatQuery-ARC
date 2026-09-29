import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SampleGallery } from "../components/SampleGallery";
import type { SampleAssetRecord, SampleScene } from "../lib/types";

function asset(overrides: Partial<SampleAssetRecord> = {}): SampleAssetRecord {
  return {
    role: "single",
    filename: "water_body__single.tif",
    sha256: "f".repeat(64),
    size_bytes: 3_100_000,
    width: 512,
    height: 512,
    band_count: 7,
    provenance: {
      source: "Copernicus Sentinel",
      collection: "sentinel-2-l2a",
      stac_item_id: "S2B_43QCD_20240112_1_L2A",
      stac_url: "https://earth-search.aws.element84.com/v1/collections/x/items/y",
      acquisition_date: "2024-01-12T05:31:00Z",
      cloud_cover_percent: 0.0,
      platform: "sentinel-2b",
      instrument: "MSI",
      bands: ["blue", "green", "red", "nir", "swir16", "swir22", "scl"],
      gsd_m: 10,
      epsg: 32643,
      licence: "Copernicus Sentinel data",
      attribution: "Contains modified Copernicus Sentinel data",
      is_simulated: false,
      simulation_note: null,
      fetched_at: "2026-09-21T13:35:00Z",
    },
    ...overrides,
  };
}

const cached: SampleScene = {
  key: "water_body",
  title: "Ukai reservoir",
  description: "A single clear-sky Sentinel-2 scene.",
  demo: "Demo 1 - single-image grounding",
  configuration: "single",
  suggested_queries: ["Highlight the water body referred to in the query"],
  place: "Ukai reservoir, Tapi basin, Gujarat",
  assets: { single: asset() },
  remedies: [],
  corrects: null,
  remedy_note: "",
};

const uncached: SampleScene = {
  ...cached,
  key: "urban_growth",
  title: "Dholera greenfield build-out",
  place: "Dholera Special Investment Region, Gujarat",
  configuration: "bi_temporal_pair",
  assets: {},
};

const simulated: SampleScene = {
  ...cached,
  key: "flood_urban",
  title: "Patna monsoon flooding",
  place: "Patna and the Ganga floodplain, Bihar",
  configuration: "cross_modal_pair",
  assets: {
    optical: asset({ role: "optical" }),
    sar: asset({
      role: "sar",
      provenance: {
        ...asset().provenance,
        collection: "simulated-sar",
        stac_item_id: null,
        licence: null,
        is_simulated: true,
        simulation_note: "Backscatter simulated from the optical land cover.",
      },
    }),
  },
};

const noop = vi.fn();

describe("SampleGallery", () => {
  it("shows the place and acquisition date for a cached scene", () => {
    render(
      <SampleGallery
        scenes={[cached]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    const gallery = screen.getByRole("region", { name: /sample scenes/i });
    expect(within(gallery).getByText("Ukai reservoir")).toBeInTheDocument();
    expect(
      within(gallery).getByText("Ukai reservoir, Tapi basin, Gujarat"),
    ).toBeInTheDocument();
    expect(within(gallery).getByText(/12 Jan 2024/)).toBeInTheDocument();
    expect(within(gallery).getByText("sentinel-2-l2a")).toBeInTheDocument();
  });

  it("labels the input configuration of each scene", () => {
    render(
      <SampleGallery
        scenes={[cached, uncached, simulated]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.getByText("1 image")).toBeInTheDocument();
    expect(screen.getByText("2 dates")).toBeInTheDocument();
    expect(screen.getByText("optical + SAR")).toBeInTheDocument();
  });

  it("flags a scene that includes a simulated asset", () => {
    render(
      <SampleGallery
        scenes={[simulated]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.getByText(/includes simulated asset/i)).toBeInTheDocument();
  });

  it("does not flag a fully real scene as simulated", () => {
    render(
      <SampleGallery
        scenes={[cached]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.queryByText(/simulated/i)).not.toBeInTheDocument();
  });

  it("explains how to populate an uncached scene instead of offering a dead button", () => {
    render(
      <SampleGallery
        scenes={[uncached]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.getByText(/Not cached/i)).toBeInTheDocument();
    expect(screen.getByText("scripts/fetch_samples.py")).toBeInTheDocument();
    expect(screen.getByText("scripts/make_synthetic.py")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Load this scene/i }),
    ).not.toBeInTheDocument();
  });

  it("loads a scene on click", async () => {
    const onLoad = vi.fn();
    render(
      <SampleGallery
        scenes={[cached]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey={null}
        onLoad={onLoad}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /Load this scene/i }));
    expect(onLoad).toHaveBeenCalledWith(cached);
  });

  it("disables every card while one is loading", () => {
    render(
      <SampleGallery
        scenes={[cached, simulated]}
        loading={false}
        error={null}
        loadingKey="water_body"
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.getByText(/Loading/)).toBeInTheDocument();
    for (const button of screen.getAllByRole("button")) {
      expect(button).toBeDisabled();
    }
  });

  it("marks the active scene as loaded", () => {
    render(
      <SampleGallery
        scenes={[cached]}
        loading={false}
        error={null}
        loadingKey={null}
        activeKey="water_body"
        onLoad={noop}
      />,
    );
    expect(screen.getByRole("button", { name: "Loaded" })).toBeInTheDocument();
  });

  it("surfaces a library error", () => {
    render(
      <SampleGallery
        scenes={[]}
        loading={false}
        error="Could not read the sample scene library"
        loadingKey={null}
        activeKey={null}
        onLoad={noop}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/Could not read/i);
  });
});
