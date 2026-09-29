import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Studio } from "../pages/Studio";
import type {
  IngestedImage,
  ReadinessReport,
  SampleManifest,
  SessionRecord,
} from "../lib/types";

const SESSION_ID = "a".repeat(32);

const opticalImage: IngestedImage = {
  role: "single",
  stored_filename: "single.tif",
  original_filename: "bhuj_2020.tif",
  content_type: "image/tiff",
  sha256: "c".repeat(64),
  ingested_at: "2026-09-21T10:00:00Z",
  thumbnail_available: true,
  thumbnail_recipe: "True colour: R=red, G=green, B=blue",
  ingest_ms: 412.5,
  metadata: {
    original_filename: "bhuj_2020.tif",
    size_bytes: 2_411_724,
    driver: "GTiff",
    format_class: "geospatial",
    width: 1024,
    height: 768,
    band_count: 6,
    dtype: "uint16",
    megapixels: 0.786,
    geo: {
      crs_wkt: "PROJCS[...]",
      epsg: 32643,
      crs_name: "WGS 84 / UTM zone 43N",
      is_projected: true,
      is_geographic: false,
      axis_unit: "metre",
      transform: [10, 0, 600000, 0, -10, 2000000],
      pixel_size_native_x: 10,
      pixel_size_native_y: 10,
      gsd_x_m: 10,
      gsd_y_m: 10,
      gsd_m: 10,
      gsd_method: "projected-linear-unit",
      bounds_native: [600000, 1992320, 610240, 2000000],
      bounds_wgs84: [69.8, 18.0, 69.9, 18.07],
      centroid_wgs84: [69.85, 18.035],
      area_km2: 78.64,
    },
    bands: [
      { index: 1, label: "blue", role: "blue", role_source: "description", dtype: "uint16", wavelength_nm: 490, stats: null },
      { index: 2, label: "green", role: "green", role_source: "description", dtype: "uint16", wavelength_nm: 560, stats: null },
      { index: 3, label: "red", role: "red", role_source: "description", dtype: "uint16", wavelength_nm: 665, stats: null },
      { index: 4, label: "nir", role: "nir", role_source: "description", dtype: "uint16", wavelength_nm: 842, stats: null },
      { index: 5, label: "swir16", role: "swir16", role_source: "description", dtype: "uint16", wavelength_nm: 1610, stats: null },
      { index: 6, label: "swir22", role: "swir22", role_source: "description", dtype: "uint16", wavelength_nm: 2190, stats: null },
    ],
    resolved_roles: ["blue", "green", "red", "nir", "swir16", "swir22"],
    modality: {
      modality: "optical",
      confidence: 0.95,
      reasons: ["6 named optical bands resolved (blue, green, nir, red, swir16, swir22)"],
    },
    nodata_value: 0,
    nodata_fraction: 0.0123,
    acquisition_date: "2020-03-14T05:31:00Z",
    date_source: "geotiff-tag",
    day_of_year: 74,
    tiling_recommended: false,
    overview_levels: [],
    block_shape: [1024, 1],
    tags: { PLATFORM: "sentinel-2a", INSTRUMENT: "MSI" },
    ingest_notes: [],
  },
};

const emptySession: SessionRecord = {
  session_id: SESSION_ID,
  created_at: "2026-09-21T10:00:00Z",
  updated_at: "2026-09-21T10:00:00Z",
  images: {},
  configuration: "incomplete",
  notes: [],
};

const filledSession: SessionRecord = {
  ...emptySession,
  images: { single: opticalImage },
  configuration: "single",
};

const readyReport: ReadinessReport = {
  session_id: SESSION_ID,
  configuration: "single",
  verdict: "ready",
  checks: [
    {
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
    },
  ],
  refusal_reasons: [],
  requirements: [],
  computed_ms: 96.4,
  common_epsg: 32643,
  overlap_bounds_native: null,
  overlap_bounds_wgs84: null,
  overlap_fraction: null,
  day_delta: null,
  month_of_year_delta: null,
  seasonal_risk: false,
};

const sampleManifest: SampleManifest = {
  generated_at: "2026-09-21T13:35:00Z",
  scenes: {
    water_body: {
      key: "water_body",
      title: "Ukai reservoir",
      description: "A single clear-sky Sentinel-2 scene.",
      demo: "Demo 1 - single-image grounding",
      configuration: "single",
      suggested_queries: ["Highlight the water body referred to in the query"],
      place: "Ukai reservoir, Tapi basin, Gujarat",
      assets: {
        single: {
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
            stac_url: "https://example.invalid/item",
            acquisition_date: "2024-01-12T05:31:00Z",
            cloud_cover_percent: 0,
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
        },
      },
      remedies: [],
      corrects: null,
      remedy_note: "",
    },
  },
};

const toolDescriptors = [
  {
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
    parameters: {},
    produces: [],
    registered: true,
  },
  {
    name: "gis-measure-engine",
    version: "1.0.0",
    implementation: "deterministic",
    summary: "Areas from pixel counts, components, centroids.",
    requirement: {
      band_roles: [],
      any_of_band_roles: [],
      modalities: [],
      configurations: ["single"],
      requires_crs: true,
      description: "Any georeferenced raster.",
    },
    parameters: {},
    produces: [],
    registered: true,
  },
];

function mockApi(
  overrides: { session?: SessionRecord; readiness?: ReadinessReport } = {},
) {
  const session = overrides.session ?? emptySession;
  const readiness = overrides.readiness ?? readyReport;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      let body: unknown = {};

      // Checked before the generic /api/sessions branch, which would otherwise
      // return a session record where a readiness report is expected.
      if (url.includes("/readiness")) {
        body = readiness;
      } else if (url.endsWith("/tools")) {
        body = url.includes("/sessions/")
          ? {
              session_id: SESSION_ID,
              configuration: "single",
              available: ["spectral-index-engine", "gis-measure-engine"],
              unavailable: [],
              declined_registration: {},
            }
          : toolDescriptors;
      } else if (url.includes("/api/samples")) {
        body = sampleManifest;
      } else if (url.includes("/api/health")) {
        body = {
          status: "ok",
          app: "SatQuery AI",
          version: "0.1.0",
          environment: "test",
          components: {},
          capabilities: {
            contract_source: "gemini",
            narration_available: true,
            sar_source: "copernicus-sentinel-1-grd",
            optical_source: "earth-search-sentinel-2-l2a",
            geospatial_compute: true,
          },
        };
      } else if (url.includes("/api/sessions")) {
        body = session;
      }

      return Promise.resolve({
        ok: true,
        status: method === "POST" ? 201 : 200,
        statusText: "OK",
        json: () => Promise.resolve(body),
        text: () => Promise.resolve(JSON.stringify(body)),
      } as Response);
    }),
  );
}

const renderStudio = () =>
  render(
    <MemoryRouter>
      <Studio />
    </MemoryRouter>,
  );

describe("Studio ingest", () => {
  beforeEach(() => mockApi());

  it("offers all three input configurations from the problem statement", async () => {
    renderStudio();
    const group = await screen.findByRole("radiogroup", {
      name: /input configuration/i,
    });
    const options = within(group).getAllByRole("radio");
    expect(options).toHaveLength(3);
    expect(options[0]).toHaveAttribute("aria-checked", "true");
    expect(within(group).getByText("Single image")).toBeInTheDocument();
    expect(within(group).getByText("Optical + SAR")).toBeInTheDocument();
    expect(within(group).getByText("Bi-temporal")).toBeInTheDocument();
  });

  it("shows one upload slot for single image and two for a pair", async () => {
    renderStudio();
    await screen.findByRole("radiogroup", { name: /input configuration/i });
    expect(
      screen.getByRole("button", { name: /Optical, multispectral, or SAR/i }),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole("radio", { name: /Optical \+ SAR/i }));

    expect(
      screen.getByRole("button", { name: /Optical \/ multispectral/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^SAR\./i })).toBeInTheDocument();
  });

  it("switching to bi-temporal asks for two dated acquisitions", async () => {
    renderStudio();
    await screen.findByRole("radiogroup", { name: /input configuration/i });
    await userEvent.click(screen.getByRole("radio", { name: /Bi-temporal/i }));

    expect(
      screen.getByRole("button", { name: /Earlier acquisition/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Later acquisition/i }),
    ).toBeInTheDocument();
  });

  it("prompts for imagery before anything is loaded", async () => {
    renderStudio();
    expect(await screen.findByText(/No imagery loaded yet/i)).toBeInTheDocument();
  });

  it("states that PNG and JPEG are benchmark-only", async () => {
    renderStudio();
    expect(
      await screen.findByText(/accepted for the prescribed benchmark datasets/i),
    ).toBeInTheDocument();
  });

  it("offers the curated sample scenes with their real acquisition dates", async () => {
    renderStudio();
    const gallery = await screen.findByRole("region", { name: /sample scenes/i });
    expect(within(gallery).getByText("Ukai reservoir")).toBeInTheDocument();
    expect(within(gallery).getByText(/12 Jan 2024/)).toBeInTheDocument();
    expect(
      within(gallery).getByRole("button", { name: /Load this scene/i }),
    ).toBeEnabled();
  });
});

describe("Studio metadata display", () => {
  beforeEach(() => mockApi({ session: filledSession }));

  it("renders measured values from the file, not from the filename", async () => {
    renderStudio();
    const panel = await screen.findByRole("region", {
      name: /Measured metadata for bhuj_2020\.tif/i,
    });
    expect(within(panel).getByText("EPSG:32643")).toBeInTheDocument();
    expect(within(panel).getByText("10 m")).toBeInTheDocument();
    expect(within(panel).getByText("1,024 \u00d7 768 px")).toBeInTheDocument();
    expect(within(panel).getByText("6 \u00d7 uint16")).toBeInTheDocument();
    expect(within(panel).getByText("78.64 km\u00b2")).toBeInTheDocument();
    // Resolution carries the derivation method, not just the number.
    expect(within(panel).getByText(/from affine transform/i)).toBeInTheDocument();
  });

  it("shows the derived configuration badge", async () => {
    renderStudio();
    await waitFor(() =>
      expect(screen.getByText("Single image", { selector: "span" })).toBeInTheDocument(),
    );
  });

  it("renders a chip per band with its centre wavelength", async () => {
    renderStudio();
    const list = await screen.findByLabelText("Resolved band roles");
    const chips = within(list).getAllByRole("listitem");
    expect(chips).toHaveLength(6);
    expect(within(list).getByText("665nm")).toBeInTheDocument();
    expect(within(list).getByText("842nm")).toBeInTheDocument();
  });

  it("explains why the modality was inferred", async () => {
    renderStudio();
    expect(
      await screen.findByText(/6 named optical bands resolved/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/95% confidence/i)).toBeInTheDocument();
  });

  it("states how the preview was composited", async () => {
    renderStudio();
    expect(
      await screen.findByText("True colour: R=red, G=green, B=blue"),
    ).toBeInTheDocument();
  });

  it("labels the preview image for screen readers", async () => {
    renderStudio();
    const image = await screen.findByAltText(/Preview of bhuj_2020.tif/i);
    expect(image).toHaveAttribute("src", expect.stringContaining("/thumb/single"));
  });
});

describe("Studio unmeasured values", () => {
  it("renders an em dash where a value could not be measured", async () => {
    const noGeo: SessionRecord = {
      ...filledSession,
      images: {
        single: {
          ...opticalImage,
          metadata: {
            ...opticalImage.metadata,
            geo: {
              ...opticalImage.metadata.geo,
              epsg: null,
              crs_name: null,
              gsd_m: null,
              gsd_x_m: null,
              gsd_y_m: null,
              gsd_method: null,
              area_km2: null,
              centroid_wgs84: null,
            },
            format_class: "benchmark_raster",
            ingest_notes: ["File carries no CRS. Accepted for benchmark imagery."],
          },
        },
      },
    };
    mockApi({ session: noGeo });
    renderStudio();

    // An absent measurement must read as absent, never as zero.
    const dashes = await screen.findAllByText("\u2014");
    expect(dashes.length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText(/carries no CRS/i)).toBeInTheDocument();
  });
});
