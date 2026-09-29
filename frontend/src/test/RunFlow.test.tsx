/**
 * The approve-and-run path, end to end in the browser environment.
 *
 * This is the flow that went blank, and it went blank because nothing exercised
 * it: jsdom ships no EventSource, so the one part of the app that depends on a
 * server-sent stream had no test at all. A component fault anywhere downstream of
 * the run took the entire page with it, because React unmounts the tree on an
 * unhandled render error and there was nothing to catch it.
 *
 * So there are two kinds of test here. One drives a complete run and asserts the
 * result actually renders. The other proves the page survives a component that
 * throws, because the next fault should cost a panel rather than the whole
 * workspace.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Studio } from "../pages/Studio";
import { FakeEventSource, installMatchMedia } from "./setup";
import type {
  AnalysisContract,
  IngestedImage,
  MapLayer,
  RunTrace,
  SessionRecord,
  StageId,
  StageStatus,
  TraceStage,
  TraceStep,
} from "../lib/types";

const SESSION_ID = "a".repeat(32);
const RUN_ID = "b".repeat(32);
const CONTRACT_HASH = "c".repeat(32);

// ---------------------------------------------------------------------------
// Fixtures: a session that has imagery, and a run that finished with a verdict
// ---------------------------------------------------------------------------

const image = {
  role: "single",
  stored_filename: "single.tif",
  original_filename: "ukai_2024.tif",
  content_type: "image/tiff",
  sha256: "d".repeat(64),
  ingested_at: "2026-09-21T10:00:00Z",
  thumbnail_available: true,
  thumbnail_recipe: "True colour",
  ingest_ms: 300,
  metadata: {
    original_filename: "ukai_2024.tif",
    size_bytes: 3_100_000,
    driver: "GTiff",
    format_class: "geospatial" as const,
    width: 512,
    height: 512,
    band_count: 6,
    dtype: "uint16",
    megapixels: 0.26,
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
      bounds_native: [600000, 1994880, 605120, 2000000],
      bounds_wgs84: [73.56, 21.22, 73.61, 21.27],
      centroid_wgs84: [73.586, 21.245],
      area_km2: 26.21,
    },
    bands: [],
    resolved_roles: ["blue", "green", "red", "nir", "swir16", "swir22"],
    modality: { modality: "optical" as const, confidence: 0.95, reasons: [] },
    nodata_value: 0,
    nodata_fraction: 0,
    acquisition_date: "2024-01-12T05:31:00Z",
    date_source: "geotiff-tag" as const,
    day_of_year: 12,
    tiling_recommended: false,
    overview_levels: [],
    block_shape: [512, 512],
    tags: {},
    ingest_notes: [],
  },
} as unknown as IngestedImage;

const session: SessionRecord = {
  session_id: SESSION_ID,
  created_at: "2026-09-21T10:00:00Z",
  updated_at: "2026-09-21T10:00:00Z",
  images: { single: image },
  configuration: "single",
  notes: [],
  sample_key: "water_body",
};

const contract: AnalysisContract = {
  session_id: SESSION_ID,
  contract_hash: CONTRACT_HASH,
  query: "Highlight the water body referred to in the query",
  claim: "Water is present in this scene and can be delineated.",
  task_type: "region_grounding",
  configuration: "single",
  required_modalities: ["optical"],
  acts_on: ["single"],
  target_classes: ["water"],
  change_direction: "unspecified",
  metrics_requested: ["area_km2"],
  confounders: [
    {
      kind: "cloud_shadow",
      label: "Cloud and shadow",
      question: "Could cloud be hiding the surface?",
      reason: "0.0% of the scene is cloud by its own classification band.",
    },
  ],
  tools: [
    {
      tool: "grounding-engine",
      version: "1.0.0",
      rationale: "locate 'water'",
      parameters: { target: "water" },
      order: 0,
    },
  ],
  expected_outputs: ["a region on the map"],
  source: "offline-rule-router",
  model: null,
  interpretation_note: "Read as a request to locate water and measure it.",
  repairs: [],
  rejections: [],
  generated_at: "2026-09-21T10:00:00Z",
  duration_ms: 12,
  fallback_reason: null,
};

const measurement = {
  key: "grounded_area_km2",
  label: "Grounded area",
  value: 9.6547,
  unit: "km2",
  formula: "pixel_count * pixel_area_m2 / 1e6",
  inputs: { pixel_count: 96547 },
  source_tool: "grounding-engine",
  source_version: "1.0.0",
  method: "conventional threshold",
  applies_to: ["single" as const],
  precision: 4,
};

function stage(
  id: StageId,
  status: StageStatus,
  steps: TraceStep[] = [],
): TraceStage {
  return {
    id,
    label: id.replace(/_/g, " "),
    purpose: "…",
    status,
    steps,
    notes: [],
    unavailable_note: "",
    started_at: "2026-09-21T10:00:00Z",
    finished_at: "2026-09-21T10:00:01Z",
    duration_ms: 120,
  };
}

const step: TraceStep = {
  id: "run_specialists:grounding-engine",
  stage: "run_specialists",
  label: "grounding-engine",
  detail: "1 mask, 6 measurements",
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
  duration_ms: 900,
};

/** A finished run carrying everything the panels read, including the newer fields. */
function finishedTrace(overrides: Partial<RunTrace> = {}): RunTrace {
  return {
    run_id: RUN_ID,
    session_id: SESSION_ID,
    contract_hash: CONTRACT_HASH,
    query: contract.query,
    claim: contract.claim,
    task_type: "region_grounding",
    configuration: "single",
    status: "completed",
    stages: [
      stage("verify_inputs", "ok"),
      stage("accept_contract", "ok"),
      stage("run_specialists", "ok", [step]),
      stage("test_confounders", "ok"),
      stage("compare_evidence", "ok"),
      stage("resolve_verdict", "ok"),
      stage("compose_packet", "ok"),
    ],
    tool_runs: [
      {
        tool: "grounding-engine",
        version: "1.0.0",
        implementation: "deterministic",
        ok: true,
        skipped_reason: null,
        parameters: { target: "water" },
        measurements: [measurement],
        masks: [],
        notes: [],
        duration_ms: 900,
      },
    ],
    measurements: [measurement],
    masks: [],
    confounders: [],
    verdict: null,
    ledger: null,
    remedies: null,
    disagreement: null,
    packet: null,
    started_at: "2026-09-21T10:00:00Z",
    finished_at: "2026-09-21T10:00:02Z",
    duration_ms: 2100,
    error: null,
    refusal_reasons: [],
    ...overrides,
  };
}

const layer: MapLayer = {
  key: "grounded_water",
  label: "Water",
  description: "MNDWI above 0",
  kind: "mask",
  png_url: `/api/sessions/${SESSION_ID}/runs/${RUN_ID}/layers/g.png`,
  geojson_url: null,
  bounds_wgs84: [73.56, 21.22, 73.61, 21.27],
  colour: "#22d3ee",
  area_km2: 9.6547,
  pixel_count: 96547,
  applies_to: ["single"],
};

function mockApi(trace: RunTrace, layers: MapLayer[] = [layer]) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      let body: unknown = {};

      if (url.includes("/readiness")) {
        body = {
          session_id: SESSION_ID,
          configuration: "single",
          verdict: "ready",
          checks: [],
          refusal_reasons: [],
          requirements: [],
          computed_ms: 10,
          common_epsg: 32643,
          overlap_bounds_native: null,
          overlap_bounds_wgs84: null,
          overlap_fraction: null,
          day_delta: null,
          month_of_year_delta: null,
          seasonal_risk: false,
        };
      } else if (url.includes("/packet")) {
        body = trace.packet ?? {};
      } else if (url.includes("/layers")) {
        body = layers;
      } else if (url.includes("/runs")) {
        body = trace;
      } else if (url.endsWith("/tools")) {
        body = url.includes("/sessions/")
          ? {
              session_id: SESSION_ID,
              configuration: "single",
              available: ["grounding-engine"],
              unavailable: [],
              declined_registration: {},
            }
          : [];
      } else if (url.includes("/api/samples")) {
        body = { generated_at: "2026-09-21T10:00:00Z", scenes: {} };
      } else if (url.includes("/api/health")) {
        body = {
          status: "ok",
          app: "SatQuery AI",
          version: "0.1.0",
          environment: "test",
          components: {},
          capabilities: {},
        };
      } else if (url.includes("/contract")) {
        body = contract;
      } else if (url.includes("/api/sessions")) {
        body = session;
      }

      return Promise.resolve({
        ok: true,
        status: method === "POST" ? 202 : 200,
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

/** Get to a drafted contract, which is where the approve button lives. */
async function draftContract() {
  renderStudio();
  await screen.findByRole("radiogroup", { name: /input configuration/i });
  const input = await screen.findByLabelText(/your question about the loaded/i);
  await userEvent.type(input, "Highlight the water body");
  await userEvent.click(screen.getByRole("button", { name: /Draft contract/i }));
  return screen.findByRole("button", { name: /Approve and run/i });
}

describe("approve and run", () => {
  beforeEach(() => {
    FakeEventSource.instances.length = 0;
  });

  it("renders the run once the stream reports it finished", async () => {
    const trace = finishedTrace();
    mockApi(trace);

    const approve = await draftContract();
    await userEvent.click(approve);

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    const stream = FakeEventSource.instances[0];
    expect(stream.url).toContain(`/runs/${RUN_ID}/stream`);

    stream.emit({
      type: "run.finished",
      run_id: RUN_ID,
      seq: 1,
      at: "2026-09-21T10:00:02Z",
      message: "completed",
      trace,
    });

    // The page has to still be here, with the run on it.
    const graph = await screen.findByLabelText(/pipeline/i);
    expect(within(graph).getByText(/compose packet/i)).toBeInTheDocument();
    expect(screen.getAllByText(/grounding-engine/).length).toBeGreaterThan(0);
  });

  it("renders the run with animation enabled, as a browser does", async () => {
    // Every other test in this suite runs with reduced motion forced on, which
    // skips the contract card's reveal gate and every framer-motion transition.
    // The browser does neither, so this is the path a user actually takes and the
    // only one where the animated code runs at all. The global afterEach puts the
    // reduced-motion default back, so this does not leak into the next test.
    installMatchMedia(false);

    const trace = finishedTrace();
    mockApi(trace);

    renderStudio();
    await screen.findByRole("radiogroup", { name: /input configuration/i });
    const input = await screen.findByLabelText(/your question about the loaded/i);
    await userEvent.type(input, "Highlight the water body");
    await userEvent.click(
      screen.getByRole("button", { name: /Draft contract/i }),
    );

    // The claim is typed out before the body opens, so the approve button takes a
    // moment to appear rather than being there immediately.
    const approve = await screen.findByRole(
      "button",
      { name: /Approve and run/i },
      { timeout: 5000 },
    );
    await userEvent.click(approve);

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({
      type: "run.finished",
      run_id: RUN_ID,
      seq: 1,
      at: "2026-09-21T10:00:02Z",
      message: "completed",
      trace,
    });

    expect(
      await screen.findByLabelText(/pipeline/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument();
  });

  it("survives the stream closing without a finished event", async () => {
    const trace = finishedTrace();
    mockApi(trace);

    const approve = await draftContract();
    await userEvent.click(approve);
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));

    // The server closes the stream when the run ends, which the browser reports
    // as an error. The trace is then read back over plain HTTP.
    FakeEventSource.instances[0].fail();

    await waitFor(() =>
      expect(screen.getByLabelText(/pipeline/i)).toBeInTheDocument(),
    );
  });

  it("renders a verdict, its remedies, and its packet together", async () => {
    const trace = finishedTrace({
      task_type: "claim_investigation",
      verdict: {
        label: "refuted",
        claim: "Vegetation decreased between the two dates.",
        asserted_direction: "decreased",
        measured_direction: "unchanged",
        confidence: 0.71,
        confidence_components: [
          {
            name: "measurement_quality",
            label: "How little the answer depends on its threshold",
            measured: 0.991,
            contribution: 0.297,
            weight: 0.3,
            best: 0.3,
            rationale: "Nudging the threshold barely moves the answer.",
            lever: "A trained classifier.",
          },
        ],
        evidence: [
          {
            measurement_key: "grounded_area_km2",
            label: "Grounded area",
            value: 9.6547,
            unit: "km2",
            display: "9.6547 km2",
            direction: "supports",
            relevance: "It is the quantity the claim is about.",
            statement: "The region covers 9.6547 km2.",
            source_tool: "grounding-engine",
            source_version: "1.0.0",
            formula: "pixel_count * pixel_area_m2 / 1e6",
            independent: false,
            weight: 1,
          },
        ],
        consistency: [],
        surviving_confounders: [],
        requirements: [],
        reasoning: "The measured area does not support the claim.",
        narrative: null,
        narrative_source: "template",
        narrative_audit: null,
        what_would_change_it: ["An acquisition from the same season."],
      },
      ledger: null,
      remedies: {
        offers: [
          {
            confounder: "seasonality",
            confounder_label: "Seasonal phenology",
            requirement: "an acquisition from the same part of the season",
            sample_key: "seasonal_farmland_same_season",
            title: "Punjab, one year apart in the same season",
            place: "North of Moga, Punjab",
            configuration: "bi_temporal_pair",
            why: "Phenology cannot account for a difference.",
            corrects_current: true,
            suggested_query: "Did vegetation decrease?",
          },
        ],
        unmet: [],
      },
      packet: {
        run_id: RUN_ID,
        session_id: SESSION_ID,
        generated_at: "2026-09-21T10:00:03Z",
        query: contract.query,
        claim: contract.claim,
        verdict_label: "refuted",
        verdict_text: "Refuted at 71% confidence",
        confidence: 0.71,
        files: [
          {
            filename: "report.pdf",
            role: "report",
            label: "Readable report",
            description: "The finding and everything behind it.",
            media_type: "application/pdf",
            size_bytes: 478_236,
            sha256: "e".repeat(64),
          },
        ],
        figures: [],
        audit: {
          checked: 350,
          traced: 350,
          untraceable: [],
          passed: true,
          note: "All 350 figures are attributed.",
        },
        archive: null,
        notes: [],
        omissions: ["The source imagery itself."],
      },
    });
    mockApi(trace);

    const approve = await draftContract();
    await userEvent.click(approve);
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));

    FakeEventSource.instances[0].emit({
      type: "run.finished",
      run_id: RUN_ID,
      seq: 1,
      at: "2026-09-21T10:00:02Z",
      message: "completed",
      trace,
    });

    expect(await screen.findByLabelText("Verdict")).toBeInTheDocument();
    expect(screen.getByText(/Refuted/)).toBeInTheDocument();
    expect(
      await screen.findByLabelText(/Imagery that would settle this/i),
    ).toBeInTheDocument();
    expect(await screen.findByLabelText(/Evidence packet/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /^Report/ })).toBeInTheDocument();
  });
});

/**
 * What the page shows while the run is still going.
 *
 * Two real failures, both reported from the same screen. A trace in flight does
 * not carry a verdict, a packet or a disagreement report, and the stream used to
 * drop null fields entirely, so those keys arrived undefined. Two panels guarded
 * with `=== null`, walked an undefined past the guard, and threw:
 *
 *   Cannot destructure property 'candidate_area_km2' of 'disagreement'
 *   Cannot read properties of undefined (reading 'files')
 *
 * The result was a run that worked perfectly, wearing two red error panels.
 *
 * The fix has two halves and both are tested here: findings are not rendered
 * until the run settles, and a panel handed nothing renders nothing rather than
 * throwing, whatever shape the nothing arrives in.
 */
describe("while the run is still going", () => {
  beforeEach(() => {
    FakeEventSource.instances.length = 0;
  });

  /** Mid-run: stages underway, and none of the result fields populated yet. */
  function runningTrace(): RunTrace {
    return finishedTrace({
      status: "running",
      finished_at: null,
      stages: [
        stage("verify_inputs", "ok"),
        stage("accept_contract", "ok"),
        stage("run_specialists", "running"),
        stage("test_confounders", "pending"),
        stage("compare_evidence", "pending"),
        stage("resolve_verdict", "pending"),
        stage("compose_packet", "pending"),
      ],
    });
  }

  async function startRunning() {
    const approve = await draftContract();
    await userEvent.click(approve);
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({
      type: "stage.started",
      run_id: RUN_ID,
      seq: 1,
      at: "2026-09-21T10:00:01Z",
      message: "Run specialists",
      trace: runningTrace(),
    });
  }

  it("shows the pipeline and the log, and holds the findings back", async () => {
    mockApi(finishedTrace());
    await startRunning();

    // Live, because they are built to be read as it happens.
    expect(await screen.findByLabelText(/pipeline/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Execution trace/i)).toBeInTheDocument();

    // Held back, because a stage that has not run has no finding.
    expect(await screen.findByLabelText(/Findings pending/i)).toBeInTheDocument();
    expect(screen.queryByLabelText("Map workspace")).toBeNull();
    expect(screen.queryByLabelText("Evidence disagreement")).toBeNull();
    expect(screen.queryByLabelText("Evidence packet")).toBeNull();
    expect(screen.queryByLabelText("Verdict")).toBeNull();
  });

  it("names the stage in flight rather than just spinning", async () => {
    mockApi(finishedTrace());
    await startRunning();

    const pending = await screen.findByLabelText(/Findings pending/i);
    expect(pending).toHaveTextContent(/Run specialists/i);
    expect(pending).toHaveTextContent(/appear together once the run finishes/i);
  });

  it("shows no panel error while running", async () => {
    // The symptom that was reported. The boundary renders this heading when a
    // panel throws, so its absence is the assertion.
    mockApi(finishedTrace());
    await startRunning();

    await screen.findByLabelText(/Findings pending/i);
    expect(screen.queryByText(/could not be displayed/i)).toBeNull();
    expect(screen.queryByText(/Cannot destructure/i)).toBeNull();
    expect(screen.queryByText(/Cannot read properties/i)).toBeNull();
  });

  it("releases every finding together once the run settles", async () => {
    const trace = finishedTrace({
      disagreement: {
        candidate_area_km2: 25.13,
        agree_area_km2: 11.88,
        disagree_area_km2: 13.25,
        uncertain_area_km2: 0,
        conflicting_fraction: 0.5273,
        methods: ["Spectral index (NDWI)", "Learned probe"],
        regions: [],
        applies_to: ["single"],
        agree_layer: "evidence_agree",
        disagree_layer: "evidence_disagree",
        uncertain_layer: "evidence_uncertain",
        notes: [],
      },
    });
    mockApi(trace);
    await startRunning();
    await screen.findByLabelText(/Findings pending/i);

    FakeEventSource.instances[0].emit({
      type: "run.finished",
      run_id: RUN_ID,
      seq: 2,
      at: "2026-09-21T10:00:02Z",
      message: "completed",
      trace,
    });

    expect(await screen.findByLabelText("Evidence disagreement")).toBeInTheDocument();
    expect(screen.getByLabelText("Map workspace")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Findings pending/i)).toBeNull();
    expect(screen.queryByText(/could not be displayed/i)).toBeNull();
  });

  it("survives a finished trace whose optional fields are absent entirely", async () => {
    // Exactly the payload the old stream produced: the keys are not null, they
    // are missing. A panel must treat that as "nothing to show", not crash.
    const trace = finishedTrace();
    const pruned = { ...trace } as Record<string, unknown>;
    for (const key of ["disagreement", "packet", "verdict", "remedies", "ledger"]) {
      delete pruned[key];
    }
    mockApi(trace);

    const approve = await draftContract();
    await userEvent.click(approve);
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({
      type: "run.finished",
      run_id: RUN_ID,
      seq: 1,
      at: "2026-09-21T10:00:02Z",
      message: "completed",
      trace: pruned,
    });

    // The run renders, and nothing on the page reports a crash.
    expect(await screen.findByLabelText(/pipeline/i)).toBeInTheDocument();
    expect(screen.queryByText(/could not be displayed/i)).toBeNull();
    expect(screen.queryByLabelText("Evidence disagreement")).toBeNull();
    expect(screen.queryByLabelText("Evidence packet")).toBeNull();
  });
});
