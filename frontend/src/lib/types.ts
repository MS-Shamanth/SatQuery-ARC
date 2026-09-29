/** Shared response shapes, mirroring the backend pydantic models. */

export type ProbeStatus =
  | "ok"
  /** Reachable and authenticated, but out of allowance. Not a fault. */
  | "limited"
  | "invalid"
  | "missing"
  | "error"
  | "unknown"
  | "skipped";

export interface ComponentProbe {
  component: string;
  status: ProbeStatus;
  detail: string;
  checked_at: number | null;
  extra?: Record<string, unknown>;
}

export interface Capabilities {
  /** Who plans contracts right now. "gemini" only appears in older cached traces. */
  contract_source: "language-model" | "gemini" | "offline-rule-router" | "cache";
  narration_available: boolean;
  /**
   * The provider in use, or "none". A cross-vendor chain reports as
   * "mistral+openrouter", so this is a plain string rather than a union.
   */
  llm_provider: string;
  /** Models that actually generated, qualified by provider in a chain. */
  llm_models: string[];
  sar_source:
    | "copernicus-sentinel-1-grd"
    | "aws-requester-pays-sentinel-1-grd"
    | "simulated-sar";
  optical_source: "earth-search-sentinel-2-l2a" | "synthetic";
  geospatial_compute: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "down";
  app: string;
  version: string;
  environment: string;
  components: Record<string, ComponentProbe>;
  capabilities: Capabilities;
}

export interface ProvenanceResponse {
  problem_statement: { id: string; title: string; organisation: string };
  training_dataset: {
    name: string;
    arxiv_id: string;
    url: string;
    image_pairs: number;
    text_annotations: number;
    modalities: string;
    annotation_types: string;
  };
  evaluation_benchmarks: { name: string; used_for: string }[];
  data_sources: { name: string; access: string; url: string; note: string }[];
  never_guess_rule: string;
}

// ---------------------------------------------------------------------------
// Raster ingest (Task 2)
// ---------------------------------------------------------------------------

export type BandRole =
  | "coastal" | "blue" | "green" | "red"
  | "rededge1" | "rededge2" | "rededge3"
  | "nir" | "nir08" | "nir09" | "cirrus"
  | "swir16" | "swir22" | "scl"
  | "vv" | "vh" | "hh" | "hv"
  | "gray" | "alpha" | "unknown";

export type Modality = "optical" | "sar" | "unknown";

export type ImageRole = "single" | "optical" | "sar" | "date_a" | "date_b";

export type InputConfiguration =
  | "single"
  | "cross_modal_pair"
  | "bi_temporal_pair"
  | "incomplete";

export type FormatClass = "geospatial" | "benchmark_raster";

export type DateSource =
  | "geotiff-tag" | "manifest" | "filename" | "user" | "unknown";

export interface BandStats {
  minimum: number;
  maximum: number;
  mean: number;
  stddev: number;
  valid_pixels: number;
  nodata_pixels: number;
  nodata_fraction: number;
  sample_fraction: number;
  percentile_2: number | null;
  percentile_98: number | null;
}

export interface BandInfo {
  index: number;
  label: string;
  role: BandRole;
  role_source: string;
  dtype: string;
  wavelength_nm: number | null;
  stats: BandStats | null;
}

export interface GeoReference {
  crs_wkt: string | null;
  epsg: number | null;
  crs_name: string | null;
  is_projected: boolean;
  is_geographic: boolean;
  axis_unit: string | null;
  transform: number[];
  pixel_size_native_x: number | null;
  pixel_size_native_y: number | null;
  gsd_x_m: number | null;
  gsd_y_m: number | null;
  gsd_m: number | null;
  gsd_method: string | null;
  bounds_native: number[] | null;
  bounds_wgs84: number[] | null;
  centroid_wgs84: number[] | null;
  area_km2: number | null;
}

export interface ModalityInference {
  modality: Modality;
  confidence: number;
  reasons: string[];
}

export interface RasterMetadata {
  original_filename: string;
  size_bytes: number;
  driver: string;
  format_class: FormatClass;
  width: number;
  height: number;
  band_count: number;
  dtype: string;
  megapixels: number;
  geo: GeoReference;
  bands: BandInfo[];
  resolved_roles: BandRole[];
  modality: ModalityInference;
  nodata_value: number | null;
  nodata_fraction: number;
  acquisition_date: string | null;
  date_source: DateSource;
  day_of_year: number | null;
  tiling_recommended: boolean;
  overview_levels: number[];
  block_shape: number[] | null;
  tags: Record<string, string>;
  ingest_notes: string[];
}

export interface IngestedImage {
  role: ImageRole;
  stored_filename: string;
  original_filename: string;
  content_type: string | null;
  sha256: string;
  ingested_at: string;
  metadata: RasterMetadata;
  thumbnail_available: boolean;
  thumbnail_recipe: string | null;
  ingest_ms: number;
}

export interface SessionRecord {
  session_id: string;
  created_at: string;
  updated_at: string;
  images: Partial<Record<ImageRole, IngestedImage>>;
  configuration: InputConfiguration;
  notes: string[];
  /** Set when the session was created by loading a sample scene (Task 13). */
  sample_key?: string | null;
}

// ---------------------------------------------------------------------------
// Readiness gate (Task 3)
// ---------------------------------------------------------------------------

export type CheckStatus = "pass" | "warn" | "fail" | "not_applicable";

export type ReadinessVerdict = "ready" | "ready_with_warnings" | "refused";

export interface ReadinessCheck {
  id: string;
  label: string;
  status: CheckStatus;
  measured: string;
  measured_numeric: number | null;
  unit: string | null;
  threshold: string | null;
  message: string;
  method: string | null;
  applies_to: ImageRole[];
}

export interface DataRequirement {
  what: string;
  why: string;
}

export interface ReadinessReport {
  session_id: string;
  configuration: InputConfiguration;
  verdict: ReadinessVerdict;
  checks: ReadinessCheck[];
  refusal_reasons: string[];
  requirements: DataRequirement[];
  computed_ms: number;
  common_epsg: number | null;
  overlap_bounds_native: number[] | null;
  overlap_bounds_wgs84: number[] | null;
  overlap_fraction: number | null;
  day_delta: number | null;
  month_of_year_delta: number | null;
  seasonal_risk: boolean;
}

// ---------------------------------------------------------------------------
// Sample scene library (Task 4)
// ---------------------------------------------------------------------------

export interface SampleProvenance {
  source: string;
  collection: string | null;
  stac_item_id: string | null;
  stac_url: string | null;
  acquisition_date: string | null;
  cloud_cover_percent: number | null;
  platform: string | null;
  instrument: string | null;
  bands: string[];
  gsd_m: number | null;
  epsg: number | null;
  licence: string | null;
  attribution: string | null;
  is_simulated: boolean;
  simulation_note: string | null;
  fetched_at: string | null;
}

export interface SampleAssetRecord {
  role: ImageRole;
  filename: string;
  sha256: string;
  size_bytes: number;
  width: number;
  height: number;
  band_count: number;
  provenance: SampleProvenance;
}

export interface SampleScene {
  key: string;
  title: string;
  description: string;
  demo: string;
  configuration: InputConfiguration;
  suggested_queries: string[];
  place: string;
  assets: Partial<Record<ImageRole, SampleAssetRecord>>;
  // Confounder kinds this scene removes by construction (Task 13).
  remedies: string[];
  // The sample key whose open objection this scene exists to settle.
  corrects: string | null;
  remedy_note: string;
}

export interface SampleManifest {
  generated_at: string;
  scenes: Record<string, SampleScene>;
}

// ---------------------------------------------------------------------------
// Tools and measurements (Task 5)
// ---------------------------------------------------------------------------

export type ToolImplementation = "deterministic" | "learned" | "llm-narration";

export interface Measurement {
  key: string;
  label: string;
  value: number;
  unit: string;
  formula: string;
  inputs: Record<string, unknown>;
  source_tool: string;
  source_version: string;
  method: string | null;
  applies_to: ImageRole[];
  precision: number;
}

export interface ToolRequirementSpec {
  band_roles: BandRole[];
  any_of_band_roles: BandRole[][];
  modalities: Modality[];
  configurations: InputConfiguration[];
  requires_crs: boolean;
  description: string;
}

export interface ToolDescriptor {
  name: string;
  version: string;
  implementation: ToolImplementation;
  summary: string;
  requirement: ToolRequirementSpec;
  parameters: Record<string, unknown>;
  produces: string[];
  registered: boolean;
}

export interface MaskSummary {
  key: string;
  label: string;
  description: string;
  pixel_count: number;
  total_pixels: number;
  coverage_fraction: number;
  area_km2: number | null;
  epsg: number | null;
  bounds_wgs84: number[] | null;
  stored_filename: string | null;
  applies_to: ImageRole[];
  threshold: number | null;
  threshold_method: string | null;
  separability: number | null;
}

export interface ToolRun {
  tool: string;
  version: string;
  implementation: ToolImplementation;
  ok: boolean;
  skipped_reason: string | null;
  parameters: Record<string, unknown>;
  measurements: Measurement[];
  masks: MaskSummary[];
  notes: string[];
  duration_ms: number;
}

export interface SessionToolAvailability {
  session_id: string;
  configuration: InputConfiguration;
  available: string[];
  unavailable: { tool: string; reason: string }[];
  declined_registration: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Analysis Contract (Task 6)
// ---------------------------------------------------------------------------

export type ContractTaskType =
  | "scene_description"
  | "visual_question"
  | "region_grounding"
  | "change_detection"
  | "change_question"
  | "cross_modal_extraction"
  | "claim_investigation";

export type ConfounderKind =
  | "seasonality"
  | "misregistration"
  | "cloud_shadow"
  | "radiometry"
  | "sensor_mismatch"
  | "sar_specific"
  | "illumination_terrain";

export type ChangeDirection =
  | "increased"
  | "decreased"
  | "unchanged"
  | "unspecified";

export type ContractSource =
  | "language-model"
  // Produced only by contracts cached before the provider abstraction existed.
  | "gemini"
  | "offline-rule-router"
  | "cache";

export interface ToolPlan {
  tool: string;
  version: string;
  rationale: string;
  parameters: Record<string, unknown>;
  order: number;
}

export interface ConfounderPlan {
  kind: ConfounderKind;
  label: string;
  question: string;
  reason: string;
}

export interface ContractRepair {
  field: string;
  detail: string;
}

export interface ContractRejection {
  field: string;
  value: string;
  reason: string;
}

export interface AnalysisContract {
  session_id: string;
  query: string;
  claim: string;
  task_type: ContractTaskType;
  configuration: InputConfiguration;
  required_modalities: Modality[];
  acts_on: ImageRole[];
  target_classes: string[];
  change_direction: ChangeDirection;
  metrics_requested: string[];
  confounders: ConfounderPlan[];
  tools: ToolPlan[];
  expected_outputs: string[];
  source: ContractSource;
  model: string | null;
  interpretation_note: string;
  repairs: ContractRepair[];
  rejections: ContractRejection[];
  contract_hash: string;
  generated_at: string;
  duration_ms: number;
  fallback_reason: string | null;
}

// ---------------------------------------------------------------------------
// Execution trace (Task 7)
// ---------------------------------------------------------------------------

export type StageId =
  | "verify_inputs"
  | "accept_contract"
  | "run_specialists"
  | "test_confounders"
  | "compare_evidence"
  | "resolve_verdict"
  | "compose_packet";

export type StageStatus =
  | "pending"
  | "running"
  | "ok"
  | "partial"
  | "skipped"
  | "failed"
  | "not_built";

export type RunStatus =
  | "running"
  | "completed"
  | "refused"
  | "failed"
  | "cancelled";

export interface TraceStep {
  id: string;
  stage: StageId;
  label: string;
  detail: string;
  status: StageStatus;
  tool: string | null;
  version: string | null;
  implementation: ToolImplementation | null;
  parameters: Record<string, unknown>;
  applies_to: ImageRole[];
  measurement_keys: string[];
  mask_keys: string[];
  notes: string[];
  reason: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number;
}

export interface TraceStage {
  id: StageId;
  label: string;
  purpose: string;
  status: StageStatus;
  steps: TraceStep[];
  notes: string[];
  unavailable_note: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number;
}

export interface RunTrace {
  run_id: string;
  session_id: string;
  contract_hash: string;
  query: string;
  claim: string;
  task_type: ContractTaskType;
  configuration: InputConfiguration;
  status: RunStatus;
  stages: TraceStage[];
  tool_runs: ToolRun[];
  measurements: Measurement[];
  masks: MaskSummary[];
  confounders: ConfounderTest[];
  verdict: Verdict | null;
  ledger: EvidenceLedger | null;
  remedies: RemedySet | null;
  disagreement: DisagreementReport | null;
  packet: EvidencePacket | null;
  started_at: string;
  finished_at: string | null;
  duration_ms: number;
  error: string | null;
  refusal_reasons: string[];
}

export type RunEventType =
  | "run.started"
  | "stage.started"
  | "stage.finished"
  | "step.started"
  | "step.finished"
  | "log"
  | "run.finished";

export interface RunEvent {
  type: RunEventType;
  run_id: string;
  seq: number;
  at: string;
  message: string;
  stage?: TraceStage;
  step?: TraceStep;
  tool_run?: ToolRun;
  trace?: RunTrace;
}

export interface MapLayer {
  key: string;
  label: string;
  description: string;
  kind: "base" | "mask";
  png_url: string;
  geojson_url: string | null;
  /** [west, south, east, north] in degrees, from a real reprojection. */
  bounds_wgs84: [number, number, number, number];
  colour: string;
  area_km2: number | null;
  pixel_count: number | null;
  applies_to: ImageRole[];
}

// ---------------------------------------------------------------------------
// Confounder tests (Task 10)
// ---------------------------------------------------------------------------

export type ConfounderVerdict =
  | "ruled_out"
  | "plausible"
  | "likely"
  | "not_tested";

export interface ConfounderTest {
  kind: ConfounderKind;
  label: string;
  question: string;
  verdict: ConfounderVerdict;
  measured: string;
  measured_numeric: number | null;
  unit: string | null;
  threshold: string;
  formula: string;
  inputs: Record<string, unknown>;
  method: string;
  explanation: string;
  /** Present when the alternative survived: the data that would settle it. */
  requirement: string | null;
  applies_to: ImageRole[];
  duration_ms: number;
}

// ---------------------------------------------------------------------------
// Evidence ledger and verdict (Task 11)
// ---------------------------------------------------------------------------

export type VerdictLabel =
  | "supported"
  | "refuted"
  | "inconclusive"
  | "unanswerable";

export type EvidenceDirection = "supports" | "refutes" | "neutral";

export interface EvidenceItem {
  measurement_key: string;
  label: string;
  value: number;
  unit: string;
  display: string;
  direction: EvidenceDirection;
  relevance: string;
  statement: string;
  source_tool: string;
  source_version: string;
  formula: string;
  independent: boolean;
  weight: number;
}

export interface ConsistencyCheck {
  quantity: string;
  label: string;
  first_key: string;
  first_value: number;
  second_key: string;
  second_value: number;
  unit: string;
  relative_difference: number;
  tolerance: number;
  agrees: boolean;
  method: string;
  explanation: string;
}

export interface ConfidenceComponent {
  name: string;
  label: string;
  measured: number;
  contribution: number;
  weight: number;
  best: number;
  rationale: string;
  lever: string;
}

export interface NumericAudit {
  checked: number;
  traced: number;
  untraceable: string[];
  passed: boolean;
  note: string;
}

export interface Verdict {
  label: VerdictLabel;
  claim: string;
  asserted_direction: ChangeDirection;
  measured_direction: ChangeDirection;
  confidence: number;
  confidence_components: ConfidenceComponent[];
  evidence: EvidenceItem[];
  consistency: ConsistencyCheck[];
  surviving_confounders: ConfounderKind[];
  requirements: string[];
  reasoning: string;
  narrative: string | null;
  narrative_source: string;
  narrative_audit: NumericAudit | null;
  what_would_change_it: string[];
}

export interface EvidenceLedger {
  items: EvidenceItem[];
  consistency: ConsistencyCheck[];
  confounders: ConfounderTest[];
  excluded: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Remedy offers (Task 13)
// ---------------------------------------------------------------------------

/** Imagery in the library that removes a surviving objection. */
export interface RemedyOffer {
  confounder: ConfounderKind;
  confounder_label: string;
  /** What the confounder test said was needed, verbatim. */
  requirement: string;
  sample_key: string;
  title: string;
  place: string;
  configuration: InputConfiguration;
  /** Why this scene settles it, from the scene's own declaration. */
  why: string;
  /** True when the scene exists specifically to answer this input's objection. */
  corrects_current: boolean;
  suggested_query: string | null;
}

export interface RemedySet {
  offers: RemedyOffer[];
  /** Objections no scene in the library can answer. */
  unmet: string[];
}

// ---------------------------------------------------------------------------
// Evidence packet (Task 14)
// ---------------------------------------------------------------------------

export type PacketFileRole =
  | "report"
  | "trace"
  | "measurements"
  | "findings"
  | "overlay"
  | "figure"
  | "manifest"
  | "archive";

export interface PacketFile {
  filename: string;
  role: PacketFileRole;
  label: string;
  description: string;
  media_type: string;
  size_bytes: number;
  /** So a recipient can show the file they hold is the file that was written. */
  sha256: string;
}

/** A number printed in the report, and where it came from. */
export interface PacketFigure {
  display: string;
  where: string;
  measurement_key: string;
  source_tool: string;
  source_version: string;
  formula: string;
  /** Set when the figure is traceable but not to a tool measurement. */
  source: string;
}

export interface PacketAudit {
  checked: number;
  traced: number;
  untraceable: string[];
  passed: boolean;
  note: string;
}

export interface EvidencePacket {
  run_id: string;
  session_id: string;
  generated_at: string;
  query: string;
  claim: string;
  verdict_label: VerdictLabel | null;
  verdict_text: string;
  confidence: number | null;
  files: PacketFile[];
  figures: PacketFigure[];
  audit: PacketAudit;
  archive: PacketFile | null;
  notes: string[];
  /** What the packet does not contain, and why. */
  omissions: string[];
}

// ---------------------------------------------------------------------------
// Evidence disagreement
// ---------------------------------------------------------------------------

export type Agreement = "agree" | "disagree" | "uncertain";

/** One method's reading of one region, in its own terms. */
export interface MethodOpinion {
  method: string;
  verdict: string;
  source_tool: string;
  source_version: string;
  basis: string;
  confidence: number | null;
  /** Cloud and missing data are not dissent. */
  could_not_see: boolean;
}

export interface ConflictRegion {
  region_id: number;
  agreement: Agreement;
  area_km2: number;
  pixel_count: number;
  centroid_wgs84: number[];
  bounds_wgs84: number[];
  opinions: MethodOpinion[];
  reason: string;
  reason_measurement_key: string;
  reason_value: number | null;
  action: string;
}

export interface DisagreementReport {
  candidate_area_km2: number;
  agree_area_km2: number;
  disagree_area_km2: number;
  uncertain_area_km2: number;
  /** The headline: how much of the candidate area is contested. */
  conflicting_fraction: number;
  methods: string[];
  regions: ConflictRegion[];
  applies_to: ImageRole[];
  agree_layer: string;
  disagree_layer: string;
  uncertain_layer: string;
  notes: string[];
}
