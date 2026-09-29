/**
 * SatQuery ARC API client + types.
 *
 * Mirrors backend/app/models/archive.py. Kept self-contained so the Search &
 * Review workspace does not depend on the investigator's type surface.
 */

const BASE = "/api";

export class ArcError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ArcError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (cause) {
    throw new ArcError(
      `Cannot reach the SatQuery ARC backend at ${BASE}${path}. Is the backend running?`,
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body === "object" && "detail" in body) detail = String(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ArcError(detail, res.status);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Concept = "built_up" | "water" | "vegetation" | "bare";
export type SearchMode = "text" | "image" | "similar";

export interface ArchiveStats {
  generated_at: string;
  tile_count: number;
  aoi_count: number;
  pair_count: number;
  regions: string[];
  sensors: string[];
  concepts: Record<string, number>;
  embedder: string;
  date_range: (string | null)[];
}

export interface ArchiveAOI {
  aoi_key: string;
  place: string;
  region: string;
  lat: number;
  lon: number;
  bounds_wgs84: number[];
  sensor: string;
  primary_concept: Concept;
  change_hint: string;
  dates: string[];
  has_pair: boolean;
  near_water: boolean;
  thumbnail_url: string;
  tile_ids: string[];
}

export interface ArchiveOverview {
  stats: ArchiveStats;
  aois: ArchiveAOI[];
}

export interface ResultTag {
  label: string;
  kind: "change" | "water" | "sensor" | "quality" | "info";
}

export interface SearchResult {
  rank: number;
  aoi_key: string;
  tile_id: string;
  place: string;
  region: string;
  lat: number;
  lon: number;
  bounds_wgs84: number[];
  sensor: string;
  acquisition_date: string | null;
  gsd_m: number | null;
  match: number;
  change_hint: string;
  has_pair: boolean;
  near_water: boolean;
  thumbnail_url: string;
  tags: ResultTag[];
  why: string;
}

export interface SearchFilters {
  region?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  sensors?: string[];
  min_quality?: number;
  concept?: Concept | null;
  near_water?: boolean;
}

export interface SearchQuery {
  text: string;
  mode: SearchMode;
  tile_id?: string | null;
  filters?: SearchFilters;
  limit?: number;
}

export interface SearchResponse {
  query: SearchQuery;
  interpreted_as: string;
  concept_weights: Record<string, number>;
  total_indexed: number;
  matched: number;
  results: SearchResult[];
  embedder: string;
  took_ms: number;
}

export interface ChangeFrame {
  role: string;
  acquisition_date: string | null;
  scene_id: string | null;
  sensor: string;
  thumbnail_url: string;
  index_label: string;
  index_value: number | null;
}

export interface ConfidenceBar {
  name: string;
  label: string;
  measured: number;
  weight: number;
  contribution: number;
}

export interface ConfounderCheck {
  kind: string;
  label: string;
  verdict: string;
  cleared: boolean;
  measured: string;
  detail: string;
}

export interface TimelineObservation {
  date: string;
  clear: boolean;
  is_earliest_supported: boolean;
}

export interface ChangeVerification {
  aoi_key: string;
  place: string;
  region: string;
  lat: number;
  lon: number;
  bounds_wgs84: number[];
  verdict_label: string;
  headline: string;
  direction: string;
  confidence: number;
  reasoning: string;
  index_name: string;
  target_label: string;
  area_before_km2: number | null;
  area_after_km2: number | null;
  delta_area_km2: number | null;
  delta_display: string;
  percentage_change: number | null;
  earliest_supported_date: string | null;
  persists_in_scenes: number;
  before: ChangeFrame;
  after: ChangeFrame;
  confidence_components: ConfidenceBar[];
  confounders: ConfounderCheck[];
  confounders_cleared: number;
  confounders_total: number;
  timeline: TimelineObservation[];
  before_scene_id: string | null;
  after_scene_id: string | null;
  embedding_model: string;
  change_model: string;
  index_engine: string;
  checksum: string;
  run_id: string;
  took_ms: number;
}

export type ReviewStatus = "pending" | "confirmed" | "rejected";

export interface ReviewItem {
  item_id: string;
  aoi_key: string;
  place: string;
  region: string;
  change_hint: string;
  verdict_label: string;
  confidence: number;
  status: ReviewStatus;
  delta_display: string;
  acquisition_date: string | null;
  sensor: string;
  thumbnail_url: string;
  created_at: string;
  decided_at: string | null;
  decided_by: string | null;
  note: string | null;
  run_id: string | null;
}

export interface AuditEntry {
  entry_id: string;
  at: string;
  actor: string;
  action: string;
  target: string;
  detail: string;
  confidence: number | null;
}

export interface ReviewState {
  items: ReviewItem[];
  audit: AuditEntry[];
  pending: number;
  confirmed: number;
  rejected: number;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export const arc = {
  archive: () => req<ArchiveOverview>("/archive"),
  stats: () => req<ArchiveStats>("/archive/stats"),
  thumbUrl: (tileId: string) => `${BASE}/archive/thumb/${tileId}.png`,

  search: (query: SearchQuery) =>
    req<SearchResponse>("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(query),
    }),

  verify: (aoiKey: string, refresh = false) =>
    req<ChangeVerification>(`/verify/${aoiKey}${refresh ? "?refresh=true" : ""}`, {
      method: "POST",
    }),

  review: () => req<ReviewState>("/review"),

  decide: (itemId: string, decision: "confirm" | "reject" | "reopen", note?: string) =>
    req<ReviewItem>(`/review/${itemId}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, actor: "analyst", note: note ?? null }),
    }),

  audit: () => req<AuditEntry[]>("/audit"),

  resetReview: () => req<{ ok: boolean }>("/review/reset", { method: "POST" }),
};
