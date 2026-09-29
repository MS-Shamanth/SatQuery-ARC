"""Models for SatQuery ARC: the searchable archive, semantic search results,
verified change, and the analyst review queue with its audit log.

These are the API-facing pydantic records. Heavy in-memory objects (numpy
arrays, tool outcomes) never appear here; only what a client needs to render the
Search & Review dashboard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------

Concept = Literal["built_up", "water", "vegetation", "bare"]
SearchMode = Literal["text", "image", "similar"]


class ConceptScores(BaseModel):
    """How much of each land-cover concept a tile shows, in [0, 1].

    Derived from real spectral indices (NDBI/MNDWI/NDVI), never assumed. These
    double as the tile's retrieval embedding for the spectral-concept embedder.
    """

    built_up: float = 0.0
    water: float = 0.0
    vegetation: float = 0.0
    bare: float = 0.0

    def vector(self) -> list[float]:
        return [self.built_up, self.water, self.vegetation, self.bare]


class ArchiveTile(BaseModel):
    """One acquisition of one place in the searchable archive."""

    tile_id: str
    aoi_key: str
    role: str  # date_a | date_b | single
    place: str
    region: str
    lat: float
    lon: float
    bounds_wgs84: list[float]  # [W, S, E, N]
    sensor: str  # e.g. "Sentinel-2 L2A"
    platform: str  # e.g. "Sentinel-2A"
    acquisition_date: str | None
    gsd_m: float | None
    epsg: int | None
    scene_id: str | None  # STAC item id when known
    cloud_fraction: float
    quality: float  # 1 - cloud_fraction, clamped
    concepts: ConceptScores
    embedding: list[float]
    thumbnail_url: str
    tags: list[str] = Field(default_factory=list)


class ArchiveAOI(BaseModel):
    """A place in the archive: one or more acquisitions that can be compared.

    The unit of search and of verified change. An AOI with two dates can be
    verified; a single-date AOI can only be retrieved.
    """

    aoi_key: str
    place: str
    region: str
    lat: float
    lon: float
    bounds_wgs84: list[float]
    sensor: str
    primary_concept: Concept
    change_hint: str  # e.g. "Built-up increase" or "Water body"
    dates: list[str]
    has_pair: bool
    near_water: bool
    thumbnail_url: str
    tile_ids: list[str]


class ArchiveStats(BaseModel):
    generated_at: str
    tile_count: int
    aoi_count: int
    pair_count: int
    regions: list[str]
    sensors: list[str]
    concepts: dict[str, int]
    embedder: str
    date_range: list[str | None]  # [earliest, latest]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    region: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sensors: list[str] = Field(default_factory=list)
    min_quality: float = 0.0
    concept: Concept | None = None
    near_water: bool = False


class SearchQuery(BaseModel):
    text: str = ""
    mode: SearchMode = "text"
    # For mode="similar": the tile whose neighbours are wanted.
    tile_id: str | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = 8


class ResultTag(BaseModel):
    label: str
    kind: Literal["change", "water", "sensor", "quality", "info"] = "info"


class SearchResult(BaseModel):
    rank: int
    aoi_key: str
    tile_id: str
    place: str
    region: str
    lat: float
    lon: float
    bounds_wgs84: list[float]
    sensor: str
    acquisition_date: str | None
    gsd_m: float | None
    match: float  # 0..1, presented as the relevance score
    change_hint: str
    has_pair: bool
    near_water: bool
    thumbnail_url: str
    tags: list[ResultTag] = Field(default_factory=list)
    why: str  # one line: why this ranked here


class SearchResponse(BaseModel):
    query: SearchQuery
    interpreted_as: str  # how the query was parsed, in plain words
    concept_weights: dict[str, float]
    total_indexed: int
    matched: int
    results: list[SearchResult]
    embedder: str
    took_ms: float


# ---------------------------------------------------------------------------
# Verified change
# ---------------------------------------------------------------------------


class ChangeFrame(BaseModel):
    role: str
    acquisition_date: str | None
    scene_id: str | None
    sensor: str
    thumbnail_url: str
    index_label: str
    index_value: float | None


class ConfidenceBar(BaseModel):
    name: str
    label: str
    measured: float
    weight: float
    contribution: float


class ConfounderCheck(BaseModel):
    kind: str
    label: str
    verdict: str  # ruled_out | plausible | likely | not_tested
    cleared: bool
    measured: str
    detail: str


class TimelineObservation(BaseModel):
    date: str
    clear: bool
    is_earliest_supported: bool = False


class ChangeVerification(BaseModel):
    aoi_key: str
    place: str
    region: str
    lat: float
    lon: float
    bounds_wgs84: list[float]

    verdict_label: str  # supported | refuted | inconclusive | unanswerable
    headline: str  # e.g. "Confirmed change - Built-up expansion"
    direction: str  # increased | decreased | unchanged | unspecified
    confidence: float
    reasoning: str

    index_name: str
    target_label: str
    area_before_km2: float | None
    area_after_km2: float | None
    delta_area_km2: float | None
    delta_display: str  # e.g. "+3.19 km²"
    percentage_change: float | None

    earliest_supported_date: str | None
    persists_in_scenes: int

    before: ChangeFrame
    after: ChangeFrame

    confidence_components: list[ConfidenceBar]
    confounders: list[ConfounderCheck]
    confounders_cleared: int
    confounders_total: int
    timeline: list[TimelineObservation]

    # Provenance
    before_scene_id: str | None
    after_scene_id: str | None
    embedding_model: str
    change_model: str
    index_engine: str
    checksum: str

    run_id: str
    took_ms: float


# ---------------------------------------------------------------------------
# Review queue + audit log
# ---------------------------------------------------------------------------

ReviewStatus = Literal["pending", "confirmed", "rejected"]


class ReviewItem(BaseModel):
    item_id: str
    aoi_key: str
    place: str
    region: str
    change_hint: str
    verdict_label: str
    confidence: float
    status: ReviewStatus = "pending"
    delta_display: str = ""
    acquisition_date: str | None = None
    sensor: str = ""
    thumbnail_url: str = ""
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    note: str | None = None
    run_id: str | None = None


class AuditEntry(BaseModel):
    entry_id: str
    at: str
    actor: str
    action: str  # searched | verified | confirmed | rejected | reopened | loaded
    target: str
    detail: str
    confidence: float | None = None


class ReviewDecision(BaseModel):
    decision: Literal["confirm", "reject", "reopen"]
    actor: str = "analyst"
    note: str | None = None


class ReviewState(BaseModel):
    items: list[ReviewItem]
    audit: list[AuditEntry]
    pending: int
    confirmed: int
    rejected: int
