"""Shared pydantic schemas and vocabulary for the SatQuery pipeline.

These types are the contract between the raster reader, the readiness gate, the
tool registry, and the API layer. Band roles in particular matter: every
spectral index declares which roles it needs, and availability gating in the
evidence ledger is driven by whether those roles were resolved on the input.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class BandRole(str, Enum):
    """Semantic role of a raster band.

    Optical roles follow the Sentinel-2 / STAC common names so that scenes
    fetched from Earth Search map across without translation. SAR roles are
    polarisation channels.
    """

    # Optical / multispectral
    COASTAL = "coastal"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    REDEDGE1 = "rededge1"
    REDEDGE2 = "rededge2"
    REDEDGE3 = "rededge3"
    NIR = "nir"
    NIR08 = "nir08"
    NIR09 = "nir09"
    CIRRUS = "cirrus"
    SWIR16 = "swir16"
    SWIR22 = "swir22"
    # Sentinel-2 scene classification layer: the real cloud/shadow mask
    SCL = "scl"
    # SAR polarisations
    VV = "vv"
    VH = "vh"
    HH = "hh"
    HV = "hv"
    # Generic
    GRAY = "gray"
    ALPHA = "alpha"
    UNKNOWN = "unknown"


OPTICAL_ROLES: frozenset[BandRole] = frozenset(
    {
        BandRole.COASTAL,
        BandRole.BLUE,
        BandRole.GREEN,
        BandRole.RED,
        BandRole.REDEDGE1,
        BandRole.REDEDGE2,
        BandRole.REDEDGE3,
        BandRole.NIR,
        BandRole.NIR08,
        BandRole.NIR09,
        BandRole.CIRRUS,
        BandRole.SWIR16,
        BandRole.SWIR22,
    }
)

SAR_ROLES: frozenset[BandRole] = frozenset(
    {BandRole.VV, BandRole.VH, BandRole.HH, BandRole.HV}
)

# Sentinel-2 MSI band centres, in nanometres. Values taken from the Earth Search
# sentinel-2-l2a collection metadata, so they match the fetched imagery exactly.
BAND_WAVELENGTH_NM: dict[BandRole, float] = {
    BandRole.COASTAL: 443.0,
    BandRole.BLUE: 490.0,
    BandRole.GREEN: 560.0,
    BandRole.RED: 665.0,
    BandRole.REDEDGE1: 704.0,
    BandRole.REDEDGE2: 740.0,
    BandRole.REDEDGE3: 783.0,
    BandRole.NIR: 842.0,
    BandRole.NIR08: 865.0,
    BandRole.NIR09: 945.0,
    BandRole.CIRRUS: 1373.5,
    BandRole.SWIR16: 1610.0,
    BandRole.SWIR22: 2190.0,
}

# Normalised band label -> role. Keys are lowercased with separators stripped,
# so "B8A", "b8a", and "B_8A" all collapse to "b8a".
BAND_ALIASES: dict[str, BandRole] = {
    # Sentinel-2 numbering
    "b1": BandRole.COASTAL, "b01": BandRole.COASTAL, "coastal": BandRole.COASTAL,
    "coastalaerosol": BandRole.COASTAL, "aerosol": BandRole.COASTAL,
    "b2": BandRole.BLUE, "b02": BandRole.BLUE, "blue": BandRole.BLUE,
    "b3": BandRole.GREEN, "b03": BandRole.GREEN, "green": BandRole.GREEN,
    "b4": BandRole.RED, "b04": BandRole.RED, "red": BandRole.RED,
    "b5": BandRole.REDEDGE1, "b05": BandRole.REDEDGE1, "rededge1": BandRole.REDEDGE1,
    "b6": BandRole.REDEDGE2, "b06": BandRole.REDEDGE2, "rededge2": BandRole.REDEDGE2,
    "b7": BandRole.REDEDGE3, "b07": BandRole.REDEDGE3, "rededge3": BandRole.REDEDGE3,
    "b8": BandRole.NIR, "b08": BandRole.NIR, "nir": BandRole.NIR,
    "nearinfrared": BandRole.NIR, "infrared": BandRole.NIR,
    "b8a": BandRole.NIR08, "nir08": BandRole.NIR08, "nir2": BandRole.NIR08,
    "b9": BandRole.NIR09, "b09": BandRole.NIR09, "nir09": BandRole.NIR09,
    "watervapour": BandRole.NIR09, "watervapor": BandRole.NIR09,
    "b10": BandRole.CIRRUS, "cirrus": BandRole.CIRRUS,
    "b11": BandRole.SWIR16, "swir16": BandRole.SWIR16, "swir1": BandRole.SWIR16,
    "b12": BandRole.SWIR22, "swir22": BandRole.SWIR22, "swir2": BandRole.SWIR22,
    "scl": BandRole.SCL, "sceneclassification": BandRole.SCL,
    "sceneclassificationmap": BandRole.SCL, "classification": BandRole.SCL,
    # SAR polarisations, including SNAP-style processed names
    "vv": BandRole.VV, "sigma0vv": BandRole.VV, "gamma0vv": BandRole.VV,
    "vvdb": BandRole.VV, "sigma0vvdb": BandRole.VV, "amplitudevv": BandRole.VV,
    "vh": BandRole.VH, "sigma0vh": BandRole.VH, "gamma0vh": BandRole.VH,
    "vhdb": BandRole.VH, "sigma0vhdb": BandRole.VH, "amplitudevh": BandRole.VH,
    "hh": BandRole.HH, "sigma0hh": BandRole.HH, "gamma0hh": BandRole.HH,
    "hv": BandRole.HV, "sigma0hv": BandRole.HV, "gamma0hv": BandRole.HV,
    # Generic
    "gray": BandRole.GRAY, "grey": BandRole.GRAY, "panchromatic": BandRole.GRAY,
    "pan": BandRole.GRAY, "alpha": BandRole.ALPHA, "mask": BandRole.ALPHA,
}


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class ImageRole(str, Enum):
    """Which slot an uploaded image occupies in the input configuration."""

    SINGLE = "single"
    OPTICAL = "optical"
    SAR = "sar"
    DATE_A = "date_a"
    DATE_B = "date_b"


class InputConfiguration(str, Enum):
    """The three input configurations the problem statement defines."""

    SINGLE = "single"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"
    INCOMPLETE = "incomplete"


class FormatClass(str, Enum):
    """Geospatial rasters carry a CRS; benchmark rasters do not.

    PNG and JPEG are accepted only for the prescribed public benchmark datasets,
    so they are tracked separately and the readiness gate treats a missing CRS
    on them as expected rather than as a failure.
    """

    GEOSPATIAL = "geospatial"
    BENCHMARK_RASTER = "benchmark_raster"


class DateSource(str, Enum):
    GEOTIFF_TAG = "geotiff-tag"
    MANIFEST = "manifest"
    FILENAME = "filename"
    USER = "user"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Raster metadata
# ---------------------------------------------------------------------------


class BandStats(BaseModel):
    """Per-band statistics.

    ``sample_fraction`` records the decimation used: large rasters are read at
    reduced resolution for speed, and saying so keeps the displayed numbers
    honest rather than implying an exhaustive scan.
    """

    minimum: float
    maximum: float
    mean: float
    stddev: float
    valid_pixels: int
    nodata_pixels: int
    nodata_fraction: float
    sample_fraction: float = 1.0
    percentile_2: float | None = None
    percentile_98: float | None = None


class BandInfo(BaseModel):
    index: int = Field(description="1-based band index as used by rasterio")
    label: str = Field(description="Band description from the file, or a derived name")
    role: BandRole
    role_source: str = Field(
        description="How the role was resolved: description, tag, convention, or unknown"
    )
    dtype: str
    wavelength_nm: float | None = None
    stats: BandStats | None = None


class GeoReference(BaseModel):
    crs_wkt: str | None = None
    epsg: int | None = None
    crs_name: str | None = None
    is_projected: bool = False
    is_geographic: bool = False
    axis_unit: str | None = None
    transform: list[float] = Field(
        default_factory=list, description="Six affine coefficients (a, b, c, d, e, f)"
    )
    # Native pixel size in CRS units, plus the metre-equivalent. For a
    # geographic CRS these differ, and the conversion is latitude-dependent.
    pixel_size_native_x: float | None = None
    pixel_size_native_y: float | None = None
    gsd_x_m: float | None = None
    gsd_y_m: float | None = None
    gsd_m: float | None = Field(
        default=None, description="Mean of gsd_x_m and gsd_y_m, the headline GSD"
    )
    gsd_method: str | None = Field(
        default=None,
        description="How GSD was derived: projected-linear-unit or geodesic-at-centre",
    )
    bounds_native: list[float] | None = None
    bounds_wgs84: list[float] | None = Field(
        default=None, description="[west, south, east, north] in degrees"
    )
    centroid_wgs84: list[float] | None = Field(
        default=None, description="[longitude, latitude] in degrees"
    )
    area_km2: float | None = None


class ModalityInference(BaseModel):
    modality: Modality
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class RasterMetadata(BaseModel):
    """Everything the pipeline knows about one raster after ingest."""

    # Identity
    original_filename: str
    size_bytes: int
    driver: str
    format_class: FormatClass

    # Shape
    width: int
    height: int
    band_count: int
    dtype: str
    megapixels: float

    # Georeferencing
    geo: GeoReference

    # Content
    bands: list[BandInfo]
    resolved_roles: list[BandRole]
    modality: ModalityInference
    nodata_value: float | None = None
    nodata_fraction: float = 0.0

    # Temporal
    acquisition_date: datetime | None = None
    date_source: DateSource = DateSource.UNKNOWN
    day_of_year: int | None = None

    # Handling hints
    tiling_recommended: bool = False
    overview_levels: list[int] = Field(default_factory=list)
    block_shape: list[int] | None = None

    # Raw file tags, useful for the execution trace and for debugging odd inputs
    tags: dict[str, str] = Field(default_factory=dict)

    # Warnings raised during ingest that are not yet readiness failures
    ingest_notes: list[str] = Field(default_factory=list)

    def has_roles(self, *roles: BandRole) -> bool:
        """True when every requested band role was resolved on this raster."""
        present = set(self.resolved_roles)
        return all(role in present for role in roles)

    def missing_roles(self, *roles: BandRole) -> list[BandRole]:
        present = set(self.resolved_roles)
        return [role for role in roles if role not in present]

    def band_index(self, role: BandRole) -> int | None:
        """1-based rasterio index for a role, or None when absent."""
        for band in self.bands:
            if band.role is role:
                return band.index
        return None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class IngestedImage(BaseModel):
    role: ImageRole
    stored_filename: str
    original_filename: str
    content_type: str | None = None
    sha256: str
    ingested_at: datetime
    metadata: RasterMetadata
    thumbnail_available: bool = False
    thumbnail_recipe: str | None = Field(
        default=None,
        description="How the preview was composited, e.g. 'R=VV dB, G=VH dB, B=VV-VH dB'",
    )
    ingest_ms: float = 0.0


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    images: dict[ImageRole, IngestedImage] = Field(default_factory=dict)
    configuration: InputConfiguration = InputConfiguration.INCOMPLETE
    notes: list[str] = Field(default_factory=list)
    # Set when the imagery came from the sample library. Used to avoid offering
    # the scene already loaded as the remedy for its own objection.
    sample_key: str | None = None

    def image_list(self) -> list[IngestedImage]:
        order = [
            ImageRole.SINGLE,
            ImageRole.OPTICAL,
            ImageRole.SAR,
            ImageRole.DATE_A,
            ImageRole.DATE_B,
        ]
        return [self.images[role] for role in order if role in self.images]


def derive_configuration(roles: set[ImageRole]) -> InputConfiguration:
    """Infer the input configuration from which slots are occupied."""
    if roles == {ImageRole.SINGLE}:
        return InputConfiguration.SINGLE
    if roles == {ImageRole.OPTICAL, ImageRole.SAR}:
        return InputConfiguration.CROSS_MODAL_PAIR
    if roles == {ImageRole.DATE_A, ImageRole.DATE_B}:
        return InputConfiguration.BI_TEMPORAL_PAIR
    return InputConfiguration.INCOMPLETE


class ApiMessage(BaseModel):
    """Generic acknowledgement payload."""

    ok: bool = True
    message: str
    detail: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class ReadinessCheck(BaseModel):
    """One readiness check, carrying the value it measured and the threshold.

    ``measured`` is the display string and ``measured_numeric`` the raw value.
    Both are present so the UI can show a formatted readout while the execution
    trace and report keep the number that the comparison actually used.
    """

    id: str
    label: str
    status: CheckStatus
    measured: str
    measured_numeric: float | None = None
    unit: str | None = None
    threshold: str | None = None
    message: str
    method: str | None = Field(
        default=None, description="How the value was obtained, for the audit trail"
    )
    applies_to: list[ImageRole] = Field(default_factory=list)


class ReadinessVerdict(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    REFUSED = "refused"


class DataRequirement(BaseModel):
    """What the user would need to supply to clear a refusal."""

    what: str
    why: str


class ReadinessReport(BaseModel):
    session_id: str
    configuration: InputConfiguration
    verdict: ReadinessVerdict
    checks: list[ReadinessCheck] = Field(default_factory=list)
    refusal_reasons: list[str] = Field(default_factory=list)
    requirements: list[DataRequirement] = Field(default_factory=list)
    computed_ms: float = 0.0

    # Carried forward to later stages so they do not recompute geometry.
    common_epsg: int | None = None
    overlap_bounds_native: list[float] | None = None
    overlap_bounds_wgs84: list[float] | None = None
    overlap_fraction: float | None = None

    # Inputs to the seasonality confounder (Task 10) and the kill-shot (Task 13).
    day_delta: int | None = None
    month_of_year_delta: int | None = None
    seasonal_risk: bool = False

    @property
    def passed(self) -> bool:
        return self.verdict is not ReadinessVerdict.REFUSED

    def by_status(self, status: CheckStatus) -> list[ReadinessCheck]:
        return [check for check in self.checks if check.status is status]


# ---------------------------------------------------------------------------
# Sample scene library
# ---------------------------------------------------------------------------


class SampleProvenance(BaseModel):
    """Where a sample raster came from.

    Real imagery records the STAC item it was cut from, so any number derived
    from it can be traced back to a specific satellite acquisition. Simulated
    imagery says so explicitly and explains how it was produced.
    """

    source: str = Field(description="Human-readable source name")
    collection: str | None = None
    stac_item_id: str | None = None
    stac_url: str | None = None
    acquisition_date: datetime | None = None
    cloud_cover_percent: float | None = None
    platform: str | None = None
    instrument: str | None = None
    bands: list[str] = Field(default_factory=list)
    gsd_m: float | None = None
    epsg: int | None = None
    licence: str | None = None
    attribution: str | None = None
    is_simulated: bool = False
    simulation_note: str | None = None
    fetched_at: datetime | None = None


class SampleAssetRecord(BaseModel):
    role: ImageRole
    filename: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    band_count: int
    provenance: SampleProvenance


class SampleScene(BaseModel):
    key: str
    title: str
    description: str
    demo: str = Field(description="Which scripted demo this scene powers")
    configuration: InputConfiguration
    suggested_queries: list[str] = Field(default_factory=list)
    place: str
    assets: dict[ImageRole, SampleAssetRecord] = Field(default_factory=dict)

    # Alternative explanations this scene removes by construction, and the scene
    # whose objection it answers. Present so a refusal can be acted on: a system
    # that names the acquisition it needs should be able to hand you one.
    remedies: list[str] = Field(default_factory=list)
    corrects: str | None = None
    remedy_note: str = ""

    @property
    def available(self) -> bool:
        return bool(self.assets)

    @property
    def has_simulated_asset(self) -> bool:
        return any(a.provenance.is_simulated for a in self.assets.values())


class SampleManifest(BaseModel):
    generated_at: datetime
    scenes: dict[str, SampleScene] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tools, measurements, and the registry
# ---------------------------------------------------------------------------


class ToolImplementation(str, Enum):
    """How a tool arrives at its output.

    Recorded on every result and surfaced in the execution trace and the report.
    It is the difference between "this number was computed from the pixels",
    "this came from a trained model", and "a language model phrased this".
    """

    DETERMINISTIC = "deterministic"
    LEARNED = "learned"
    LLM_NARRATION = "llm-narration"


class Measurement(BaseModel):
    """A single quantitative result, carrying how it was obtained.

    The Never-Guess Rule is enforced here in practice: nothing reaches the UI as
    a number unless it arrived as a Measurement, and a Measurement cannot be
    constructed without a formula and the inputs that formula consumed.
    """

    key: str
    label: str
    value: float
    unit: str
    formula: str = Field(description="The expression evaluated, with its constants")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Every value the formula consumed"
    )
    source_tool: str
    source_version: str
    method: str | None = None
    applies_to: list[ImageRole] = Field(default_factory=list)
    precision: int = Field(default=2, description="Suggested display precision")

    def render(self) -> str:
        return f"{self.value:.{self.precision}f} {self.unit}".strip()


class ToolRequirement(BaseModel):
    """What a tool needs before it can run."""

    band_roles: list[BandRole] = Field(
        default_factory=list, description="Roles that must be present on each input"
    )
    any_of_band_roles: list[list[BandRole]] = Field(
        default_factory=list,
        description="Alternative role sets; at least one group must be satisfied",
    )
    modalities: list[Modality] = Field(default_factory=list)
    configurations: list[InputConfiguration] = Field(default_factory=list)
    requires_crs: bool = False
    description: str = ""


class ToolDescriptor(BaseModel):
    """Public description of a registered tool.

    Used by the API, and by the Analysis Contract validator to reject a tool name
    the language model invented.
    """

    name: str
    version: str
    implementation: ToolImplementation
    summary: str
    requirement: ToolRequirement
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Permitted parameters and their defaults"
    )
    produces: list[str] = Field(
        default_factory=list, description="Measurement and mask keys this can emit"
    )
    registered: bool = True


class MaskSummary(BaseModel):
    """Serialisable description of a raster mask a tool produced."""

    key: str
    label: str
    description: str = ""
    pixel_count: int
    total_pixels: int
    coverage_fraction: float
    area_km2: float | None = None
    epsg: int | None = None
    bounds_wgs84: list[float] | None = None
    stored_filename: str | None = None
    applies_to: list[ImageRole] = Field(default_factory=list)
    # For threshold-derived masks: the value and how confidently it separates.
    threshold: float | None = None
    threshold_method: str | None = None
    separability: float | None = None


class ToolRun(BaseModel):
    """Serialisable record of one tool execution, for the trace and the packet."""

    tool: str
    version: str
    implementation: ToolImplementation
    ok: bool
    skipped_reason: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    measurements: list[Measurement] = Field(default_factory=list)
    masks: list[MaskSummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0

    def measurement(self, key: str) -> Measurement | None:
        for item in self.measurements:
            if item.key == key:
                return item
        return None
