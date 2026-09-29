"""The sample scene library.

Four curated scenes, one per scripted demo, cut from real Copernicus imagery over
Indian study areas. Each is cached locally as a small GeoTIFF with its STAC item
id and acquisition date embedded, so every measurement taken from it traces back
to a named satellite acquisition.

When SAR cannot be retrieved, a backscatter scene is simulated from the real
optical land cover instead. That path is always labelled ``is_simulated`` and
carries a note explaining how it was produced, because a simulated radar image
must never be mistaken for a measured one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.raster_io import sha256_of
from app.core.sample_sources import (
    COPERNICUS_ATTRIBUTION,
    COPERNICUS_LICENCE,
    FetchedScene,
    TargetGrid,
    now_utc,
)
from app.core.sessions import SessionStore
from app.models.contract import ConfounderKind
from app.models.schemas import (
    ImageRole,
    InputConfiguration,
    SampleAssetRecord,
    SampleManifest,
    SampleProvenance,
    SampleScene,
    SessionRecord,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

# 512 pixels at 10 m is a 5.12 km square: large enough to hold a reservoir or a
# city edge, small enough that each scene stays a few megabytes.
DEFAULT_SIZE_PX = 512


@dataclass(frozen=True)
class OpticalSpec:
    date_range: str
    max_cloud: float = 12.0


@dataclass(frozen=True)
class SarSpec:
    date_range: str


@dataclass(frozen=True)
class PresetDef:
    key: str
    title: str
    description: str
    demo: str
    place: str
    configuration: InputConfiguration
    lon: float
    lat: float
    suggested_queries: tuple[str, ...]
    optical: dict[ImageRole, OpticalSpec] = field(default_factory=dict)
    sar: dict[ImageRole, SarSpec] = field(default_factory=dict)
    size_px: int = DEFAULT_SIZE_PX
    # The role whose grid every other asset is cut onto. Cross-modal pairs use
    # the optical grid so the SAR lands on identical pixels.
    grid_role: ImageRole = ImageRole.SINGLE

    # Alternative explanations this scene removes by construction. A system that
    # says "you would need a same-season acquisition" is more useful if it can
    # then hand you one, so a scene declares what it settles.
    remedies: tuple[ConfounderKind, ...] = ()
    # The preset this one is the corrected counterpart of, when it exists to
    # answer a specific objection raised against that one.
    corrects: str | None = None
    # Why this scene settles the objection, in the user's language.
    remedy_note: str = ""


PRESETS: tuple[PresetDef, ...] = (
    PresetDef(
        key="water_body",
        title="Ukai reservoir",
        description=(
            "A single clear-sky Sentinel-2 scene over the Ukai reservoir on the "
            "Tapi. One large, unambiguous water body with a complex shoreline, "
            "which makes it a fair test of text-guided grounding."
        ),
        demo="Demo 1 - single-image grounding",
        place="Ukai reservoir, Tapi basin, Gujarat",
        configuration=InputConfiguration.SINGLE,
        lon=73.586,
        lat=21.245,
        suggested_queries=(
            "Highlight the water body referred to in the query",
            "Describe the land cover and major objects visible in this image",
            "How much of this scene is covered by water?",
        ),
        optical={ImageRole.SINGLE: OpticalSpec("2024-01-01/2024-03-20", 8.0)},
        grid_role=ImageRole.SINGLE,
    ),
    PresetDef(
        key="reservoir_change",
        title="Hirakud, four years of storage apart",
        description=(
            "The same reservoir in the first week of May four years apart, when "
            "storage is at its lowest and an inter-annual difference is real rather "
            "than seasonal. MNDWI separates open water from everything else cleanly "
            "here, so the area it reports is a measurement rather than a cut through "
            "a single population, and the shoreline it draws can be checked by eye."
        ),
        demo="Demo 2 - claim investigation",
        place="Hirakud reservoir, Mahanadi basin, Odisha",
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        # Chosen by measurement, and the search took three attempts because each
        # earlier attempt was rejected by the system's own checks rather than by
        # opinion. scripts/probe_aoi.py records the whole path.
        #
        # 1. Six built-up candidates were measured first, because urban growth is the
        #    obvious claim to investigate. Every one scored NDBI separability at or
        #    near zero: at ten metres NDBI cannot tell concrete from dry soil, so the
        #    areas it reports are a cut through a single population. The best of them
        #    produced a supported verdict whose own overlay showed the "newly built"
        #    regions vegetated on both dates. That scene is kept as urban_growth, as
        #    the demonstration of an instrument that cannot answer.
        # 2. Four reservoirs were then measured on MNDWI, which does separate water.
        #    The best of those compared a 2019 Sentinel-2A scene against a 2025
        #    Sentinel-2C one, and the radiometry test called that the likely
        #    explanation: surfaces that should not have changed differed by -0.066
        #    reflectance, larger than the change being claimed. The test was right.
        # 3. Comparing acquisitions from the same generation of the constellation
        #    fixed it. Measured with the full pipeline, change then confounders then
        #    verdict:
        #
        #   candidate          net km2   sens    sep  mo  verdict       conf
        #   hirakud 2020-24     +0.373  0.012  0.769   0  supported     0.59
        #   tungabhadra 2020-24 -0.379  0.008  0.763   1  refuted       0.64
        #   tungabhadra 2021-24 -0.446  0.036  0.497   0  inconclusive  0.49
        #   tungabhadra 2019-25 +0.309  0.003  0.724   0  inconclusive  0.33
        #
        # This pair carries the claim: four of five alternative explanations ruled
        # out, radiometry only plausible, and nudging the threshold by a twentieth of
        # the index range moves the answer by one percent.
        lon=83.870,
        lat=21.530,
        suggested_queries=(
            "Has the water spread increased since 2020?",
            "Did the reservoir's surface water area change between these dates?",
            "What changed between these two dates, and where did the change occur?",
        ),
        optical={
            ImageRole.DATE_A: OpticalSpec("2020-05-01/2020-06-10", 8.0),
            ImageRole.DATE_B: OpticalSpec("2024-05-01/2024-06-10", 8.0),
        },
        grid_role=ImageRole.DATE_A,
    ),
    PresetDef(
        key="urban_growth",
        title="New Chandigarh, where the index cannot answer",
        description=(
            "Real construction, six years apart, in the same weeks of the dry "
            "season. NDBI still cannot carry the question: at ten metres it cannot "
            "tell concrete from dry soil, and on this scene it does not divide the "
            "imagery into two populations at all. The system measures a large "
            "increase and then declines to support it, which is the honest answer "
            "and the one a threshold-only pipeline would miss."
        ),
        demo="Demo 5 - when the instrument cannot answer",
        place="New Chandigarh, Sahibzada Ajit Singh Nagar, Punjab",
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        # Chosen by measurement, not from a map. The first centre for this demo was
        # Dholera (72.199E, 22.245N), picked by eye because the region is famous for
        # construction. Measured, that AOI is dry farmland where NDBI tracks field
        # state rather than building: built-up came out at 22.64 -> 21.26 km2, an
        # apparent DECREASE on a scene named for growth, and the verdict engine
        # correctly refused to agree with the claim.
        #
        # scripts/probe_aoi.py then measured six candidates with this same change
        # engine. This one won on every figure that matters:
        #
        #   candidate     net km2   gain    loss   sensitivity  season offset  NDVI
        #   mohali         +2.379   3.187   0.808     0.280       0 months      96%
        #   noida_ext      +0.520   1.642   1.122     0.057       1 month       76%
        #   rajarhat       +0.388   1.686   1.298     0.061       1 month       85%
        #   ulwe           -0.978   0.360   1.338     0.003       0 months        -
        #   kokapet        +0.008   0.850   0.842     0.005       0 months        -
        #   devanahalli    -4.416   1.050   5.466     0.105       2 months        -
        #
        # Gain outweighs loss four to one, the two dates sit in the same calendar
        # window so the seasonality test can rule phenology out rather than merely
        # flag it, and NDVI independently corroborates 96% of the NDBI gain. The
        # losing candidates are kept in the probe script: Ulwe and Devanahalli
        # measure a decrease, and Kokapet is built on rock, which NDBI cannot tell
        # from concrete.
        lon=76.690,
        lat=30.760,
        suggested_queries=(
            "Has the built-up area increased, decreased, or remained unchanged?",
            "What changed between these two dates, and where did the change occur?",
            "Has built-up area increased since 2020?",
        ),
        optical={
            ImageRole.DATE_A: OpticalSpec("2019-12-01/2020-02-28", 8.0),
            ImageRole.DATE_B: OpticalSpec("2025-12-01/2026-02-28", 8.0),
        },
        grid_role=ImageRole.DATE_A,
    ),
    PresetDef(
        key="flood_urban",
        title="Patna monsoon flooding",
        description=(
            "Co-registered optical and radar over the Ganga floodplain during the "
            "2019 monsoon flooding. The optical scene is partly cloud-covered, "
            "which is exactly the situation SAR exists for, and exactly where the "
            "two sensors will disagree."
        ),
        demo="Demo 3 - optical and SAR debate",
        place="Patna and the Ganga floodplain, Bihar",
        configuration=InputConfiguration.CROSS_MODAL_PAIR,
        lon=85.140,
        lat=25.620,
        suggested_queries=(
            "Use the optical and SAR images together to identify built-up and "
            "water-covered regions",
            "Find flooded built-up areas",
            "Where do the optical and radar evidence disagree?",
        ),
        # Cloud tolerance is deliberately high: partial cloud is the point.
        optical={ImageRole.OPTICAL: OpticalSpec("2019-09-15/2019-10-25", 45.0)},
        sar={ImageRole.SAR: SarSpec("2019-09-25/2019-10-18")},
        grid_role=ImageRole.OPTICAL,
    ),
    PresetDef(
        key="seasonal_farmland",
        title="Punjab cropping cycle",
        description=(
            "Wheat at peak green in March against the same fields after the April "
            "harvest. Vegetation genuinely declines, yet nothing was lost: this is "
            "the pair that separates a system which investigates from one which "
            "merely answers."
        ),
        demo="Demo 4 - seasonal confounder",
        place="Irrigated cropland north of Moga, Punjab",
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        # Chosen by measurement, not by eye: mean NDVI here falls from 0.745 in
        # March to 0.151 by the end of May, the largest seasonal swing among the
        # candidate sites tested. An earlier centre sat on Ludhiana city, where
        # the same comparison moved NDVI by only 0.024 and the demo had nothing
        # to show.
        lon=75.180,
        lat=30.900,
        suggested_queries=(
            "Did vegetation decrease?",
            "Has vegetation cover been lost between these two dates?",
            "Is this scene showing land degradation?",
        ),
        optical={
            ImageRole.DATE_A: OpticalSpec("2024-02-25/2024-03-25", 10.0),
            ImageRole.DATE_B: OpticalSpec("2024-05-10/2024-06-10", 10.0),
        },
        grid_role=ImageRole.DATE_A,
    ),
    PresetDef(
        key="seasonal_farmland_same_season",
        title="Punjab, one year apart in the same season",
        description=(
            "The same fields as the seasonal pair, but both acquisitions taken at "
            "peak green a year apart. This is the acquisition the system asks for "
            "when it refuses the seasonal comparison, and it exists so the refusal "
            "can be acted on rather than merely stated."
        ),
        demo="Demo 4 - the corrected comparison",
        place="Irrigated cropland north of Moga, Punjab",
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        # Identical centre to seasonal_farmland, so the two runs are about the
        # same ground and only the calendar differs.
        lon=75.180,
        lat=30.900,
        suggested_queries=(
            "Did vegetation decrease?",
            "Has vegetation cover been lost between these two dates?",
        ),
        optical={
            ImageRole.DATE_A: OpticalSpec("2024-02-25/2024-03-25", 10.0),
            ImageRole.DATE_B: OpticalSpec("2025-02-25/2025-03-25", 10.0),
        },
        grid_role=ImageRole.DATE_A,
        remedies=(ConfounderKind.SEASONALITY,),
        corrects="seasonal_farmland",
        remedy_note=(
            "Both dates sit at the same point in the growing cycle a year apart, "
            "so phenology cannot account for a difference between them."
        ),
    ),
)

PRESETS_BY_KEY: dict[str, PresetDef] = {preset.key: preset for preset in PRESETS}


def asset_filename(preset_key: str, role: ImageRole) -> str:
    return f"{preset_key}__{role.value}.tif"


# ---------------------------------------------------------------------------
# Simulated SAR fallback
# ---------------------------------------------------------------------------

# Backscatter in dB for (VV, VH) per land-cover class. Water is specular and
# therefore dark; built-up is bright from double-bounce off walls.
SIMULATED_SIGMA0_DB: dict[str, tuple[float, float]] = {
    "water": (-22.0, -28.0),
    "vegetation": (-9.0, -15.0),
    "builtup": (-4.0, -11.0),
    "bare": (-12.5, -20.0),
}

SIMULATION_NOTE = (
    "Backscatter simulated from the co-located Sentinel-2 land cover. Water, "
    "vegetation, built-up, and bare classes were derived from NDWI, NDVI, and "
    "NDBI, assigned literature-typical VV/VH sigma-0 values, and given "
    "multiplicative Gamma speckle at 6 looks. This is a physically plausible "
    "stand-in, not a radar measurement, and must not be read as one."
)


def simulate_sar_from_optical(
    optical_cube: np.ndarray,
    grid: TargetGrid,
    *,
    seed: int = 17,
    looks: int = 6,
) -> FetchedScene:
    """Derive a plausible VV/VH scene from real optical land cover.

    Used only when no radar acquisition could be retrieved. The output is always
    flagged as simulated by the caller.
    """
    blue, green, red, nir, swir16, _swir22 = (
        optical_cube[i].astype("float64") for i in range(6)
    )

    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denominator = a + b
        return np.where(denominator != 0, (a - b) / np.where(denominator == 0, 1, denominator), 0.0)

    ndwi = ratio(green, nir)
    ndvi = ratio(nir, red)
    ndbi = ratio(swir16, nir)

    water = ndwi > 0.15
    vegetation = (~water) & (ndvi > 0.35)
    builtup = (~water) & (~vegetation) & (ndbi > 0.0)
    bare = ~(water | vegetation | builtup)

    rng = np.random.default_rng(seed)
    planes: list[np.ndarray] = []
    for pol_index in range(2):
        linear = np.zeros(ndvi.shape, dtype="float64")
        for name, mask in (
            ("water", water),
            ("vegetation", vegetation),
            ("builtup", builtup),
            ("bare", bare),
        ):
            if mask.any():
                linear[mask] = 10.0 ** (SIMULATED_SIGMA0_DB[name][pol_index] / 10.0)
        speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=linear.shape)
        planes.append((linear * speckle).astype("float32"))

    return FetchedScene(
        cube=np.stack(planes),
        band_names=["VV", "VH"],
        grid=grid,
        nodata=None,
        collection="simulated-sar",
        tags={
            "PLATFORM": "simulated",
            "INSTRUMENT": "simulated-C-SAR",
            "PRODUCT": "SIMULATED_SIGMA0",
            "POLARISATIONS": "VV VH",
            "BACKSCATTER_UNITS": "gamma0 linear power",
            "SIMULATED": "true",
            "SIMULATION_NOTE": SIMULATION_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_path(samples_dir: Path) -> Path:
    return Path(samples_dir) / MANIFEST_NAME


def blank_manifest() -> SampleManifest:
    """A manifest describing every preset, with no assets cached yet."""
    return SampleManifest(
        generated_at=now_utc(),
        scenes={
            preset.key: SampleScene(
                key=preset.key,
                title=preset.title,
                description=preset.description,
                demo=preset.demo,
                configuration=preset.configuration,
                suggested_queries=list(preset.suggested_queries),
                place=preset.place,
                remedies=[kind.value for kind in preset.remedies],
                corrects=preset.corrects,
                remedy_note=preset.remedy_note,
            )
            for preset in PRESETS
        },
    )


def save_manifest(manifest: SampleManifest, samples_dir: Path) -> Path:
    path = manifest_path(samples_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(samples_dir: Path) -> SampleManifest:
    """Read the manifest, dropping any asset whose file is not on disk.

    The cached GeoTIFFs are regenerable and therefore not committed, so a fresh
    checkout has a manifest listing assets that are absent. Filtering here means
    the API never advertises a scene it cannot actually load.
    """
    path = manifest_path(samples_dir)
    if not path.exists():
        return blank_manifest()

    try:
        manifest = SampleManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not break the API
        logger.warning("sample manifest unreadable (%s); reporting none cached", exc)
        return blank_manifest()

    for scene in manifest.scenes.values():
        present = {
            role: asset
            for role, asset in scene.assets.items()
            if (Path(samples_dir) / asset.filename).exists()
        }
        if len(present) != len(scene.assets):
            logger.info(
                "scene %s: %d of %d cached files missing from disk",
                scene.key,
                len(scene.assets) - len(present),
                len(scene.assets),
            )
        scene.assets = present

    # Presets added after the manifest was written still get advertised.
    for preset in PRESETS:
        if preset.key not in manifest.scenes:
            manifest.scenes[preset.key] = blank_manifest().scenes[preset.key]

    return manifest


def record_asset(
    preset: PresetDef,
    role: ImageRole,
    scene: FetchedScene,
    path: Path,
    *,
    is_simulated: bool = False,
) -> SampleAssetRecord:
    """Build the manifest entry for a written asset."""
    count, height, width = scene.cube.shape
    simulated = is_simulated or scene.collection == "simulated-sar"

    provenance = SampleProvenance(
        source=(
            "Simulated from co-located Sentinel-2 land cover"
            if simulated
            else "Copernicus Sentinel"
        ),
        collection=scene.collection,
        stac_item_id=scene.item_id,
        stac_url=scene.item_url,
        acquisition_date=scene.acquisition_date,
        cloud_cover_percent=scene.cloud_cover,
        platform=scene.platform,
        instrument=scene.tags.get("INSTRUMENT"),
        bands=list(scene.band_names),
        gsd_m=abs(scene.grid.transform.a),
        epsg=scene.grid.epsg,
        licence=None if simulated else COPERNICUS_LICENCE,
        attribution=None if simulated else COPERNICUS_ATTRIBUTION,
        is_simulated=simulated,
        simulation_note=SIMULATION_NOTE if simulated else None,
        fetched_at=now_utc(),
    )

    return SampleAssetRecord(
        role=role,
        filename=path.name,
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        width=width,
        height=height,
        band_count=count,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Loading a sample into a session
# ---------------------------------------------------------------------------


class SampleNotAvailable(LookupError):
    pass


def load_sample_into_session(
    key: str, store: SessionStore, samples_dir: Path
) -> SessionRecord:
    """Create a session pre-populated with a cached sample scene."""
    manifest = load_manifest(samples_dir)
    scene = manifest.scenes.get(key)
    if scene is None:
        raise SampleNotAvailable(f"No sample scene named '{key}'.")
    if not scene.available:
        raise SampleNotAvailable(
            f"Sample '{key}' is not cached. Run scripts/fetch_samples.py "
            f"(or scripts/make_synthetic.py to work offline)."
        )

    record = store.create()
    for role, asset in scene.assets.items():
        source = Path(samples_dir) / asset.filename
        store.ingest(
            session_id=record.session_id,
            role=role,
            source=source,
            original_filename=asset.filename,
            content_type="image/tiff",
            move=False,
        )

    loaded = store.load(record.session_id)
    loaded.sample_key = key
    loaded.notes.append(f"Loaded sample scene '{key}': {scene.title}")
    if scene.has_simulated_asset:
        loaded.notes.append(
            "This scene includes a simulated radar asset; see its provenance note."
        )
    store.save(loaded)
    logger.info(
        "loaded sample %s into session %s (%d assets)",
        key,
        record.session_id,
        len(scene.assets),
    )
    return store.load(record.session_id)
