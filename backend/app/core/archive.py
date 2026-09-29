"""The searchable tile archive.

Scans the real Sentinel-2 GeoTIFFs already on disk (the curated sample scenes
plus the AOI-probe pairs) into a catalogue that can be searched by meaning and
compared date-to-date. Each tile carries its real georeferencing, acquisition
date, sensor, cloud fraction, spectral land-cover fractions and a rendered
thumbnail. Nothing here is invented from a filename: place and region labels are
metadata, but every number is measured from the pixels.

The index is built once by ``build_archive`` (a script or the first request) and
cached as ``data/archive/index.json`` plus per-tile thumbnails.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.core.embedding import active_embedder_name, extract_features
from app.core.raster_io import build_thumbnail, read_metadata, sha256_of
from app.core.samples import PRESETS_BY_KEY, asset_filename
from app.models.archive import (
    ArchiveAOI,
    ArchiveStats,
    ArchiveTile,
    Concept,
    ConceptScores,
)
from app.models.schemas import ImageRole

logger = logging.getLogger(__name__)

INDEX_NAME = "index.json"

# Place / region / concept labels for the AOI-probe scenes. These are editorial
# metadata (what and where), never a measurement. The change hint is a prior; the
# verified-change pipeline is what actually decides whether it happened.
AOI_META: dict[str, dict[str, str]] = {
    # Built-up growth candidates
    "probe_mohali": {"place": "New Chandigarh, Mohali", "region": "Punjab", "concept": "built_up", "hint": "Built-up increase"},
    "probe_noida_ext": {"place": "Noida Extension", "region": "Uttar Pradesh", "concept": "built_up", "hint": "Built-up increase"},
    "probe_rajarhat": {"place": "Rajarhat, New Town", "region": "West Bengal", "concept": "built_up", "hint": "Built-up increase"},
    "probe_ulwe": {"place": "Ulwe, Navi Mumbai", "region": "Maharashtra", "concept": "built_up", "hint": "Built-up change"},
    "probe_kokapet": {"place": "Kokapet, Hyderabad", "region": "Telangana", "concept": "built_up", "hint": "Built-up increase"},
    "probe_devanahalli": {"place": "Devanahalli, Bengaluru", "region": "Karnataka", "concept": "built_up", "hint": "Built-up change"},
    # Water / reservoir candidates
    "probe_hirakud": {"place": "Hirakud reservoir", "region": "Odisha", "concept": "water", "hint": "Water extent change"},
    "probe_hirakud_2020": {"place": "Hirakud reservoir", "region": "Odisha", "concept": "water", "hint": "Water extent change"},
    "probe_tungabhadra": {"place": "Tungabhadra reservoir", "region": "Karnataka", "concept": "water", "hint": "Water extent change"},
    "probe_tungabhadra_2019": {"place": "Tungabhadra reservoir", "region": "Karnataka", "concept": "water", "hint": "Water extent change"},
    "probe_tungabhadra_2020": {"place": "Tungabhadra reservoir", "region": "Karnataka", "concept": "water", "hint": "Water extent change"},
    "probe_tungabhadra_2021": {"place": "Tungabhadra reservoir", "region": "Karnataka", "concept": "water", "hint": "Water extent change"},
    "probe_nagarjuna": {"place": "Nagarjuna Sagar", "region": "Telangana", "concept": "water", "hint": "Water extent change"},
    "probe_ukai_drawdown": {"place": "Ukai reservoir", "region": "Gujarat", "concept": "water", "hint": "Water drawdown"},
}

# Region / concept for the curated sample presets (keyed by preset key).
SAMPLE_META: dict[str, dict[str, str]] = {
    "water_body": {"region": "Gujarat", "concept": "water", "hint": "Water body"},
    "reservoir_change": {"region": "Odisha", "concept": "water", "hint": "Water extent change"},
    "urban_growth": {"region": "Punjab", "concept": "built_up", "hint": "Built-up increase"},
    "flood_urban": {"region": "Bihar", "concept": "water", "hint": "Flood extent"},
    "seasonal_farmland": {"region": "Punjab", "concept": "vegetation", "hint": "Seasonal vegetation"},
    "seasonal_farmland_same_season": {"region": "Punjab", "concept": "vegetation", "hint": "Seasonal vegetation"},
}

WATER_PRESENCE_THRESHOLD = 0.06


@dataclass
class ArchiveIndex:
    """In-memory archive: API records plus the server-only source paths."""

    tiles: dict[str, ArchiveTile]
    aois: dict[str, ArchiveAOI]
    sources: dict[str, Path]  # tile_id -> absolute file path
    stats: ArchiveStats

    def aoi_tiles(self, aoi_key: str) -> list[ArchiveTile]:
        aoi = self.aois.get(aoi_key)
        if aoi is None:
            return []
        return [self.tiles[t] for t in aoi.tile_ids if t in self.tiles]

    def pair_paths(self, aoi_key: str) -> tuple[Path, Path] | None:
        """The (date_a, date_b) source files for an AOI, if it has a pair."""
        aoi = self.aois.get(aoi_key)
        if aoi is None or not aoi.has_pair:
            return None
        a = b = None
        for tile_id in aoi.tile_ids:
            tile = self.tiles.get(tile_id)
            if tile is None:
                continue
            if tile.role == ImageRole.DATE_A.value:
                a = self.sources.get(tile_id)
            elif tile.role == ImageRole.DATE_B.value:
                b = self.sources.get(tile_id)
        if a and b:
            return (a, b)
        return None


_index: ArchiveIndex | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _sensor_label(meta) -> tuple[str, str]:
    """(sensor, platform) from tags, falling back to modality."""
    tags = meta.tags or {}
    platform = tags.get("PLATFORM") or tags.get("platform") or ""
    modality = meta.modality.modality.value
    if modality == "sar":
        return ("Sentinel-1 RTC", platform or "Sentinel-1")
    if platform.lower().startswith("landsat"):
        return ("Landsat", platform)
    if platform.lower().startswith("sentinel-2") or platform.startswith("S2"):
        return ("Sentinel-2 L2A", platform or "Sentinel-2")
    return ("Sentinel-2 L2A", platform or "Sentinel-2")


def _discover() -> list[tuple[str, str, Path, dict[str, str]]]:
    """Enumerate (aoi_key, role, path, meta) for every archive source file."""
    settings = get_settings()
    found: list[tuple[str, str, Path, dict[str, str]]] = []

    # Curated samples: {preset_key}__{role}.tif
    samples_dir = settings.samples_dir
    for key, preset in PRESETS_BY_KEY.items():
        meta = SAMPLE_META.get(key, {"region": "India", "concept": "bare", "hint": "Scene"})
        info = {"place": preset.place, "region": meta["region"], "concept": meta["concept"], "hint": meta["hint"]}
        roles = list(preset.optical.keys())  # only optical is searchable
        for role in roles:
            path = samples_dir / asset_filename(key, role)
            if path.exists():
                found.append((key, role.value, path, info))

    # AOI probe pairs: probe_{aoi}__{date_a|date_b}.tif
    probe_dir = settings.cache_dir / "aoi_probe"
    if probe_dir.exists():
        for path in sorted(probe_dir.glob("probe_*.tif")):
            stem = path.stem  # probe_mohali__date_a
            if "__" not in stem:
                continue
            aoi_key, role = stem.rsplit("__", 1)
            info = AOI_META.get(
                aoi_key,
                {"place": aoi_key.replace("probe_", "").replace("_", " ").title(),
                 "region": "India", "concept": "bare", "hint": "Scene"},
            )
            found.append((aoi_key, role, path, info))

    return found


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_archive(force: bool = True) -> ArchiveStats:
    """Scan the source imagery into the archive index and render thumbnails."""
    settings = get_settings()
    settings.ensure_dirs()
    thumbs_dir = settings.archive_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    tiles: dict[str, ArchiveTile] = {}
    sources: dict[str, Path] = {}
    by_aoi: dict[str, list[ArchiveTile]] = {}
    aoi_info: dict[str, dict[str, str]] = {}

    for aoi_key, role, path, info in _discover():
        aoi_info.setdefault(aoi_key, info)
        tile_id = f"{aoi_key}__{role}"
        try:
            meta = read_metadata(path, original_filename=path.name)
        except Exception as exc:  # noqa: BLE001 - one bad file is not fatal
            logger.warning("archive: skipping %s (%s)", path.name, exc)
            continue

        thumb_path = thumbs_dir / f"{tile_id}.png"
        if force or not thumb_path.exists():
            try:
                build_thumbnail(path, meta, thumb_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("archive: thumbnail failed for %s (%s)", tile_id, exc)

        features = extract_features(path, meta)
        acq = meta.acquisition_date
        if hasattr(acq, "date"):
            acq = acq.date().isoformat()
        elif acq is not None:
            acq = str(acq)[:10]
        sensor, platform = _sensor_label(meta)
        tags_src = meta.tags or {}
        centroid = meta.geo.centroid_wgs84 or [0.0, 0.0]
        lon, lat = float(centroid[0]), float(centroid[1])
        bounds = [float(x) for x in (meta.geo.bounds_wgs84 or [lon, lat, lon, lat])]
        cloud = round(float(meta.nodata_fraction or 0.0), 4)

        result_tags: list[str] = [sensor]
        if features.concepts.water >= WATER_PRESENCE_THRESHOLD:
            result_tags.append("Water present")

        tile = ArchiveTile(
            tile_id=tile_id,
            aoi_key=aoi_key,
            role=role,
            place=info["place"],
            region=info["region"],
            lat=round(lat, 5),
            lon=round(lon, 5),
            bounds_wgs84=bounds,
            sensor=sensor,
            platform=platform,
            acquisition_date=acq,
            gsd_m=round(meta.geo.gsd_m, 2) if meta.geo.gsd_m else None,
            epsg=meta.geo.epsg,
            scene_id=tags_src.get("STAC_ITEM_ID") or tags_src.get("stac_item_id"),
            cloud_fraction=cloud,
            quality=round(max(0.0, 1.0 - cloud), 4),
            concepts=features.concepts,
            embedding=features.embedding(),
            thumbnail_url=f"/api/archive/thumb/{tile_id}.png",
            tags=result_tags,
        )
        tiles[tile_id] = tile
        sources[tile_id] = path.resolve()
        by_aoi.setdefault(aoi_key, []).append(tile)

    # Group tiles into AOIs.
    aois: dict[str, ArchiveAOI] = {}
    for aoi_key, aoi_tiles in by_aoi.items():
        info = aoi_info.get(aoi_key) or AOI_META.get(aoi_key) or {
            "place": aoi_tiles[0].place, "region": aoi_tiles[0].region,
            "concept": "bare", "hint": "Scene",
        }
        # Sort tiles by acquisition date (None last).
        aoi_tiles.sort(key=lambda t: (t.acquisition_date or "9999"))
        dates = [t.acquisition_date for t in aoi_tiles if t.acquisition_date]
        roles = {t.role for t in aoi_tiles}
        has_pair = ImageRole.DATE_A.value in roles and ImageRole.DATE_B.value in roles
        latest = aoi_tiles[-1]
        near_water = any(t.concepts.water >= WATER_PRESENCE_THRESHOLD for t in aoi_tiles) or info["concept"] == "water"
        concept: Concept = info["concept"]  # type: ignore[assignment]

        aois[aoi_key] = ArchiveAOI(
            aoi_key=aoi_key,
            place=info["place"],
            region=info["region"],
            lat=latest.lat,
            lon=latest.lon,
            bounds_wgs84=latest.bounds_wgs84,
            sensor=latest.sensor,
            primary_concept=concept,
            change_hint=info["hint"],
            dates=[d for d in dates],
            has_pair=has_pair,
            near_water=near_water,
            thumbnail_url=latest.thumbnail_url,
            tile_ids=[t.tile_id for t in aoi_tiles],
        )

    concept_counts: dict[str, int] = {}
    for aoi in aois.values():
        concept_counts[aoi.primary_concept] = concept_counts.get(aoi.primary_concept, 0) + 1
    all_dates = sorted([t.acquisition_date for t in tiles.values() if t.acquisition_date])
    stats = ArchiveStats(
        generated_at=datetime.now(timezone.utc).isoformat(),
        tile_count=len(tiles),
        aoi_count=len(aois),
        pair_count=sum(1 for a in aois.values() if a.has_pair),
        regions=sorted({a.region for a in aois.values()}),
        sensors=sorted({t.sensor for t in tiles.values()}),
        concepts=concept_counts,
        embedder=active_embedder_name(),
        date_range=[all_dates[0] if all_dates else None, all_dates[-1] if all_dates else None],
    )

    index = ArchiveIndex(tiles=tiles, aois=aois, sources=sources, stats=stats)
    _persist(index)
    global _index
    with _lock:
        _index = index
    logger.info(
        "archive built: %d tiles, %d AOIs (%d pairs), embedder=%s",
        stats.tile_count, stats.aoi_count, stats.pair_count, stats.embedder,
    )
    return stats


def _persist(index: ArchiveIndex) -> None:
    settings = get_settings()
    payload = {
        "stats": index.stats.model_dump(),
        "tiles": [t.model_dump() for t in index.tiles.values()],
        "aois": [a.model_dump() for a in index.aois.values()],
        "sources": {k: str(v) for k, v in index.sources.items()},
    }
    path = settings.archive_dir / INDEX_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_from_disk() -> ArchiveIndex | None:
    settings = get_settings()
    path = settings.archive_dir / INDEX_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tiles = {t["tile_id"]: ArchiveTile.model_validate(t) for t in payload["tiles"]}
        aois = {a["aoi_key"]: ArchiveAOI.model_validate(a) for a in payload["aois"]}
        sources = {k: Path(v) for k, v in payload.get("sources", {}).items()}
        stats = ArchiveStats.model_validate(payload["stats"])
        return ArchiveIndex(tiles=tiles, aois=aois, sources=sources, stats=stats)
    except Exception as exc:  # noqa: BLE001
        logger.warning("archive index unreadable (%s); will rebuild", exc)
        return None


def get_index(rebuild: bool = False) -> ArchiveIndex:
    """The process-wide archive index, built on first use if needed."""
    global _index
    with _lock:
        if _index is not None and not rebuild:
            return _index
    if not rebuild:
        loaded = _load_from_disk()
        if loaded is not None:
            with _lock:
                _index = loaded
            return loaded
    build_archive(force=rebuild)
    with _lock:
        assert _index is not None
        return _index


def thumbnail_path(tile_id: str) -> Path | None:
    settings = get_settings()
    # tile_id is validated by the route; still guard against traversal.
    safe = "".join(c for c in tile_id if c.isalnum() or c in "_-")
    candidate = settings.archive_dir / "thumbs" / f"{safe}.png"
    return candidate if candidate.exists() else None


__all__ = ["ArchiveIndex", "build_archive", "get_index", "thumbnail_path"]
