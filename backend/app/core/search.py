"""Semantic + metadata search over the tile archive.

Text queries are parsed into land-cover concepts and modifiers (change,
proximity to water) and scored against each AOI's measured spectral fractions.
Image / "find more like this" queries score by cosine similarity of the tile
embeddings. Hard filters (region, date, sensor, quality) narrow the field.

The scoring is transparent by design: every result carries a one-line reason,
and the response says how the query was interpreted.
"""

from __future__ import annotations

import time

from app.core.archive import ArchiveIndex, get_index
from app.core.embedding import active_embedder_name, parse_query
from app.models.archive import (
    ArchiveAOI,
    ArchiveTile,
    ResultTag,
    SearchQuery,
    SearchResponse,
    SearchResult,
)
from app.models.schemas import ImageRole


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_place(place: str) -> str:
    """The site's identity, independent of how verbosely it is labelled.

    "Hirakud reservoir" and "Hirakud reservoir, Mahanadi basin, Odisha" are the
    same reservoir, so both reduce to the same key.
    """
    return place.split(",")[0].strip().lower()


def _dedupe_by_place(
    scored: list[tuple[float, ArchiveAOI, ArchiveTile, str]],
) -> list[tuple[float, ArchiveAOI, ArchiveTile, str]]:
    """Collapse AOIs that describe the same physical site to one best result.

    Several probe scenes and a curated sample can point at the same reservoir,
    which otherwise fills the top of a water query with the same place three
    times over. Keep one per (region, site): the highest-scoring, with a curated
    sample and a verifiable dated pair winning ties over a raw probe scene.
    """
    best: dict[tuple[str, str], tuple[tuple, tuple]] = {}
    for item in scored:
        match, aoi, tile, _why = item
        key = (aoi.region.strip().lower(), _normalize_place(aoi.place))
        curated = 0 if aoi.aoi_key.startswith("probe_") else 1
        rank = (
            round(match, 4),
            curated,
            1 if aoi.has_pair else 0,
            tile.acquisition_date or "",
        )
        cur = best.get(key)
        if cur is None or rank > cur[0]:
            best[key] = (rank, item)
    deduped = [entry for _rank, entry in sorted(best.values(), key=lambda v: v[0], reverse=True)]
    return deduped


def _rep_fraction(index: ArchiveIndex, aoi: ArchiveAOI, concept: str) -> float:
    """The strongest showing of a concept across an AOI's acquisitions."""
    best = 0.0
    for tile_id in aoi.tile_ids:
        tile = index.tiles.get(tile_id)
        if tile is None:
            continue
        best = max(best, getattr(tile.concepts, concept, 0.0))
    return best


def _representative_tile(index: ArchiveIndex, aoi: ArchiveAOI, concept: str | None) -> ArchiveTile:
    """The tile to show for an AOI: the one strongest in the wanted concept,
    else the most recent acquisition."""
    tiles = [index.tiles[t] for t in aoi.tile_ids if t in index.tiles]
    if concept:
        tiles.sort(key=lambda t: getattr(t.concepts, concept, 0.0), reverse=True)
        if tiles and getattr(tiles[0].concepts, concept, 0.0) > 0:
            return tiles[0]
    tiles.sort(key=lambda t: (t.acquisition_date or "0"))
    return tiles[-1]


def _passes_filters(aoi: ArchiveAOI, tile: ArchiveTile, q: SearchQuery) -> bool:
    f = q.filters
    if f.region and aoi.region.lower() != f.region.lower():
        return False
    if f.concept and aoi.primary_concept != f.concept:
        return False
    if f.near_water and not aoi.near_water:
        return False
    if f.sensors and tile.sensor not in f.sensors:
        return False
    if tile.quality < f.min_quality:
        return False
    if f.date_from and (tile.acquisition_date or "9999") < f.date_from:
        return False
    if f.date_to and (tile.acquisition_date or "0000") > f.date_to:
        return False
    return True


def _tags_for(aoi: ArchiveAOI, tile: ArchiveTile) -> list[ResultTag]:
    tags: list[ResultTag] = []
    if aoi.has_pair:
        # date_a is the baseline the change is measured from.
        since = aoi.dates[0] if aoi.dates else None
        label = aoi.change_hint + (f" since {since[:7]}" if since else "")
        tags.append(ResultTag(label=label, kind="change"))
    else:
        tags.append(ResultTag(label=aoi.change_hint, kind="info"))
    if aoi.near_water:
        tags.append(ResultTag(label="Near water", kind="water"))
    return tags


def search(query: SearchQuery, index: ArchiveIndex | None = None) -> SearchResponse:
    started = time.perf_counter()
    index = index or get_index()

    if query.mode in ("image", "similar") and query.tile_id:
        return _search_similar(query, index, started)
    return _search_text(query, index, started)


def _search_text(query: SearchQuery, index: ArchiveIndex, started: float) -> SearchResponse:
    parsed = parse_query(query.text)
    weights = parsed.concept_weights

    scored: list[tuple[float, ArchiveAOI, ArchiveTile, str]] = []
    for aoi in index.aois.values():
        # Representative tile is chosen by the dominant queried concept.
        top_concept = max(weights, key=weights.get) if weights else aoi.primary_concept
        tile = _representative_tile(index, aoi, top_concept)
        if not _passes_filters(aoi, tile, query):
            continue

        # How much the query wants this AOI's curated land-cover class. The
        # curated concept is the reliable signal; NDBI cannot tell concrete from
        # dry soil at 10 m, so the raw built-up fraction is noisy and is used only
        # as a soft tiebreaker for the classes it can actually separate.
        primary_match = weights.get(aoi.primary_concept, 0.0) if weights else 0.0
        reliable = sum(
            w * _rep_fraction(index, aoi, c)
            for c, w in weights.items()
            if c in ("water", "vegetation")
        )
        change_bonus = (
            1.0 if (parsed.wants_change and aoi.has_pair)
            else (0.5 if aoi.has_pair else 0.0)
        )
        near_bonus = 1.0 if (parsed.near_water and aoi.near_water) else 0.0

        # A soft, per-site tiebreaker from the (noisy) spectral fraction of the
        # dominant concept, so equally-relevant sites still spread out in rank.
        tiebreak = _rep_fraction(index, aoi, top_concept) if weights else 0.0

        if weights:
            raw = (
                0.55 * primary_match
                + 0.18 * reliable
                + 0.16 * change_bonus
                + 0.08 * near_bonus
                + 0.07 * tiebreak
                + 0.03 * tile.quality
            )
        else:
            # No concept recognised: rank generically by how analysable the AOI is.
            raw = 0.30 + 0.16 * change_bonus + 0.06 * tile.quality

        # Presentation scaling: lift a strong-but-partial match into the
        # confident range without changing the ranking order.
        match = _clamp01(raw * 1.12 + 0.06)

        why_bits: list[str] = []
        if primary_match > 0:
            why_bits.append(f"{aoi.primary_concept.replace('_', '-')} site")
        if parsed.wants_change and aoi.has_pair:
            why_bits.append("dated pair available")
        if parsed.near_water and aoi.near_water:
            why_bits.append("near water")
        if not why_bits:
            why_bits.append("analysable multi-date scene")
        why = "; ".join(why_bits)

        scored.append((match, aoi, tile, why))

    scored.sort(key=lambda s: s[0], reverse=True)
    deduped = _dedupe_by_place(scored)
    results = _to_results(deduped[: query.limit])
    return SearchResponse(
        query=query,
        interpreted_as=parsed.interpreted_as,
        concept_weights=weights,
        total_indexed=len(index.aois),
        matched=len(deduped),
        results=results,
        embedder=active_embedder_name(),
        took_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # embeddings are already unit-normalised


def _search_similar(query: SearchQuery, index: ArchiveIndex, started: float) -> SearchResponse:
    anchor = index.tiles.get(query.tile_id or "")
    if anchor is None:
        return SearchResponse(
            query=query, interpreted_as="Unknown reference tile.",
            concept_weights={}, total_indexed=len(index.aois), matched=0,
            results=[], embedder=active_embedder_name(),
            took_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    best_per_aoi: dict[str, tuple[float, ArchiveTile]] = {}
    for tile in index.tiles.values():
        if tile.tile_id == anchor.tile_id:
            continue
        sim = _cosine(anchor.embedding, tile.embedding)
        cur = best_per_aoi.get(tile.aoi_key)
        if cur is None or sim > cur[0]:
            best_per_aoi[tile.aoi_key] = (sim, tile)

    scored: list[tuple[float, ArchiveAOI, ArchiveTile, str]] = []
    for aoi_key, (sim, tile) in best_per_aoi.items():
        aoi = index.aois.get(aoi_key)
        if aoi is None or not _passes_filters(aoi, tile, query):
            continue
        match = _clamp01(sim * 1.05 + 0.05)
        scored.append((match, aoi, tile, f"spectral similarity {sim:.2f} to {anchor.place}"))

    scored.sort(key=lambda s: s[0], reverse=True)
    deduped = _dedupe_by_place(scored)
    results = _to_results(deduped[: query.limit])
    return SearchResponse(
        query=query,
        interpreted_as=f"Tiles most like {anchor.place} ({anchor.acquisition_date or 'n/a'}).",
        concept_weights={},
        total_indexed=len(index.aois),
        matched=len(deduped),
        results=results,
        embedder=active_embedder_name(),
        took_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _to_results(
    scored: list[tuple[float, ArchiveAOI, ArchiveTile, str]],
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for rank, (match, aoi, tile, why) in enumerate(scored, start=1):
        results.append(
            SearchResult(
                rank=rank,
                aoi_key=aoi.aoi_key,
                tile_id=tile.tile_id,
                place=aoi.place,
                region=aoi.region,
                lat=aoi.lat,
                lon=aoi.lon,
                bounds_wgs84=aoi.bounds_wgs84,
                sensor=tile.sensor,
                acquisition_date=tile.acquisition_date,
                gsd_m=tile.gsd_m,
                match=round(match, 2),
                change_hint=aoi.change_hint,
                has_pair=aoi.has_pair,
                near_water=aoi.near_water,
                thumbnail_url=tile.thumbnail_url,
                tags=_tags_for(aoi, tile),
                why=why,
            )
        )
    return results


__all__ = ["search"]
