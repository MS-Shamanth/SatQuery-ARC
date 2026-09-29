"""Verified change for one archive AOI.

Given a place with two acquisitions, this drives the existing, unmodified
measurement pipeline - readiness gate, change engine, confounder firewall, and
the verdict/confidence engine - and distils the result into a compact
``ChangeVerification`` for the dashboard. Every number comes from that pipeline;
this module only selects and formats.

The heavy pipeline runs in a throwaway session so nothing about the archive's
source files is mutated, and the session is deleted afterwards.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from app.core.archive import ArchiveIndex, get_index
from app.core.contract import generate_contract
from app.core.embedding import active_embedder_name
from app.core.orchestrator import execute_run
from app.core.raster_io import sha256_of
from app.core.readiness import evaluate_readiness
from app.core.registry import get_registry
from app.core.sessions import get_store
from app.models.archive import (
    ChangeFrame,
    ChangeVerification,
    ConfidenceBar,
    ConfounderCheck,
    TimelineObservation,
)
from app.models.schemas import ImageRole
from app.tools.base import ToolContext

logger = logging.getLogger(__name__)

# Query templates per concept, phrased as the change question the AOI poses.
# Each query asserts the direction the AOI's change hint expects, so the verdict
# engine has a claim to support or refute (a direction-neutral "did it change?"
# is, correctly, always inconclusive). A wrong assertion yields a confident
# REFUTED, which is an equally settled and useful outcome.
_QUERY = {
    "built_up": "Has the built-up area increased between these two dates?",
    "water": "Has the surface water extent increased between these two dates?",
    "vegetation": "Has vegetation cover decreased between these two dates?",
    "bare": "Has the bare/cleared land increased between these two dates?",
}

_TARGET_LABEL = {
    "built_up": "built-up",
    "water": "water",
    "vegetation": "vegetation",
    "bare": "surface",
}

_CONFOUNDER_LABEL = {
    "seasonality": "Same-season baseline",
    "misregistration": "Co-registration",
    "cloud_shadow": "Cloud / haze mask",
    "radiometry": "Radiometric consistency",
    "sensor_mismatch": "Sensor match",
    "illumination_terrain": "Illumination / terrain",
    "sar_specific": "SAR geometry",
}

_cache: dict[str, ChangeVerification] = {}
_lock = threading.Lock()


def _measurements(trace) -> dict:
    return {m.key: m for m in trace.measurements}


def _index_from_trace(trace) -> str:
    for run in trace.tool_runs:
        if run.tool == "change-cva-engine":
            idx = run.parameters.get("index")
            if idx:
                return str(idx)
    return "NDVI"


def _timeline(index: ArchiveIndex, place: str, earliest_supported: str | None) -> list[TimelineObservation]:
    """All real acquisitions the archive holds for this place, as a timeline.

    Places with several probe variants (e.g. reservoirs sampled across years)
    yield a multi-year strip; a lone pair yields two points. Every date is a real
    Sentinel-2 acquisition, never a synthesised observation.
    """
    seen: dict[str, bool] = {}
    for tile in index.tiles.values():
        if tile.place != place or not tile.acquisition_date:
            continue
        clear = tile.quality >= 0.8
        # If any tile on a date is clear, treat the date as clear.
        seen[tile.acquisition_date] = seen.get(tile.acquisition_date, False) or clear
    obs = [
        TimelineObservation(
            date=date,
            clear=clear,
            is_earliest_supported=(date == earliest_supported),
        )
        for date, clear in sorted(seen.items())
    ]
    return obs


def _headline(label: str, direction: str, concept: str) -> str:
    target = _TARGET_LABEL.get(concept, "surface")
    if label == "supported":
        if direction == "increased":
            noun = {"built_up": "Built-up expansion", "water": "Water expansion",
                    "vegetation": "Vegetation gain"}.get(concept, "Increase")
        elif direction == "decreased":
            noun = {"built_up": "Built-up loss", "water": "Water reduction",
                    "vegetation": "Vegetation loss"}.get(concept, "Decrease")
        else:
            noun = "Change confirmed"
        return f"Confirmed change - {noun}"
    if label == "refuted":
        return f"No supported {target} change"
    if label == "inconclusive":
        return f"Inconclusive - {target} change not settled"
    return f"Unanswerable on this pair"


def verify_aoi(aoi_key: str, *, refresh: bool = False, actor: str = "system") -> ChangeVerification:
    """Run (or reuse) verified change for an AOI and return the compact result."""
    with _lock:
        if not refresh and aoi_key in _cache:
            return _cache[aoi_key]

    index = get_index()
    aoi = index.aois.get(aoi_key)
    if aoi is None:
        raise ValueError(f"Unknown AOI '{aoi_key}'.")
    pair = index.pair_paths(aoi_key)
    if pair is None:
        raise ValueError(f"AOI '{aoi_key}' has no dated pair to verify.")
    path_a, path_b = pair

    started = time.perf_counter()
    store = get_store()
    record = store.create()
    session_id = record.session_id
    try:
        store.ingest(session_id, ImageRole.DATE_A, path_a, path_a.name, move=False)
        store.ingest(session_id, ImageRole.DATE_B, path_b, path_b.name, move=False)
        record = store.load(session_id)

        readiness = evaluate_readiness(record, store)
        context = ToolContext(session=record, store=store, readiness=readiness)
        query = _QUERY.get(aoi.primary_concept, _QUERY["bare"])
        contract = generate_contract(query, context, get_registry(), force_offline=True)
        # Verified change is a sovereign, offline operation: no scene or measurement
        # leaves the building. Force the deterministic (non-LLM) explanation so the
        # verdict never depends on a network call.
        for plan in contract.tools:
            if plan.tool == "verdict-engine":
                plan.parameters = {**plan.parameters, "narrate": False}
        trace, _outcomes = execute_run(
            contract, context, get_registry(),
            emit=lambda _e: None, cancelled=lambda: False,
        )
    finally:
        store.delete(session_id)

    verification = _build_verification(aoi, index, trace, path_a, path_b, started)
    with _lock:
        _cache[aoi_key] = verification
    logger.info(
        "verified %s -> %s (%.0f%%) in %.0f ms",
        aoi_key, verification.verdict_label, verification.confidence * 100,
        verification.took_ms,
    )
    return verification


def _build_verification(aoi, index, trace, path_a: Path, path_b: Path, started: float) -> ChangeVerification:
    m = _measurements(trace)
    index_name = _index_from_trace(trace)
    concept = aoi.primary_concept
    target_label = _TARGET_LABEL.get(concept, "surface")

    verdict = trace.verdict
    if verdict is not None:
        label = verdict.label.value if hasattr(verdict.label, "value") else str(verdict.label)
        confidence = float(verdict.confidence)
        reasoning = verdict.reasoning
        direction = (
            verdict.measured_direction.value
            if hasattr(verdict.measured_direction, "value")
            else str(verdict.measured_direction)
        )
        components = [
            ConfidenceBar(
                name=c.name, label=c.label,
                measured=round(float(c.measured), 3),
                weight=round(float(c.weight), 3),
                contribution=round(float(c.contribution), 3),
            )
            for c in verdict.confidence_components
        ]
    else:
        label = "unanswerable"
        confidence = 0.0
        reasoning = "; ".join(trace.refusal_reasons) or "The pipeline produced no verdict."
        direction = "unspecified"
        components = []

    # Areas / delta.
    before = m.get("area_km2_before")
    after = m.get("area_km2_after")
    change = m.get("area_change_km2") or m.get("net_change_km2")
    pct = m.get("percentage_change")
    area_before = round(before.value, 3) if before else None
    area_after = round(after.value, 3) if after else None
    delta = round(change.value, 3) if change else None
    delta_display = (
        f"{'+' if (delta or 0) >= 0 else ''}{delta:.2f} km\u00b2" if delta is not None else "n/a"
    )

    # Per-date index means for the before/after chips.
    def _idx_val(role: str) -> float | None:
        key = f"{index_name.lower()}.{role}_mean"
        meas = m.get(key)
        return round(meas.value, 3) if meas else None

    dates = aoi.dates
    date_a = dates[0] if len(dates) >= 1 else None
    date_b = dates[-1] if len(dates) >= 2 else None

    # Confounder firewall.
    confounders: list[ConfounderCheck] = []
    cleared = 0
    for test in trace.confounders:
        kind = test.kind.value if hasattr(test.kind, "value") else str(test.kind)
        vlabel = test.verdict.value if hasattr(test.verdict, "value") else str(test.verdict)
        is_cleared = vlabel == "ruled_out"
        if is_cleared:
            cleared += 1
        confounders.append(
            ConfounderCheck(
                kind=kind,
                label=_CONFOUNDER_LABEL.get(kind, test.label),
                verdict=vlabel,
                cleared=is_cleared,
                measured=test.measured,
                detail=test.explanation,
            )
        )

    supported = label == "supported"
    earliest_supported = date_b if supported and date_b else None
    timeline = _timeline(index, aoi.place, earliest_supported)
    persists = sum(1 for o in timeline if o.clear and (o.date or "") >= (date_a or "")) if supported else 0

    checksum = sha256_of(path_b)[:12]

    return ChangeVerification(
        aoi_key=aoi.aoi_key,
        place=aoi.place,
        region=aoi.region,
        lat=aoi.lat,
        lon=aoi.lon,
        bounds_wgs84=aoi.bounds_wgs84,
        verdict_label=label,
        headline=_headline(label, direction, concept),
        direction=direction,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        index_name=index_name,
        target_label=target_label,
        area_before_km2=area_before,
        area_after_km2=area_after,
        delta_area_km2=delta,
        delta_display=delta_display,
        percentage_change=round(pct.value, 1) if pct else None,
        earliest_supported_date=earliest_supported,
        persists_in_scenes=persists,
        before=ChangeFrame(
            role="date_a", acquisition_date=date_a,
            scene_id=_scene_id(index, aoi, "date_a"),
            sensor=aoi.sensor,
            thumbnail_url=f"/api/archive/thumb/{aoi.aoi_key}__date_a.png",
            index_label=index_name, index_value=_idx_val("date_a"),
        ),
        after=ChangeFrame(
            role="date_b", acquisition_date=date_b,
            scene_id=_scene_id(index, aoi, "date_b"),
            sensor=aoi.sensor,
            thumbnail_url=f"/api/archive/thumb/{aoi.aoi_key}__date_b.png",
            index_label=index_name, index_value=_idx_val("date_b"),
        ),
        confidence_components=components,
        confounders=confounders,
        confounders_cleared=cleared,
        confounders_total=len(confounders),
        timeline=timeline,
        before_scene_id=_scene_id(index, aoi, "date_a"),
        after_scene_id=_scene_id(index, aoi, "date_b"),
        embedding_model=active_embedder_name(),
        change_model="CCDC-style delta + change-vector analysis",
        index_engine=f"change-cva-engine \u00b7 {index_name}",
        checksum=f"sha256:{checksum}",
        run_id=trace.run_id,
        took_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _scene_id(index: ArchiveIndex, aoi, role: str) -> str | None:
    tile = index.tiles.get(f"{aoi.aoi_key}__{role}")
    return tile.scene_id if tile else None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


__all__ = ["verify_aoi", "clear_cache"]
