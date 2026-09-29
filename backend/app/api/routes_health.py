"""Health and provenance routes.

``/api/health`` is deliberately informative: it reports whether each external
dependency actually works, so a rejected API key or an unreachable data
provider is visible in the UI rather than discovered mid-demo.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    ARC_REFERENCES,
    PROBLEM_STATEMENT_ID,
    PROBLEM_STATEMENT_ORG,
    PROBLEM_STATEMENT_TITLE,
    get_settings,
)
from app.core.credentials import registry

router = APIRouter(tags=["health"])


def _overall_status(components: dict[str, dict[str, Any]]) -> str:
    """Collapse component statuses into one word.

    ``raster_stack`` is the only hard requirement: without rasterio there is no
    pipeline at all. Everything else degrades to a documented fallback, so a
    failure there is ``degraded`` rather than ``down``.
    """
    raster = components.get("raster_stack", {}).get("status")
    if raster != "ok":
        return "down"
    if any(c.get("status") in ("invalid", "error") for c in components.values()):
        return "degraded"
    # A spent allowance and an absent key land in the same place: a documented
    # fallback is carrying the work. "Operating with fallbacks" is the honest
    # summary of that, and it is amber rather than red because nothing is wrong.
    if any(
        c.get("status") in ("missing", "limited") for c in components.values()
    ):
        return "degraded"
    return "ok"


def _capabilities(components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Translate probe results into what the system can actually do right now.

    Reports the provider by name and nothing about its credentials. A reader of
    this endpoint should be able to tell whether a model is in play and which one
    answered, and should not be able to learn anything about the key.
    """
    llm = components.get("llm_provider", {})
    llm_ok = llm.get("status") == "ok"
    extra = llm.get("extra") or {}
    cdse_ok = components.get("copernicus", {}).get("status") == "ok"
    earth_ok = components.get("earth_search", {}).get("status") == "ok"
    settings = get_settings()

    if llm_ok and not settings.force_offline_contract:
        contract_source = "language-model"
    else:
        contract_source = "offline-rule-router"

    if cdse_ok:
        sar_source = "copernicus-sentinel-1-grd"
    elif settings.has_aws:
        sar_source = "aws-requester-pays-sentinel-1-grd"
    else:
        sar_source = "simulated-sar"

    # Which embedder actually backs semantic retrieval right now. RemoteCLIP is
    # used when its weights and runtime are present; otherwise the always-available
    # spectral-concept embedder carries retrieval, and this says so honestly.
    try:
        from app.core.embedding import active_embedder_name

        retrieval_source = active_embedder_name()
    except Exception:  # noqa: BLE001 - retrieval must never break health
        retrieval_source = "spectral-concept"

    return {
        "contract_source": contract_source,
        "narration_available": llm_ok,
        # Which provider, and which models actually generated. Not advertised
        # models: a catalogue entry is not evidence a model will answer.
        "llm_provider": settings.selected_provider,
        "llm_models": list(extra.get("usable_models") or []),
        "sar_source": sar_source,
        "optical_source": "earth-search-sentinel-2-l2a" if earth_ok else "synthetic",
        "geospatial_compute": components.get("raster_stack", {}).get("status") == "ok",
        # Archive search + verified change (SatQuery ARC).
        "retrieval_source": retrieval_source,
        "change_engine": "ccdc-ndbi-cva + confounder-firewall",
        "runs_offline": True,
    }


@router.get("/health")
async def health(
    refresh: bool = Query(
        default=False, description="Force re-probing of external dependencies"
    ),
) -> dict[str, Any]:
    settings = get_settings()
    probes = await registry.get(settings, force=refresh)
    components = {name: result.to_dict() for name, result in probes.items()}

    return {
        "status": _overall_status(components),
        "app": APP_NAME,
        "version": APP_VERSION,
        "environment": settings.satquery_env,
        "components": components,
        "capabilities": _capabilities(components),
    }


@router.get("/provenance")
async def provenance() -> dict[str, Any]:
    """Static provenance shown in the UI and embedded in the PDF report.

    Kept server-side so the citations on screen come from one authoritative
    place rather than being retyped into the frontend.
    """
    return {
        "problem_statement": {
            "id": PROBLEM_STATEMENT_ID,
            "title": PROBLEM_STATEMENT_TITLE,
            "organisation": PROBLEM_STATEMENT_ORG,
            "tagline": APP_TAGLINE,
        },
        "research_foundation": ARC_REFERENCES,
        "evaluation_datasets": [
            {"name": "OSCD (Onera Satellite Change Detection)", "used_for": "urban change detection benchmark"},
            {"name": "SpaceNet-7", "used_for": "multi-temporal building/settlement change"},
            {"name": "RSITMD", "used_for": "remote-sensing image-text retrieval"},
        ],
        "data_sources": [
            {
                "name": "Sentinel-2 L2A (Cloud-Optimized GeoTIFF)",
                "access": "Earth Search STAC API, unauthenticated",
                "url": "https://earth-search.aws.element84.com/v1",
                "note": (
                    "13 spectral bands plus the SCL scene-classification layer, "
                    "which provides the cloud and shadow mask"
                ),
            },
            {
                "name": "Sentinel-1 RTC / GRD",
                "access": "Copernicus Data Space Ecosystem / Planetary Computer",
                "url": "https://dataspace.copernicus.eu",
                "note": (
                    "Radar backscatter, which sees through cloud and provides an "
                    "independent confirmation of optical change"
                ),
            },
        ],
        "never_guess_rule": (
            "Quantitative results are produced by deterministic geospatial "
            "computation. Retrieval ranks the archive and the language model "
            "interprets the query and phrases the explanation; neither ever "
            "produces a measurement. Every change is verified against season, "
            "cloud and misregistration before it reaches an analyst, fully offline."
        ),
    }
