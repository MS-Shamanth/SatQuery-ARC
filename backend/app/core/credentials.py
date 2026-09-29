"""Startup credential probes.

The point of this module is to fail loudly on day one rather than at the moment
a demo is being recorded. Each probe makes one cheap, read-only request and
records whether the credential actually works.

Nothing here ever logs or returns a secret value. Probe results carry only a
status, a human-readable detail string, and non-sensitive metadata.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# "limited" is deliberately not "error".
#
# A free tier that has spent its daily allowance answers 429, and a system whose
# every number comes from a deterministic tool is not broken by that: contracts are
# planned by the offline rule router and verdicts are explained from the ledger.
# Painting it red said the opposite, and the one thing a status light must not do is
# report a working system as a failed one.
ProbeStatus = Literal[
    "ok", "limited", "invalid", "missing", "error", "unknown", "skipped"
]

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

PROBE_TIMEOUT_SECONDS = 12.0
PROBE_TTL_SECONDS = 300.0


@dataclass
class ProbeResult:
    """Outcome of a single credential or dependency probe."""

    component: str
    status: ProbeStatus
    detail: str
    checked_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status in ("ok", "skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "detail": self.detail,
            "checked_at": self.checked_at,
            **({"extra": self.extra} if self.extra else {}),
        }


def _truncate(text: str, limit: int = 240) -> str:  # noqa: D401
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


async def _probe_generation(settings: Settings, model: str) -> tuple[bool, str]:
    """Attempt one tiny generation. Returns (worked, detail).

    Listing models is not enough. A model can appear in the catalogue and still
    refuse generateContent: the gemini-2.5 family returns 404 "no longer
    available to new users" for recently issued keys. Only a real call tells the
    truth, and this probe is what stops /api/health reporting false confidence.
    """
    url = f"{GEMINI_MODELS_URL}/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
        "generationConfig": {"maxOutputTokens": 8, "temperature": 0.0},
    }
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url, headers={"x-goog-api-key": settings.gemini_api_key}, json=body
            )
    except httpx.HTTPError as exc:
        return False, f"unreachable: {_truncate(str(exc), 120)}"

    if response.status_code == 200:
        return True, "generation succeeded"
    return False, f"HTTP {response.status_code} {_truncate(response.text, 140)}"


async def probe_gemini(settings: Settings) -> ProbeResult:
    """Verify the Gemini key by actually generating, not merely by listing.

    Walks the configured model chain and reports the first that works, so the
    health endpoint names the model the pipeline will really use.
    """
    if settings.selected_provider != "gemini":
        # Gemini is no longer the default provider. It stays implemented and
        # selectable, but it is not called unless it is the one in use: probing an
        # unused provider on every health check spends someone's quota to learn
        # nothing.
        return ProbeResult(
            component="gemini",
            status="skipped",
            detail=(
                "Not the selected provider. Set LLM_PROVIDER=gemini to use it; "
                f"the current provider is {settings.selected_provider}."
            ),
            extra={"selected": settings.selected_provider},
            checked_at=time.time(),
        )

    if not settings.has_gemini:
        return ProbeResult(
            component="gemini",
            status="missing",
            detail=(
                "No GEMINI_API_KEY set. The offline rule-based contract router "
                "will be used instead."
            ),
            checked_at=time.time(),
        )

    headers = {"x-goog-api-key": settings.gemini_api_key}
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(
                GEMINI_MODELS_URL, headers=headers, params={"pageSize": 200}
            )
    except httpx.HTTPError as exc:
        return ProbeResult(
            component="gemini",
            status="error",
            detail=f"Could not reach the Gemini API: {_truncate(str(exc))}",
            checked_at=time.time(),
        )

    if response.status_code != 200:
        return ProbeResult(
            component="gemini",
            status="invalid",
            detail=(
                f"Gemini rejected the key (HTTP {response.status_code}). "
                f"{_truncate(response.text)}"
            ),
            checked_at=time.time(),
            extra={"http_status": response.status_code},
        )

    payload = response.json()
    names = [
        str(model.get("name", "")).removeprefix("models/")
        for model in payload.get("models", [])
    ]

    chain = settings.gemini_model_chain
    tried: list[str] = []
    working: str | None = None
    detail = ""
    for model in chain:
        worked, outcome = await _probe_generation(settings, model)
        tried.append(f"{model}: {outcome}")
        if worked:
            working = model
            detail = outcome
            break

    if working is None:
        return ProbeResult(
            component="gemini",
            status="invalid",
            detail=(
                "The key is valid but no configured model would generate. "
                + "; ".join(tried[:3])
            ),
            checked_at=time.time(),
            extra={
                "model_chain": chain,
                "attempts": tried,
                "listed_model_count": len(names),
            },
        )

    return ProbeResult(
        component="gemini",
        status="ok",
        detail=(
            f"Key accepted and '{working}' generated successfully"
            + (
                f" after {len(tried) - 1} unavailable model(s)."
                if len(tried) > 1
                else "."
            )
        ),
        checked_at=time.time(),
        extra={
            "working_model": working,
            "requested_model": settings.gemini_model,
            "model_chain": chain,
            "attempts": tried,
            "listed_model_count": len(names),
        },
    )


async def probe_llm_provider(settings: Settings) -> ProbeResult:
    """Verify the configured provider by generating, and never echo its key.

    The provider's own ``health`` does the work, which keeps one definition of
    "reachable" for both this endpoint and the command-line check. Everything in
    ``detail`` and ``extra`` here is a model name, a count, or a redacted message:
    the key is never read in this function and nothing derived from it is returned,
    not its prefix and not its length.
    """
    from app.core.llm import get_provider
    from app.core.llm.base import redact

    provider = get_provider(settings)
    selected = settings.selected_provider

    if not provider.configured:
        wanted = (settings.llm_provider or "none").strip().lower()
        return ProbeResult(
            component="llm_provider",
            status="missing" if wanted in ("", "none") else "invalid",
            detail=(
                "No language model is configured. Contracts are planned by the "
                "offline rule router and verdicts are explained from the evidence "
                "ledger. Every measurement is unaffected."
                if wanted in ("", "none")
                else (
                    f"LLM_PROVIDER is {wanted!r} but its key is not set, so the "
                    "offline rule router will be used."
                )
            ),
            extra={"selected": selected, "requested": wanted},
            checked_at=time.time(),
        )

    # The provider's probe is synchronous httpx; run it off the event loop so a
    # slow endpoint cannot stall the health route.
    health = await asyncio.to_thread(provider.health)

    if health.reachable:
        status: ProbeStatus = "ok"
        detail = redact(health.detail)
    elif health.rejected:
        status = "invalid"
        detail = redact(health.detail)
    elif health.throttled:
        # Reachable, credentials fine, allowance spent. Nothing to fix and nothing
        # degraded about the measurements, so this is a note rather than a fault.
        status = "limited"
        detail = redact(health.detail)
    else:
        status = "error"
        detail = redact(health.detail)

    return ProbeResult(
        component="llm_provider",
        status=status,
        detail=_truncate(detail),
        extra={
            "provider": health.provider,
            "selected": selected,
            "usable_models": health.usable_models,
            "endpoint": (
                settings.openrouter_chat_url if selected == "openrouter" else ""
            ),
        },
        checked_at=time.time(),
    )


async def probe_copernicus(settings: Settings) -> ProbeResult:
    """Verify Copernicus Data Space credentials via an OAuth token request.

    The access token is discarded immediately; only the fact that one was
    issued is retained.
    """
    if not settings.has_cdse:
        return ProbeResult(
            component="copernicus",
            status="missing",
            detail=(
                "No CDSE credentials set. Real Sentinel-1 SAR is unavailable; "
                "the simulated-SAR fallback will be used."
            ),
            checked_at=time.time(),
        )

    form = {
        "grant_type": "password",
        "username": settings.cdse_username,
        "password": settings.cdse_password,
        "client_id": "cdse-public",
    }
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.post(CDSE_TOKEN_URL, data=form)
    except httpx.HTTPError as exc:
        return ProbeResult(
            component="copernicus",
            status="error",
            detail=f"Could not reach Copernicus identity service: {_truncate(str(exc))}",
            checked_at=time.time(),
        )

    if response.status_code != 200:
        return ProbeResult(
            component="copernicus",
            status="invalid",
            detail=(
                f"Copernicus rejected the credentials (HTTP {response.status_code}). "
                f"{_truncate(response.text)}"
            ),
            checked_at=time.time(),
            extra={"http_status": response.status_code},
        )

    body = response.json()
    return ProbeResult(
        component="copernicus",
        status="ok",
        detail="Credentials accepted; access token issued for Sentinel-1 retrieval.",
        checked_at=time.time(),
        extra={
            "token_type": body.get("token_type", "unknown"),
            "expires_in_seconds": body.get("expires_in"),
        },
    )


async def probe_earth_search(_settings: Settings) -> ProbeResult:
    """Check the unauthenticated Earth Search STAC endpoint for Sentinel-2."""
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{EARTH_SEARCH_URL}/collections/sentinel-2-l2a")
    except httpx.HTTPError as exc:
        return ProbeResult(
            component="earth_search",
            status="error",
            detail=f"Could not reach Earth Search STAC: {_truncate(str(exc))}",
            checked_at=time.time(),
        )

    if response.status_code != 200:
        return ProbeResult(
            component="earth_search",
            status="error",
            detail=f"Earth Search returned HTTP {response.status_code}.",
            checked_at=time.time(),
        )

    body = response.json()
    return ProbeResult(
        component="earth_search",
        status="ok",
        detail="Sentinel-2 L2A collection reachable without credentials.",
        checked_at=time.time(),
        extra={
            "collection": body.get("id"),
            "asset_count": len(body.get("item_assets", {})),
            "requester_pays": bool(
                body.get("summaries", {}).get("storage:requester_pays", [False])[0]
            )
            if body.get("summaries", {}).get("storage:requester_pays")
            else False,
        },
    )


def probe_raster_stack(_settings: Settings) -> ProbeResult:
    """Report the geospatial stack versions.

    This is the check that confirms the pip wheels brought their own GDAL, so
    that a missing system GDAL is detected immediately.
    """
    versions: dict[str, Any] = {}
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError as exc:  # pragma: no cover - dependency guaranteed
        return ProbeResult(
            component="raster_stack",
            status="error",
            detail=f"numpy unavailable: {exc}",
            checked_at=time.time(),
        )

    try:
        import rasterio

        versions["rasterio"] = rasterio.__version__
        versions["gdal"] = rasterio.__gdal_version__
    except ImportError as exc:
        return ProbeResult(
            component="raster_stack",
            status="error",
            detail=(
                "rasterio is not importable, so no GeoTIFF can be read: "
                f"{_truncate(str(exc))}"
            ),
            checked_at=time.time(),
            extra=versions,
        )

    for module_name, key in (
        ("skimage", "scikit_image"),
        ("shapely", "shapely"),
        ("pyproj", "pyproj"),
        ("PIL", "pillow"),
    ):
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[key] = "missing"

    missing = [key for key, value in versions.items() if value == "missing"]
    if missing:
        return ProbeResult(
            component="raster_stack",
            status="error",
            detail=f"Missing geospatial dependencies: {', '.join(missing)}.",
            checked_at=time.time(),
            extra=versions,
        )

    return ProbeResult(
        component="raster_stack",
        status="ok",
        detail=(
            f"rasterio {versions['rasterio']} on GDAL {versions['gdal']}; "
            "GeoTIFF reads available."
        ),
        checked_at=time.time(),
        extra=versions,
    )


class CredentialRegistry:
    """Caches probe results so /api/health stays fast."""

    def __init__(self, ttl_seconds: float = PROBE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._results: dict[str, ProbeResult] = {}
        self._lock = asyncio.Lock()

    @property
    def results(self) -> dict[str, ProbeResult]:
        return dict(self._results)

    def _is_stale(self) -> bool:
        if not self._results:
            return True
        oldest = min(
            (r.checked_at or 0.0) for r in self._results.values()
        )
        return (time.time() - oldest) > self._ttl

    async def refresh(self, settings: Settings | None = None) -> dict[str, ProbeResult]:
        settings = settings or get_settings()
        async with self._lock:
            sync_result = probe_raster_stack(settings)
            async_results = await asyncio.gather(
                probe_llm_provider(settings),
                probe_gemini(settings),
                probe_copernicus(settings),
                probe_earth_search(settings),
                return_exceptions=True,
            )

            collected: list[ProbeResult] = [sync_result]
            for name, outcome in zip(
                ("llm_provider", "gemini", "copernicus", "earth_search"),
                async_results,
                strict=True,
            ):
                if isinstance(outcome, BaseException):
                    logger.warning("Probe %s raised: %s", name, outcome)
                    collected.append(
                        ProbeResult(
                            component=name,
                            status="error",
                            detail=f"Probe raised: {_truncate(str(outcome))}",
                            checked_at=time.time(),
                        )
                    )
                else:
                    collected.append(outcome)

            self._results = {result.component: result for result in collected}
            for result in collected:
                logger.info(
                    "probe %-14s %-8s %s", result.component, result.status, result.detail
                )
            return dict(self._results)

    async def get(
        self, settings: Settings | None = None, force: bool = False
    ) -> dict[str, ProbeResult]:
        if force or self._is_stale():
            return await self.refresh(settings)
        return dict(self._results)


registry = CredentialRegistry()
