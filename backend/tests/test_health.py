"""Tests for the health, capability, and provenance routes."""

from __future__ import annotations

import json

from app.config import ARC_REFERENCES, get_settings
from app.core.credentials import probe_gemini, probe_raster_stack


def test_root_returns_app_metadata(client) -> None:
    body = client.get("/").json()
    assert body["app"] == "SatQuery ARC"
    assert body["health"] == "/api/health"


def test_health_reports_every_component(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {
        "raster_stack",
        "gemini",
        "copernicus",
        "earth_search",
    }


def test_health_is_down_when_raster_stack_unavailable(client, seed_registry) -> None:
    """rasterio is the one hard dependency: without it there is no pipeline."""
    seed_registry(raster_stack="error")
    assert client.get("/api/health").json()["status"] == "down"


def test_health_is_degraded_when_gemini_rejected(client, seed_registry) -> None:
    seed_registry(gemini="invalid")
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    # The contract path must fall back rather than break.
    assert body["capabilities"]["contract_source"] == "offline-rule-router"
    assert body["capabilities"]["narration_available"] is False


def test_capabilities_report_simulated_sar_without_credentials(
    client, seed_registry
) -> None:
    seed_registry(copernicus="missing")
    caps = client.get("/api/health").json()["capabilities"]
    assert caps["sar_source"] == "simulated-sar"


def test_capabilities_report_real_sar_with_copernicus(client, seed_registry) -> None:
    seed_registry(copernicus="ok")
    caps = client.get("/api/health").json()["capabilities"]
    assert caps["sar_source"] == "copernicus-sentinel-1-grd"


def test_health_never_leaks_secret_values(client) -> None:
    """A credential value must never appear in an API response."""
    settings = get_settings()
    payload = json.dumps(client.get("/api/health").json())
    for secret in (
        settings.gemini_api_key,
        settings.cdse_password,
        settings.cdse_username,
    ):
        if secret:
            assert secret not in payload


def test_provenance_cites_research_foundation(client) -> None:
    body = client.get("/api/provenance").json()
    foundation = body["research_foundation"]
    assert foundation["retrieval"]["arxiv_id"] == ARC_REFERENCES["retrieval"]["arxiv_id"]
    assert foundation["retrieval"]["name"] == "RemoteCLIP"
    assert foundation["change"]["name"].startswith("CCDC")
    assert body["problem_statement"]["id"] == "SIH26227"
    assert body["problem_statement"]["tagline"] == "Archive Search & Verified Change"
    assert "never_guess_rule" in body


def test_raster_stack_probe_finds_gdal(settings_factory) -> None:
    """Confirms the pip wheels brought a working GDAL with them."""
    result = probe_raster_stack(settings_factory())
    assert result.status == "ok", result.detail
    assert result.extra["gdal"]
    assert result.extra["rasterio"]
    assert result.extra["scikit_image"] != "missing"


async def test_gemini_is_not_probed_unless_it_is_the_selected_provider(
    settings_factory,
) -> None:
    """An unused provider is not called, so its quota is not spent on nothing."""
    result = await probe_gemini(
        settings_factory(llm_provider="openrouter", openrouter_api_key="x")
    )
    assert result.status == "skipped"
    assert "not the selected provider" in result.detail.lower()


async def test_gemini_probe_reports_missing_key_without_network(
    settings_factory,
) -> None:
    result = await probe_gemini(
        settings_factory(llm_provider="gemini", gemini_api_key="")
    )
    assert result.status == "skipped"
    # With no key it is not selectable, so it is skipped rather than probed; either
    # way nothing reaches the network.
    assert "offline" in result.detail.lower() or "selected" in result.detail.lower()


async def test_the_provider_probe_never_reveals_the_key(settings_factory) -> None:
    """The one thing this endpoint must never do."""
    from app.core.credentials import probe_llm_provider

    secret = "sk-or-v1-" + "f" * 48
    result = await probe_llm_provider(
        settings_factory(llm_provider="none", openrouter_api_key=secret)
    )
    rendered = f"{result.detail} {result.extra}"
    assert secret not in rendered
    assert "sk-or-v1" not in rendered


async def test_no_provider_configured_is_reported_as_supported(
    settings_factory,
) -> None:
    """Running without a model is a configuration, not a fault."""
    from app.core.credentials import probe_llm_provider

    result = await probe_llm_provider(settings_factory(llm_provider="none"))
    assert result.status == "missing"
    assert "measurement is unaffected" in result.detail.lower()


def test_cors_origins_parse_from_comma_separated_string(settings_factory) -> None:
    settings = settings_factory(cors_origins="http://a.test, http://b.test")
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
