"""Shared test fixtures.

Tests must not depend on outbound network access. The ``seeded_registry``
fixture injects synthetic probe results so the health route can be exercised
without contacting Gemini, Copernicus, or Earth Search.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config as config_module
from app.config import Settings, get_settings
from app.core import sessions as sessions_module
from app.core.credentials import ProbeResult, registry
from app.core.sessions import SessionStore, reset_store_for_tests
from app.main import create_app


@pytest.fixture
def settings_factory():
    """Build Settings without reading the real .env file."""

    def _make(**overrides: object) -> Settings:
        defaults: dict[str, object] = {
            "gemini_api_key": "",
            "gemini_model": "gemini-2.5-flash",
            "cdse_username": "",
            "cdse_password": "",
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "satquery_env": "test",
            "force_offline_contract": False,
        }
        defaults.update(overrides)
        return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]

    return _make


def _probe(component: str, status: str, detail: str = "seeded") -> ProbeResult:
    return ProbeResult(
        component=component,
        status=status,  # type: ignore[arg-type]
        detail=detail,
        checked_at=time.time(),
    )


@pytest.fixture
def seed_registry() -> Iterator[callable]:
    """Seed the credential registry cache, bypassing all network probes."""
    original = registry.results

    def _seed(**statuses: str) -> None:
        defaults = {
            "raster_stack": "ok",
            "gemini": "ok",
            "copernicus": "ok",
            "earth_search": "ok",
        }
        defaults.update(statuses)
        registry._results = {  # noqa: SLF001 - deliberate test seam
            name: _probe(name, status) for name, status in defaults.items()
        }

    yield _seed
    registry._results = original  # noqa: SLF001


@pytest.fixture
def client(seed_registry) -> Iterator[TestClient]:
    """A TestClient that does not run the lifespan, so no probes fire."""
    seed_registry()
    get_settings.cache_clear()
    app = create_app()
    # Not used as a context manager on purpose: that skips the lifespan hook
    # and therefore skips the startup network probes.
    yield TestClient(app)
    get_settings.cache_clear()


@pytest.fixture
def isolated_data(tmp_path, monkeypatch) -> Iterator[Path]:
    """Redirect the whole data directory into a temp tree.

    Settings derive sessions/samples/models/cache from the module-level DATA_DIR
    at call time, so patching it isolates all of them at once. Without this the
    contract cache persists between runs and a test asserting a freshly generated
    contract sees a cached one instead.
    """
    root = tmp_path / "data"
    for name in ("sessions", "samples", "models", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "DATA_DIR", root)
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture
def store(isolated_data) -> Iterator[SessionStore]:
    """Session store rooted in the isolated data directory."""
    original = sessions_module._store  # noqa: SLF001 - deliberate test seam
    yield reset_store_for_tests(isolated_data / "sessions")
    sessions_module._store = original  # noqa: SLF001


@pytest.fixture
def api(client: TestClient, store: SessionStore) -> TestClient:
    """TestClient whose session routes write into the temp store."""
    assert store is not None
    return client


@pytest.fixture
def make_context(store: SessionStore):
    """Build a ToolContext from one or more synthetic scenes.

    Usage::

        context = make_context({ImageRole.SINGLE: scene.path})
    """
    from app.core.readiness import evaluate_readiness
    from app.tools.base import ToolContext

    def _make(
        images: dict,
        *,
        parameters: dict | None = None,
        with_readiness: bool = False,
        target_classes: list[str] | None = None,
    ) -> ToolContext:
        record = store.create()
        for role, path in images.items():
            store.ingest(record.session_id, role, path, path.name, move=False)
        reloaded = store.load(record.session_id)
        return ToolContext(
            session=reloaded,
            store=store,
            readiness=evaluate_readiness(reloaded, store) if with_readiness else None,
            parameters=parameters or {},
            target_classes=target_classes or [],
        )

    return _make


@pytest.fixture
def tool_registry():
    """A fresh default tool registry, isolated from the process-wide one.

    Deliberately not named ``registry``: that name is already bound in this module
    to the credential-probe singleton, and a fixture of the same name would
    shadow it.
    """
    from app.core import registry as registry_module
    from app.core.registry import build_default_registry

    original = registry_module._registry  # noqa: SLF001 - deliberate test seam
    fresh = build_default_registry()
    registry_module._registry = fresh  # noqa: SLF001
    yield fresh
    registry_module._registry = original  # noqa: SLF001


@pytest.fixture
def sample_library(isolated_data) -> Path:
    """The samples directory inside the isolated data tree."""
    return isolated_data / "samples"
