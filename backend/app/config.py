"""Central configuration for the SatQuery ARC backend.

Every secret and tunable is read from the environment (via ``backend/.env``)
through this module. No credential is ever hardcoded elsewhere in the source
tree, and no secret value is ever logged or returned by an API route.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# satquery/backend
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"

APP_NAME = "SatQuery ARC"
APP_TAGLINE = "Archive Search & Verified Change"
APP_VERSION = "0.2.0"

# Problem-statement provenance, surfaced in the UI and the PDF report so that
# every citation on screen is traceable to a real source.
PROBLEM_STATEMENT_ID = "SIH26227"
PROBLEM_STATEMENT_TITLE = (
    "Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery"
)
PROBLEM_STATEMENT_ORG = "Smart India Hackathon 2026 - Space Technology"

# The research foundation SatQuery ARC builds on, surfaced in /api/provenance and
# the PDF report. Each entry is a real, citable source.
ARC_REFERENCES = {
    "retrieval": {
        "name": "RemoteCLIP",
        "venue": "IEEE TGRS 2024",
        "arxiv_id": "2306.11029",
        "url": "https://arxiv.org/abs/2306.11029",
        "use": (
            "Contrastive vision-language model for remote sensing: text-to-tile "
            "and image-to-image retrieval, Apache-2.0 weights, runs fully offline."
        ),
    },
    "change": {
        "name": "CCDC (Continuous Change Detection and Classification)",
        "venue": "Remote Sensing of Environment 2014",
        "doi": "10.1016/j.rse.2014.01.011",
        "url": "https://doi.org/10.1016/j.rse.2014.01.011",
        "use": (
            "Seasonal harmonic model of every clear observation; a change is "
            "proven by persistent anomalies and dated to the earliest usable image."
        ),
    },
    "index": {
        "name": "HNSW (Hierarchical Navigable Small World)",
        "authors": "Malkov & Yashunin",
        "arxiv_id": "1603.09320",
        "url": "https://arxiv.org/abs/1603.09320",
        "use": (
            "Approximate nearest-neighbour graph index over tile embeddings that "
            "grows incrementally as the archive is ingested, with no rebuild."
        ),
    },
    "pretraining": {
        "name": "SSL4EO-S12",
        "arxiv_id": "2211.07044",
        "url": "https://arxiv.org/abs/2211.07044",
        "use": (
            "Self-supervised Sentinel-1/2 pretraining corpus behind the multi-sensor "
            "image encoders."
        ),
    },
}


class Settings(BaseSettings):
    """Runtime settings, populated from ``backend/.env`` or the environment."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Which provider drafts contracts and phrases explanations ---
    #
    # No provider ever measures anything, whichever is selected. They read the
    # question and plan, or they phrase a ledger that is already complete. Every
    # number comes from the deterministic tools, so "none" is a fully supported
    # configuration rather than a degraded one.
    llm_provider: str = "openrouter"
    # A model that hangs must not hang the request. Past this, the offline rule
    # router takes over, which it is always able to do.
    llm_timeout_seconds: float = 25.0

    # --- OpenRouter: the default provider ---
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_model_fallbacks: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Sent as HTTP-Referer and X-Title. OpenRouter uses them for attribution in
    # its dashboard; neither is required and neither carries anything secret.
    openrouter_site_url: str = "http://localhost:5173"
    openrouter_app_name: str = APP_NAME

    @field_validator("openrouter_model_fallbacks", mode="before")
    @classmethod
    def _split_openrouter_models(cls, value: object) -> object:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    @property
    def openrouter_model_chain(self) -> list[str]:
        """The primary model followed by its fallbacks, de-duplicated."""
        chain: list[str] = []
        for name in [self.openrouter_model, *self.openrouter_model_fallbacks]:
            if name and name not in chain:
                chain.append(name)
        return chain

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    # --- Mistral: a second provider with its own allowance ---
    #
    # A second provider, not a second model. OpenRouter meters free usage per
    # account per day, so a chain of free OpenRouter models shares one allowance and
    # is exhausted together. A different vendor is the only thing that actually
    # gives the chain somewhere to go.
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_model_fallbacks: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    mistral_base_url: str = "https://api.mistral.ai/v1"

    @field_validator("mistral_model_fallbacks", mode="before")
    @classmethod
    def _split_mistral_models(cls, value: object) -> object:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    @property
    def mistral_model_chain(self) -> list[str]:
        chain: list[str] = []
        for name in [self.mistral_model, *self.mistral_model_fallbacks]:
            if name and name not in chain:
                chain.append(name)
        return chain

    @property
    def mistral_chat_url(self) -> str:
        return f"{self.mistral_base_url.rstrip('/')}/chat/completions"

    @property
    def has_mistral(self) -> bool:
        return bool(self.mistral_api_key.strip())

    @property
    def openrouter_chat_url(self) -> str:
        return f"{self.openrouter_base_url.rstrip('/')}/chat/completions"

    # --- Gemini: available, no longer the default ---
    gemini_api_key: str = ""
    # Verified working with structured output on 2026-09-21. The 2.5 family now
    # returns 404 "no longer available to new users" for newly issued keys.
    gemini_model: str = "gemini-3.7-flash"
    # Tried in order when the primary model is temporarily unavailable. The 3.x
    # models return 503 "experiencing high demand" intermittently, which would
    # otherwise fail a demo at random; a chain makes that a non-event.
    gemini_model_fallbacks: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "gemini-flash-lite-latest",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
        ]
    )

    @field_validator("gemini_model_fallbacks", mode="before")
    @classmethod
    def _split_models(cls, value: object) -> object:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    @property
    def gemini_model_chain(self) -> list[str]:
        """The primary model followed by its fallbacks, de-duplicated."""
        chain: list[str] = []
        for name in [self.gemini_model, *self.gemini_model_fallbacks]:
            if name and name not in chain:
                chain.append(name)
        return chain

    # --- Copernicus Data Space Ecosystem: real Sentinel-1 SAR ---
    cdse_username: str = ""
    cdse_password: str = ""

    # --- AWS: Earth Search sentinel-1-grd is requester-pays ---
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "eu-central-1"

    # --- Runtime ---
    satquery_env: str = "development"
    log_level: str = "INFO"
    # NoDecode stops pydantic-settings from attempting a JSON parse of the raw
    # dotenv value, which lets the validator below accept a plain
    # comma-separated list instead of requiring JSON array syntax in .env.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    force_offline_contract: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string from the .env file."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # --- Derived paths ---
    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def sessions_dir(self) -> Path:
        return DATA_DIR / "sessions"

    @property
    def samples_dir(self) -> Path:
        return DATA_DIR / "samples"

    @property
    def models_dir(self) -> Path:
        return DATA_DIR / "models"

    @property
    def cache_dir(self) -> Path:
        return DATA_DIR / "cache"

    @property
    def archive_dir(self) -> Path:
        """Where the searchable tile archive (index + thumbnails) lives."""
        return DATA_DIR / "archive"

    @property
    def review_dir(self) -> Path:
        """Where the analyst review queue and audit log are persisted."""
        return DATA_DIR / "review"

    # --- Credential presence (never exposes the value itself) ---
    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def has_provider_key(self) -> dict[str, bool]:
        """Every known provider and whether its key is present. Names only."""
        return {
            "mistral": self.has_mistral,
            "openrouter": self.has_openrouter,
            "gemini": self.has_gemini,
        }

    @property
    def provider_chain(self) -> list[str]:
        """The providers to try, in order, that actually have a key.

        ``LLM_PROVIDER`` accepts a comma-separated list, so
        ``mistral,openrouter`` means "ask Mistral, and if it is out of allowance ask
        OpenRouter". That is the only arrangement that survives a daily quota,
        because quotas are per account and a list of models from one vendor shares
        one.

        A name without a key is dropped rather than substituted, and the drop is
        visible in ``/api/health``. Quietly promoting a provider nobody asked for
        would make the trace's record of who drafted a contract untrue, and the
        trace is the product.
        """
        present = self.has_provider_key
        chain: list[str] = []
        for raw in (self.llm_provider or "").split(","):
            name = raw.strip().lower()
            if not name or name in chain:
                continue
            if name == "none":
                break
            if present.get(name):
                chain.append(name)
        return chain

    @property
    def requested_providers(self) -> list[str]:
        """What was asked for, key or not. Used to explain what was dropped."""
        seen: list[str] = []
        for raw in (self.llm_provider or "").split(","):
            name = raw.strip().lower()
            if name and name not in seen and name != "none":
                seen.append(name)
        return seen

    @property
    def selected_provider(self) -> str:
        """The provider that will be tried first, or ``none``.

        Retained as a single name because the trace, the contract cache key and the
        health summary each want one answer to "who is drafting". The rest of the
        chain is a fallback, and whichever member actually replies is recorded on
        the response itself rather than inferred from here.
        """
        chain = self.provider_chain
        return chain[0] if chain else "none"

    @property
    def has_cdse(self) -> bool:
        return bool(self.cdse_username.strip() and self.cdse_password.strip())

    @property
    def has_aws(self) -> bool:
        return bool(self.aws_access_key_id.strip() and self.aws_secret_access_key.strip())

    def ensure_dirs(self) -> None:
        """Create the runtime directories if they do not yet exist."""
        for path in (
            self.data_dir,
            self.sessions_dir,
            self.samples_dir,
            self.models_dir,
            self.cache_dir,
            self.archive_dir,
            self.review_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
