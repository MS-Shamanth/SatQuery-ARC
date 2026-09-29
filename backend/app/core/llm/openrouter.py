"""OpenRouter, over its OpenAI-compatible chat completions endpoint.

Everything about the protocol, the wall-clock deadline and the redaction lives in
:mod:`app.core.llm.openai_compatible`, because Mistral and most other vendors speak
the same thing and two copies of that logic would mean the next fix landing in one
of them.

What is specific to OpenRouter is here: the attribution headers its dashboard uses,
and the fact that ``openrouter/free`` is a router across a pool rather than a model.
That last point matters more than it sounds. The pool includes large reasoning
models, so the same request can be answered in two seconds or queued for minutes
depending on where it lands, and the model that replies is frequently not the one
asked for. The trace records the replying model for exactly that reason.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.core.llm.openai_compatible import (
    CHAIN_BUDGET_FACTOR,
    HEALTH_MAX_TOKENS,
    HEALTH_TIMEOUT_SECONDS,
    TRANSIENT_STATUS,
    OpenAICompatibleProvider,
    extract_json,
)


class OpenRouterProvider(OpenAICompatibleProvider):
    """Chat completions through OpenRouter, with a model chain."""

    name = "openrouter"
    label = "OpenRouter"
    key_env = "OPENROUTER_API_KEY"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def _api_key(self) -> str:
        return self._settings.openrouter_api_key

    @property
    def _chat_url(self) -> str:
        return self._settings.openrouter_chat_url

    @property
    def _model_chain(self) -> list[str]:
        return self._settings.openrouter_model_chain

    @property
    def _timeout_seconds(self) -> float:
        return self._settings.llm_timeout_seconds

    def _extra_headers(self) -> dict[str, str]:
        """Optional attribution. OpenRouter shows these in its dashboard; neither
        is required and neither carries anything secret."""
        settings = self._settings
        headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
        return headers


__all__ = [
    "CHAIN_BUDGET_FACTOR",
    "HEALTH_MAX_TOKENS",
    "HEALTH_TIMEOUT_SECONDS",
    "TRANSIENT_STATUS",
    "OpenRouterProvider",
    "extract_json",
]
