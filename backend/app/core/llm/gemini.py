"""Gemini, kept behind the same interface.

Not the default any more, and kept rather than deleted for a reason: a provider
abstraction with one implementation is not an abstraction, it is indirection. Having
a second real provider is what makes the seam honest, and it means switching is a
line in ``.env`` if OpenRouter is unavailable during a demonstration.

The behaviour preserved from the original implementation is the model chain, which
exists because of something measured rather than assumed: the 3.x models return 503
"experiencing high demand" and 429 at random, and a single model would fail a
demonstration on a coin flip. Listing models is also not evidence a model will
generate, which is why the health probe issues a real request.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings
from app.core.llm.base import (
    LlmError,
    LlmProvider,
    LlmRateLimited,
    LlmResponse,
    LlmUnavailable,
    Message,
    ProviderHealth,
)

logger = logging.getLogger(__name__)

# Substrings that mark a failure as worth trying the next model for.
TRANSIENT_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "429",
    "503",
    "504",
    "high demand",
    "overloaded",
)


def _is_transient(message: str) -> bool:
    return any(marker in message for marker in TRANSIENT_MARKERS)


class GeminiProvider(LlmProvider):
    """Google's generative API, through the google-genai SDK."""

    name = "gemini"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def configured(self) -> bool:
        return self._settings.has_gemini

    def _client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - the SDK is pinned
            raise LlmUnavailable(
                "the google-genai package is not installed"
            ) from exc
        return genai.Client(api_key=self._settings.gemini_api_key)

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        if not self.configured:
            raise LlmUnavailable(
                "GEMINI_API_KEY is not set, so Gemini cannot be used."
            )

        from google.genai import types as genai_types

        client = self._client()
        # Gemini takes the system instruction separately from the turns.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m.content for m in messages if m.role != "system"]

        config: dict[str, Any] = {"temperature": temperature}
        if system:
            config["system_instruction"] = system
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        transient: list[str] = []
        for model in self._settings.gemini_model_chain:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents="\n\n".join(turns),
                    config=genai_types.GenerateContentConfig(**config),
                )
            except Exception as exc:  # noqa: BLE001 - the SDK raises broadly
                message = f"{type(exc).__name__}: {exc}"
                if _is_transient(message):
                    transient.append(f"{model}: {message[:120]}")
                    logger.info("gemini model %s unavailable", model)
                    continue
                raise LlmError(message) from exc

            text = (getattr(response, "text", "") or "").strip()
            if not text:
                transient.append(f"{model}: empty completion")
                continue

            return LlmResponse(
                text=text, model=model, provider=self.name,
                detail={"asked_for": model},
            )

        raise LlmRateLimited(
            "Every configured Gemini model was unavailable. "
            + "; ".join(transient[:3])
        )

    def health(self) -> ProviderHealth:
        if not self.configured:
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail="GEMINI_API_KEY is not set.",
            )

        usable: list[str] = []
        problems: list[str] = []
        for model in self._settings.gemini_model_chain:
            try:
                self._probe(model)
                usable.append(model)
            except LlmError as exc:
                problems.append(f"{model}: {exc}")

        if usable:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=True,
                usable_models=usable,
                detail=f"{len(usable)} model(s) generated: {', '.join(usable)}.",
            )
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=False,
            detail="No configured model generated. " + "; ".join(problems[:3]),
        )

    def _probe(self, model: str) -> None:
        """One real generation. A model listing proves nothing."""
        from google.genai import types as genai_types

        client = self._client()
        try:
            response = client.models.generate_content(
                model=model,
                contents="Reply with the word ok.",
                config=genai_types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=16
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise LlmError(f"{type(exc).__name__}: {exc}") from exc
        if not (getattr(response, "text", "") or "").strip():
            raise LlmError("empty completion")


__all__ = ["TRANSIENT_MARKERS", "GeminiProvider"]
