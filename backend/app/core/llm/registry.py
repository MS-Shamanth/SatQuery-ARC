"""Selecting a provider from configuration.

One place decides which provider is in use, so the rest of the system asks for
"the provider" and never branches on a name. Switching is a line in ``.env``.

Selection never silently substitutes. If ``LLM_PROVIDER=openrouter`` and no key is
present, the answer is the null provider rather than Gemini, because a trace that
recorded a contract as drafted by one model when another drafted it would be a
false record, and the trace is the product.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.config import Settings, get_settings
from app.core.llm.base import LlmProvider, NullProvider
from app.core.llm.chain import ChainProvider

logger = logging.getLogger(__name__)

_BUILDERS: dict[str, Callable[[Settings], LlmProvider]] = {}


def _builders() -> dict[str, Callable[[Settings], LlmProvider]]:
    """Provider constructors, imported lazily.

    Lazily because a provider's dependency should not be able to break process
    start-up for a configuration that does not use it: Gemini needs the google
    SDK, and a missing SDK must not stop an OpenRouter deployment from booting.
    """
    if _BUILDERS:
        return _BUILDERS

    def build_openrouter(settings: Settings) -> LlmProvider:
        from app.core.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(settings)

    def build_gemini(settings: Settings) -> LlmProvider:
        from app.core.llm.gemini import GeminiProvider

        return GeminiProvider(settings)

    def build_mistral(settings: Settings) -> LlmProvider:
        from app.core.llm.mistral import MistralProvider

        return MistralProvider(settings)

    _BUILDERS.update(
        {
            "mistral": build_mistral,
            "openrouter": build_openrouter,
            "gemini": build_gemini,
            "none": lambda _settings: NullProvider(),
        }
    )
    return _BUILDERS


_provider: LlmProvider | None = None


def get_provider(settings: Settings | None = None) -> LlmProvider:
    """The provider this deployment is configured to use."""
    global _provider
    if _provider is not None and settings is None:
        return _provider

    resolved = settings or get_settings()
    chain = resolved.provider_chain
    builders = _builders()

    if not chain:
        provider: LlmProvider = NullProvider()
        wanted = resolved.requested_providers
        if wanted:
            # Worth saying out loud: the difference between "no model wanted" and "a
            # model was wanted and its key is missing" is a configuration mistake.
            logger.warning(
                "LLM_PROVIDER names %s but none of them has a key, so no language "
                "model will be used. Contracts will be planned by the offline rule "
                "router.",
                ", ".join(wanted),
            )
    else:
        members = [builders[name](resolved) for name in chain if name in builders]
        # One provider is not wrapped. A chain of one adds a layer of indirection
        # to every error message for no benefit.
        provider = members[0] if len(members) == 1 else ChainProvider(members)

        dropped = [
            name for name in resolved.requested_providers if name not in chain
        ]
        if dropped:
            logger.warning(
                "LLM_PROVIDER names %s without a key, so %s was dropped from the "
                "chain. Using %s.",
                ", ".join(dropped),
                "it" if len(dropped) == 1 else "they",
                provider.name,
            )

    if settings is None:
        _provider = provider
    return provider


def available_providers(settings: Settings | None = None) -> dict[str, bool]:
    """Every known provider and whether it has what it needs.

    Names and readiness only. No key, no prefix, no length.
    """
    resolved = settings or get_settings()
    return {**resolved.has_provider_key, "none": True}


def reset_provider_for_tests(provider: LlmProvider | None = None) -> None:
    """Replace the process-wide provider. Test-only seam."""
    global _provider
    _provider = provider


__all__ = ["available_providers", "get_provider", "reset_provider_for_tests"]
