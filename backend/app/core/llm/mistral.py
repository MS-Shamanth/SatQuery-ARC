"""Mistral, over its OpenAI-compatible chat completions endpoint.

Added as a genuine second provider rather than a second model, and the distinction
is the whole point of it. OpenRouter's free tier meters
``free-models-per-day`` against the account, not the model, so a chain of free
OpenRouter models shares one allowance and runs out together: two models, one
quota, and both answering 429 within a second of each other.

Mistral is a different account with its own allowance, so a chain that crosses from
one provider to the other has somewhere to go when the first is spent. That is what
makes ``LLM_PROVIDER=mistral,openrouter`` worth more than any list of models.

Nothing here reads or reports the key beyond putting it in an Authorization header
built per request, and every error passes through the redaction in
:class:`~app.core.llm.base.LlmError` before it can reach a log.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.core.llm.base import Message
from app.core.llm.openai_compatible import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    """Chat completions through Mistral's API, with a model chain."""

    name = "mistral"
    label = "Mistral"
    key_env = "MISTRAL_API_KEY"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def _api_key(self) -> str:
        return self._settings.mistral_api_key

    @property
    def _chat_url(self) -> str:
        return self._settings.mistral_chat_url

    @property
    def _model_chain(self) -> list[str]:
        return self._settings.mistral_model_chain

    @property
    def _timeout_seconds(self) -> float:
        return self._settings.llm_timeout_seconds

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Mistral's JSON mode.

        It accepts ``{"type": "json_object"}`` across the range and only supports
        full ``json_schema`` on newer models, where an unrecognised schema is a 422
        rather than something quietly ignored. Asking for the mode it always honours
        and putting the shape in the prompt is the reliable combination: the caller
        parses and validates either way, so nothing is lost by not declaring the
        schema here, whereas a 422 would cost the draft entirely.
        """
        del schema  # Carried in the prompt instead; see _augment_messages.
        return {"type": "json_object"}

    def _augment_messages(
        self, messages: list[Message], schema: dict[str, Any]
    ) -> list[Message]:
        """Put the schema in the conversation, since the request cannot carry it.

        Without this the model is told to match a schema it has never seen. It
        responded by inventing a plausible envelope and describing the shape it
        expected back, which the validator then rejected: a provider that looked
        reachable and healthy but could not draft a single usable contract.
        """
        return [*messages, self._schema_message(schema)]


__all__ = ["MistralProvider"]
