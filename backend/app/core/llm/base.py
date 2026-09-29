"""The provider interface, and the rules every provider obeys.

The interface is small on purpose. Everything that makes this system trustworthy
happens after the model returns: the contract validator rejects invented tools, and
the numeric audit discards any phrasing containing a figure that is not in the
ledger. A provider therefore needs to do exactly one thing, which is turn messages
into text, and needs to be honest about the three ways that can fail.

The three failures are distinguished because the caller does different things with
them. Rate limiting and transient server errors are worth trying the next model in
the chain for. An unavailable provider is not worth retrying at all. Anything else
is a fault worth surfacing.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Anything shaped like a bearer credential, whoever issued it. Applied to provider
# error text before it can reach a log or an API response, because an exception
# message is the most likely way a key escapes a process that is careful everywhere
# else.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenRouter, OpenAI, Anthropic and most others: a prefix then a long opaque
    # body. Deliberately broad; a false positive costs nothing but clarity.
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    # Underscore-prefixed keys, which the rules above all missed. Mistral issues
    # `mstrl_…`, Groq `gsk_…`, Hugging Face `hf_…`, and adding a provider should not
    # require remembering to extend this list, so the prefix is matched generically:
    # a short lowercase tag, a separator, then a long opaque body.
    #
    # Found by a test, not by inspection. A Mistral key went straight through
    # redaction into an error message, which is exactly the leak this module exists
    # to prevent and exactly the kind that goes unnoticed without a case for it.
    re.compile(r"\b[a-z][a-z0-9]{1,11}_[A-Za-z0-9\-]{20,}\b"),
)

REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Remove anything that looks like a credential from free text.

    Used on every provider error before it is raised, so no caller has to
    remember. A key that reaches a log file has leaked whether or not anyone
    intended to write it there.
    """
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTED, cleaned)
    return cleaned


class LlmError(RuntimeError):
    """A provider failed. The message is always redacted."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))


class LlmRateLimited(LlmError):
    """Throttled or temporarily overloaded: worth trying the next model."""


class LlmUnavailable(LlmError):
    """Not configured, or refused outright. Retrying will not help."""


@dataclass(frozen=True)
class Message:
    """One turn. ``role`` is ``system``, ``user`` or ``assistant``."""

    role: str
    content: str


@dataclass
class LlmResponse:
    """What came back, and enough provenance to record it in the trace."""

    text: str
    # The model that actually answered, which is not always the one asked for: a
    # chain walks past anything that is throttled. The trace records the one that
    # answered, or the record of who drafted a contract would be untrue.
    model: str
    provider: str
    # Token counts when the provider reports them, for cost visibility. Never
    # invented when it does not.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Anything else worth keeping, none of it secret.
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Whether a provider can be reached, with no secret in the answer."""

    provider: str
    configured: bool
    reachable: bool
    # The models that answered a real request, not the ones merely advertised.
    # Listing a model is not evidence it will generate; that lesson cost a day.
    usable_models: list[str] = field(default_factory=list)
    detail: str = ""
    # Populated when a key is present but the provider rejected it, which is a
    # different problem from having no key at all.
    rejected: bool = False
    # Reachable and authenticated, but out of allowance: a 429, a spent daily free
    # quota, or every model in the chain queueing. Distinguished from unreachable
    # because there is nothing wrong and nothing to fix, and because the two want
    # different words in front of a user.
    throttled: bool = False


class LlmProvider(ABC):
    """Turns messages into text. Knows nothing about contracts or imagery."""

    name: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        """Whether this provider has what it needs to be tried at all."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        """Generate text.

        ``schema`` asks for JSON matching it. Providers that cannot enforce a
        schema must still honour the request by instructing the model to return
        only JSON: the caller parses and validates either way, so a provider is
        never trusted to have got the shape right.

        Raises :class:`LlmUnavailable` when not configured, :class:`LlmRateLimited`
        when throttled, and :class:`LlmError` for anything else.
        """

    @abstractmethod
    def health(self) -> ProviderHealth:
        """Probe reachability without revealing the key."""


class NullProvider(LlmProvider):
    """No model at all, and a fully supported configuration.

    Every number in this system comes from a deterministic tool, contracts can be
    planned by the offline rule router, and the verdict's reasoning is assembled
    from the ledger without any phrasing help. So the honest thing for "no
    provider" to be is a provider that says so, rather than a special case
    threaded through the call sites.
    """

    name = "none"

    @property
    def configured(self) -> bool:
        return False

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        raise LlmUnavailable(
            "No language model is configured. Contracts are planned by the offline "
            "rule router and explanations are assembled from the evidence ledger; "
            "every measurement is unaffected."
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.name,
            configured=False,
            reachable=False,
            detail=(
                "No provider selected. The offline rule router drafts contracts and "
                "the deterministic reasoning explains verdicts."
            ),
        )


__all__ = [
    "REDACTED",
    "LlmError",
    "LlmProvider",
    "LlmRateLimited",
    "LlmResponse",
    "LlmUnavailable",
    "Message",
    "NullProvider",
    "ProviderHealth",
    "redact",
]
