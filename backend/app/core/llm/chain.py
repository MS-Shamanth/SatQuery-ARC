"""Several providers, tried in order.

This exists because of a specific and initially surprising failure. OpenRouter
meters its free tier as ``free-models-per-day`` against the *account*, so a chain of
free OpenRouter models shares a single allowance: configuring a fallback model
bought nothing, and when the allowance ran out both entries returned the same 429
within a second of each other. Two models, one quota.

Crossing providers is the only arrangement that survives that, because a different
vendor is a different allowance. ``LLM_PROVIDER=mistral,openrouter`` means ask
Mistral, and if Mistral is spent or throttled, ask OpenRouter.

Two rules keep it honest.

**Whoever answers is recorded.** Each provider stamps its own name on the response,
so a contract drafted by the second member says so. A chain that reported the first
member's name would be a false record in the one artefact meant to be auditable.

**Falling through is for exhaustion, not for disagreement.** A rate limit or an
unavailable provider moves to the next member. A malformed response does not: that
is a real fault worth surfacing, and retrying it against a different vendor would
hide a bug in the prompt behind a second opinion.
"""

from __future__ import annotations

import logging

from app.core.llm.base import (
    LlmError,
    LlmProvider,
    LlmRateLimited,
    LlmResponse,
    LlmUnavailable,
    Message,
    ProviderHealth,
)
from typing import Any

logger = logging.getLogger(__name__)


class ChainProvider(LlmProvider):
    """Tries each provider in turn until one generates."""

    def __init__(self, members: list[LlmProvider]) -> None:
        if not members:
            raise ValueError("a chain needs at least one provider")
        self._members = members
        # Named for what it is, so a log line or a health payload reads sensibly.
        self.name = "+".join(member.name for member in members)

    @property
    def members(self) -> list[LlmProvider]:
        return list(self._members)

    @property
    def configured(self) -> bool:
        return any(member.configured for member in self._members)

    def complete(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        exhausted: list[str] = []

        for member in self._members:
            if not member.configured:
                continue
            try:
                response = member.complete(
                    messages,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (LlmRateLimited, LlmUnavailable) as exc:
                # Out of allowance, or refusing to be used. Both mean "ask someone
                # else"; the messages are already redacted.
                exhausted.append(f"{member.name}: {exc}")
                logger.info("provider %s unavailable, trying the next", member.name)
                continue

            if exhausted:
                # Worth a line: the run was drafted by the fallback, and the trace
                # will say so. Silence here would make that look like the default.
                logger.info(
                    "drafted by %s after %d provider(s) were unavailable",
                    member.name,
                    len(exhausted),
                )
            return response

        raise LlmRateLimited(
            "Every configured provider was unavailable. "
            + "; ".join(exhausted[:3])
            + " The offline rule router plans the contract instead."
        )

    def health(self) -> ProviderHealth:
        """The chain's health, which is the best of its members.

        Reachable if any member generates, because one working provider is all the
        system needs. Throttled only if every configured member is throttled, since
        that is the state where waiting is the remedy rather than fixing something.
        """
        reports = [
            (member, member.health())
            for member in self._members
            if member.configured
        ]
        if not reports:
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail="No provider in the chain has a key.",
            )

        usable: list[str] = []
        details: list[str] = []
        for member, report in reports:
            # Qualified by provider, because two vendors can offer a model with the
            # same name and the reader needs to know whose answered.
            usable.extend(f"{member.name}/{model}" for model in report.usable_models)
            details.append(f"{member.name}: {report.detail}")

        if usable:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=True,
                usable_models=usable,
                detail=" | ".join(details),
            )

        every_throttled = all(report.throttled for _, report in reports)
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=False,
            throttled=every_throttled,
            rejected=any(report.rejected for _, report in reports),
            detail=(
                (
                    "Every provider in the chain is out of allowance, so this resets "
                    "rather than needing a fix. Contracts are planned by the offline "
                    "rule router and verdicts are explained from the evidence "
                    "ledger, so every measurement is unaffected. "
                    if every_throttled
                    else "No provider in the chain generated. "
                )
                + " | ".join(details)
            ),
        )


__all__ = ["ChainProvider"]
