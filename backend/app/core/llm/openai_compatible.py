"""The chat-completions protocol, shared by every provider that speaks it.

OpenRouter, Mistral, Groq, Together and OpenAI itself accept the same request and
return the same response. What differs between them is a base URL, a key, a model
list and at most a couple of headers, so that is all a subclass supplies.

This exists because the interesting code here is not the protocol. It is the
handling of the ways a remote model fails, and every line of it was bought with a
bug:

* The wall-clock deadline, because an httpx timeout only bounds silence and a
  queued free model dribbles keep-alive bytes down an open connection. A 25 second
  timeout held one request for over three minutes.
* Reading ``reasoning`` when ``content`` is empty, because reasoning models spend
  their budget thinking and leave the field a caller would naturally read blank.
* Rate limiting counted apart from failure, because a spent free allowance is not
  a broken provider and must not be reported as one.
* Redaction inside the error type rather than at the call sites, because the most
  likely way a key escapes is an exception message somebody logs.

Two copies of that would mean the next fix landing in one of them. A provider is a
subclass and a few lines of configuration.
"""

from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from typing import Any

import httpx

from app.core.llm.base import (
    LlmError,
    LlmProvider,
    LlmRateLimited,
    LlmResponse,
    LlmUnavailable,
    Message,
    ProviderHealth,
    redact,
)

logger = logging.getLogger(__name__)

# Worth trying the next model in the chain for. Everything else is a real fault and
# retrying it only wastes the user's time.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Kept short. A provider that is not answering is no reason to keep a user waiting
# when the offline rule router is sitting right there.
HEALTH_TIMEOUT_SECONDS = 20.0

# Roomy for a one-word answer, deliberately. Pools that include reasoning models
# spend tokens thinking before they write anything, and a tight budget makes them
# return nothing and look broken.
HEALTH_MAX_TOKENS = 200

# How much longer than one attempt the whole chain may take. Two attempts' worth:
# enough for a fallback to be worth configuring, short enough that a user is not
# left waiting on a queue of slow models.
CHAIN_BUDGET_FACTOR = 2.0


class OpenAICompatibleProvider(LlmProvider):
    """Chat completions over the OpenAI-compatible protocol."""

    name = "openai-compatible"
    # Shown in errors and health text. Never a key, never derived from one.
    label = "the provider"
    # Environment variable a subclass expects its key in, named in messages so a
    # misconfiguration says what to set.
    key_env = ""

    # -- what a subclass supplies -----------------------------------------
    @property
    @abstractmethod
    def _api_key(self) -> str:
        """The bearer token. Read per request and never stored on the instance."""

    @property
    @abstractmethod
    def _chat_url(self) -> str:
        """Fully qualified chat-completions endpoint."""

    @property
    @abstractmethod
    def _model_chain(self) -> list[str]:
        """Primary model first, then fallbacks, de-duplicated."""

    @property
    @abstractmethod
    def _timeout_seconds(self) -> float:
        """Per-attempt budget, in seconds."""

    def _extra_headers(self) -> dict[str, str]:
        """Non-secret headers this provider likes. Attribution and the like."""
        return {}

    @property
    def configured(self) -> bool:
        return bool(self._api_key.strip())

    # -- request building --------------------------------------------------
    def _headers(self) -> dict[str, str]:
        """Built fresh per request, so no attribute anywhere holds a credential
        that a repr, a debugger dump or a crash report could expose."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._extra_headers())
        return headers

    def _body(
        self,
        model: str,
        messages: list[Message],
        schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        prepared = (
            self._augment_messages(messages, schema) if schema is not None else messages
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in prepared
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if schema is not None:
            payload["response_format"] = self._response_format(schema)
        return payload

    def _augment_messages(
        self, messages: list[Message], schema: dict[str, Any]
    ) -> list[Message]:
        """Chance to put the schema where the model can see it.

        Providers that carry a schema in ``response_format`` need nothing here.
        Providers that only offer a generic JSON mode do: the caller's prompt says
        "reply with a JSON object matching the schema", and without the schema in
        either the request or the prompt the model is being asked to match something
        it was never shown. It duly invented a shape and the draft was rejected.
        """
        del schema
        return messages

    @staticmethod
    def _schema_message(schema: dict[str, Any]) -> Message:
        """The schema as a system turn, for providers that cannot send one."""
        return Message(
            role="system",
            content=(
                "The JSON object must conform to this JSON Schema. Use exactly "
                "these property names, include every required property, and add "
                "nothing else:\n" + json.dumps(schema, separators=(",", ":"))
            ),
        )

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """How this provider is asked for JSON.

        Honoured by models that support it and ignored by those that do not, which
        is safe either way: the caller parses and validates, so a provider is never
        trusted to have got the shape right.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "satquery_contract",
                "strict": False,
                "schema": schema,
            },
        }

    # -- the call ----------------------------------------------------------
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
                f"{self.key_env} is not set, so {self.label} cannot be used."
            )

        chain = self._model_chain
        if not chain:
            raise LlmUnavailable(f"No {self.label} model is configured.")

        budget = self._timeout_seconds
        transient: list[str] = []
        # A budget for the whole chain, not per model: each attempt is bounded
        # already, but a chain of eight would otherwise be allowed eight times the
        # configured wait, and the caller is a user staring at a spinner.
        overall = time.monotonic() + budget * CHAIN_BUDGET_FACTOR

        with httpx.Client(timeout=budget) as client:
            for model in chain:
                if time.monotonic() > overall:
                    transient.append(f"the chain ran out of time before {model}")
                    break
                try:
                    return self._one(
                        client, model, messages, schema, temperature, max_tokens
                    )
                except LlmRateLimited as exc:
                    # Worth the next model; the message is already redacted.
                    transient.append(f"{model}: {exc}")
                    logger.info("%s model %s unavailable: %s", self.name, model, exc)
                    continue

        raise LlmRateLimited(
            f"Every configured {self.label} model was unavailable. "
            + "; ".join(transient[:3])
        )

    def _one(
        self,
        client: httpx.Client,
        model: str,
        messages: list[Message],
        schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> LlmResponse:
        """One attempt, bounded by wall-clock time rather than only by silence.

        Read as a stream and timed against a deadline on purpose. An httpx timeout
        limits how long a single read may block and every byte received resets it,
        so a provider that holds a queued request open and dribbles keep-alives
        down it can run for minutes under a 25 second timeout. With the deadline,
        silence is caught by the read timeout and a slow dribble is caught here.
        """
        budget = self._timeout_seconds
        deadline = time.monotonic() + budget
        body = bytearray()

        try:
            with client.stream(
                "POST",
                self._chat_url,
                headers=self._headers(),
                json=self._body(model, messages, schema, temperature, max_tokens),
            ) as response:
                status = response.status_code
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if time.monotonic() > deadline:
                        raise LlmRateLimited(
                            f"{model} was still sending after {budget:.0f}s"
                        )
        except httpx.TimeoutException as exc:
            # A timeout is transient by nature, and the offline router is ready.
            raise LlmRateLimited(f"timed out after {budget:.0f}s ({exc!s})") from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"could not reach {self.label}: {exc!s}") from exc

        raw = bytes(body)

        if status in TRANSIENT_STATUS:
            raise LlmRateLimited(f"HTTP {status} {self._explain(raw)}")
        if status in (401, 403):
            # Deliberately does not echo the response body, which some gateways
            # helpfully include the offending credential in.
            raise LlmUnavailable(
                f"{self.label} rejected the credentials (HTTP {status}). "
                f"Check {self.key_env}."
            )
        if status >= 400:
            raise LlmError(f"HTTP {status} {self._explain(raw)}")

        return self._parse(raw, model)

    @staticmethod
    def _explain(raw: bytes) -> str:
        """A short reason from the body, redacted and truncated."""
        text = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except ValueError:
            return redact(text[:200])
        if not isinstance(payload, dict):
            return redact(text[:200])
        error = payload.get("error")
        if isinstance(error, dict):
            return redact(str(error.get("message", ""))[:200])
        if isinstance(error, str):
            return redact(error[:200])
        # Mistral reports some failures as a bare {"message": ...} or a
        # {"detail": [...]} validation list rather than an error object.
        for key in ("message", "detail"):
            if payload.get(key):
                return redact(str(payload[key])[:200])
        return redact(text[:200])

    def _parse(self, raw: bytes, asked_for: str) -> LlmResponse:
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise LlmError(
                f"{self.label} returned a body that is not JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise LlmError(f"{self.label} returned a body that is not JSON")

        # Some gateways report failures with HTTP 200 and an error object.
        error = payload.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            if code in TRANSIENT_STATUS or code == 429:
                raise LlmRateLimited(f"{code}: {message}")
            raise LlmError(str(message))

        choices = payload.get("choices") or []
        if not choices:
            raise LlmError(f"{self.label} returned no choices")

        message = choices[0].get("message") or {}
        refusal = (message.get("refusal") or "").strip()
        if refusal:
            raise LlmError(f"{asked_for} refused: {refusal[:200]}")

        text = self._content_of(message)

        if not text:
            finish = choices[0].get("finish_reason")
            if finish == "length":
                raise LlmError(
                    f"{asked_for} hit the token limit before producing any content"
                )
            raise LlmRateLimited(f"{asked_for} returned an empty completion")

        usage = payload.get("usage") or {}
        return LlmResponse(
            text=text,
            # What answered, not what was asked for. A router may substitute, and
            # the trace has to record the model that actually replied.
            model=str(payload.get("model") or asked_for),
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            detail={
                "asked_for": asked_for,
                "finish_reason": choices[0].get("finish_reason"),
                "generation_id": payload.get("id"),
            },
        )

    @staticmethod
    def _content_of(message: dict[str, Any]) -> str:
        """The assistant's text, wherever this provider put it.

        ``content`` is the obvious field and usually right. Reasoning models leave
        it empty and answer in ``reasoning``, especially on a tight budget, and some
        providers return ``content`` as a list of typed parts rather than a string.
        Reading only the obvious field made the provider fail intermittently for no
        visible reason, which was the first bug this layer had.
        """
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            ]
            joined = "".join(parts).strip()
            if joined:
                return joined
        return (message.get("reasoning") or "").strip()

    # -- health ------------------------------------------------------------
    def health(self) -> ProviderHealth:
        """Ask each configured model to say one word.

        A real generation, not a model listing. A catalogue entry is not evidence a
        model will answer: one can appear in the list and still 404 on use, which
        cost a day of debugging once already.
        """
        if not self.configured:
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=f"{self.key_env} is not set.",
            )

        usable: list[str] = []
        problems: list[str] = []
        rejected = False
        throttled = 0
        chain = self._model_chain

        with httpx.Client(timeout=HEALTH_TIMEOUT_SECONDS) as client:
            for model in chain:
                try:
                    self._one(
                        client,
                        model,
                        [Message(role="user", content="Reply with the word ok.")],
                        None,
                        0.0,
                        HEALTH_MAX_TOKENS,
                    )
                    usable.append(model)
                except LlmUnavailable as exc:
                    rejected = True
                    problems.append(f"{model}: {exc}")
                except LlmRateLimited as exc:
                    # Counted apart from failure. Every model throttled is a spent
                    # allowance, not a broken provider, and the difference decides
                    # whether the status light goes amber or red.
                    throttled += 1
                    problems.append(f"{model}: {exc}")
                except LlmError as exc:
                    problems.append(f"{model}: {exc}")

        if usable:
            detail = f"{len(usable)} model(s) generated: {', '.join(usable)}."
            if problems:
                detail += f" Unavailable: {'; '.join(problems[:2])}"
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=True,
                usable_models=usable,
                detail=detail,
            )

        if chain and throttled == len(chain):
            quota = any(
                marker in problem
                for problem in problems
                for marker in ("per-day", "per_day", "quota", "capacity")
            )
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                throttled=True,
                detail=(
                    (
                        f"The free allowance on this {self.label} account is spent, "
                        "so it resets rather than needing a fix. "
                        if quota
                        else "Every model in the chain is rate limited or queueing. "
                    )
                    + "Contracts are planned by the offline rule router and verdicts "
                    "are explained from the evidence ledger, so every measurement is "
                    "unaffected. "
                    + "; ".join(problems[:2])
                ),
            )

        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=False,
            rejected=rejected,
            detail=(
                "No configured model generated. "
                + "; ".join(problems[:3])
                + " The offline rule router drafts contracts meanwhile."
            ),
        )


def extract_json(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a completion.

    Models wrap JSON in prose or fences even when told not to, and a provider is
    not the right place to be strict about it: the validator downstream decides
    whether the content is acceptable, and refusing to parse recoverable output
    would push work onto the offline router for no gain.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```", 2)[1] if body.count("```") >= 2 else body[3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        body = body.strip().rstrip("`").strip()

    try:
        parsed = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise LlmError("the completion contained no JSON object") from None
        try:
            parsed = json.loads(body[start : end + 1])
        except ValueError as exc:
            raise LlmError("the completion's JSON could not be parsed") from exc

    if not isinstance(parsed, dict):
        raise LlmError("the completion's JSON was not an object")
    return parsed


__all__ = [
    "CHAIN_BUDGET_FACTOR",
    "HEALTH_MAX_TOKENS",
    "HEALTH_TIMEOUT_SECONDS",
    "TRANSIENT_STATUS",
    "OpenAICompatibleProvider",
    "extract_json",
]
