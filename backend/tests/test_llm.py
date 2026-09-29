"""Provider layer tests.

Two things are being protected here.

The key. It must not appear in an exception, a log line, a health payload, or
anywhere else a caller could plausibly print. Redaction lives in ``LlmError``'s
constructor rather than at the call sites, so the test that matters is that
raising an error containing a key produces an error that does not.

The truth of the trace. The trace records who drafted a contract, so selection
must never silently substitute one provider for another, and the recorded model
must be the one that actually answered rather than the one that was asked for.

Every test runs against a mocked transport. Nothing here touches the network, so
the suite is the same with or without a key in the environment.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from app.config import Settings
from app.core.llm.base import (
    REDACTED,
    LlmError,
    LlmRateLimited,
    LlmResponse,
    LlmUnavailable,
    Message,
    NullProvider,
    redact,
)
from app.core.llm.chain import ChainProvider
from app.core.llm.mistral import MistralProvider
from app.core.llm.openrouter import OpenRouterProvider, extract_json
from app.core.llm.registry import available_providers, get_provider

# Matches the redactor's ``sk-`` shape without being a real key. Deliberately not
# the ``sk-or-v1-`` + 64-hex form of an actual OpenRouter key, so it exercises the
# redaction path without tripping secret scanners on push.
FAKE_KEY = "sk-or-test-not-a-real-key-fixture-only"

ASK = [Message(role="user", content="say ok")]


def settings_for(**overrides) -> Settings:
    """Settings built from defaults and the overrides below, and nothing else.

    ``_env_file=None`` is the load-bearing part. Without it pydantic-settings reads
    ``backend/.env``, and every field this helper does not name explicitly comes
    from whatever the developer happens to have configured. That is not
    hypothetical: adding one fallback model to ``.env`` broke the health test,
    which asserts exactly which models answered. A suite whose result depends on an
    untracked local file is not telling you about your code.

    Every field that changes provider behaviour is pinned here for the same reason,
    so a future addition to ``.env`` cannot reach in either.
    """
    base = {
        "llm_provider": "openrouter",
        "openrouter_api_key": FAKE_KEY,
        "openrouter_model": "openrouter/free",
        "openrouter_model_fallbacks": [],
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_site_url": "",
        "openrouter_app_name": "SatQuery ARC",
        "gemini_api_key": "",
        "gemini_model": "",
        "gemini_model_fallbacks": [],
        "mistral_api_key": "",
        "mistral_model": "mistral-small-latest",
        "mistral_model_fallbacks": [],
        "mistral_base_url": "https://api.mistral.ai/v1",
        "force_offline_contract": False,
        "llm_timeout_seconds": 5,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def transport_returning(
    status: int, payload: dict | str, *, capture: list | None = None
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def completion(content: str, *, model: str = "vendor/some-model:free", **extra) -> dict:
    message = {"role": "assistant", "content": content}
    message.update(extra)
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


@pytest.fixture
def mock_httpx(monkeypatch):
    """Point every httpx.Client the provider builds at a mock transport."""

    def install(transport: httpx.MockTransport) -> None:
        original = httpx.Client.__init__

        def patched(self, *args, **kwargs):
            kwargs["transport"] = transport
            original(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", patched)

    return install


# -- redaction ------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        FAKE_KEY,
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
        f"Bearer {FAKE_KEY}",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
        # Underscore-prefixed keys. Every one of these went through untouched until
        # a Mistral key turned up in an error message and proved the gap.
        #
        # Structure only. A fixture derived from a real key by tweaking a character
        # is a committed key with extra steps, which is what the first version of
        # this line was: anyone reading it could recover the original.
        "mstrl_" + "A" * 38,
        "gsk_" + "B" * 36,
        "hf_" + "C" * 30,
    ],
)
def test_anything_shaped_like_a_credential_is_removed(secret):
    cleaned = redact(f"upstream said: {secret} was rejected")
    assert secret not in cleaned
    assert REDACTED in cleaned


def test_redaction_happens_in_the_error_constructor_not_at_the_call_sites():
    """The likeliest leak is an exception someone logs, so it cannot be opt-in."""
    for kind in (LlmError, LlmRateLimited, LlmUnavailable):
        raised = kind(f"Authorization: Bearer {FAKE_KEY} failed")
        assert FAKE_KEY not in str(raised)
        assert REDACTED in str(raised)


def test_a_key_in_a_provider_error_body_never_reaches_the_caller(mock_httpx):
    """Some gateways echo the offending credential back in the error body."""
    mock_httpx(
        transport_returning(
            400, {"error": {"message": f"invalid token {FAKE_KEY} supplied"}}
        )
    )
    provider = OpenRouterProvider(settings_for())

    with pytest.raises(LlmError) as raised:
        provider.complete(ASK)

    assert FAKE_KEY not in str(raised.value)


def test_the_provider_holds_no_attribute_containing_the_key(mock_httpx):
    """Headers are built per request, so no repr or debugger dump exposes one."""
    provider = OpenRouterProvider(settings_for())
    for value in vars(provider).values():
        assert FAKE_KEY not in repr(value) or isinstance(value, Settings)
    # It is on the settings object, which is the one place it has to live.
    assert provider.configured


# -- status handling ------------------------------------------------------


def test_the_authorization_header_is_sent_but_is_not_in_the_body(mock_httpx):
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion("ok"), capture=seen))

    OpenRouterProvider(settings_for()).complete(ASK)

    assert len(seen) == 1
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in seen[0].content.decode()


def test_a_429_is_rate_limiting_so_the_caller_can_fall_back(mock_httpx):
    mock_httpx(transport_returning(429, {"error": {"message": "slow down"}}))
    with pytest.raises(LlmRateLimited):
        OpenRouterProvider(settings_for()).complete(ASK)


@pytest.mark.parametrize("status", [500, 502, 503, 504, 408])
def test_server_side_failures_are_treated_as_transient(mock_httpx, status):
    mock_httpx(transport_returning(status, {"error": {"message": "upstream"}}))
    with pytest.raises(LlmRateLimited):
        OpenRouterProvider(settings_for()).complete(ASK)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_unavailable_rather_than_transient(mock_httpx, status):
    """Retrying a refused credential wastes the user's time and never succeeds."""
    mock_httpx(transport_returning(status, {"error": {"message": "no"}}))
    with pytest.raises(LlmUnavailable, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_an_error_object_under_http_200_is_still_an_error(mock_httpx):
    """OpenRouter reports some failures with a 200 and an error body."""
    mock_httpx(
        transport_returning(200, {"error": {"code": 429, "message": "rate limited"}})
    )
    with pytest.raises(LlmRateLimited):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_a_timeout_is_transient_because_the_offline_router_is_ready(mock_httpx):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("took too long", request=request)

    mock_httpx(httpx.MockTransport(handler))
    with pytest.raises(LlmRateLimited, match="timed out"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_a_provider_that_dribbles_forever_is_cut_off_at_the_budget(mock_httpx):
    """The bug this exists for: a queued free model held a contract request for
    three minutes under a 25 second timeout, because every keep-alive byte reset
    the read clock. LLM_TIMEOUT_SECONDS has to bound the wall clock, not silence."""

    def slow_stream():
        # Never finishes, and never goes quiet, so only a deadline stops it.
        while True:
            time.sleep(0.02)
            yield b" "

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow_stream())

    mock_httpx(httpx.MockTransport(handler))

    started = time.monotonic()
    with pytest.raises(LlmRateLimited, match="still sending"):
        OpenRouterProvider(settings_for(llm_timeout_seconds=1)).complete(ASK)

    # Bounded by the budget and the chain factor, not by the stream.
    assert time.monotonic() - started < 6.0


def test_a_missing_key_is_unavailable_before_any_request_is_made(mock_httpx):
    def must_not_run(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("a request was made without a key")

    mock_httpx(httpx.MockTransport(must_not_run))
    provider = OpenRouterProvider(settings_for(openrouter_api_key=""))
    assert not provider.configured
    with pytest.raises(LlmUnavailable, match="OPENROUTER_API_KEY"):
        provider.complete(ASK)


# -- parsing --------------------------------------------------------------


def test_the_recorded_model_is_the_one_that_answered_not_the_one_asked_for(mock_httpx):
    """`openrouter/free` routes across a pool, and the trace must name the router's
    choice. Recording the request would make the trace's account of who drafted a
    contract untrue."""
    mock_httpx(transport_returning(200, completion("ok", model="vendor/actual:free")))

    result = OpenRouterProvider(settings_for()).complete(ASK)

    assert isinstance(result, LlmResponse)
    assert result.model == "vendor/actual:free"
    assert result.detail["asked_for"] == "openrouter/free"
    assert result.provider == "openrouter"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 3


def test_an_empty_content_field_falls_back_to_the_reasoning_field(mock_httpx):
    """Reasoning models in the free pool leave `content` empty and answer in
    `reasoning`. Not reading it made the provider fail intermittently for no
    visible reason, which was the first bug this layer had."""
    mock_httpx(
        transport_returning(200, completion("", reasoning="the answer is ok"))
    )
    result = OpenRouterProvider(settings_for()).complete(ASK)
    assert result.text == "the answer is ok"


def test_a_completion_empty_in_both_fields_is_transient(mock_httpx):
    mock_httpx(transport_returning(200, completion("", reasoning="")))
    with pytest.raises(LlmRateLimited, match="empty completion"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_hitting_the_token_limit_with_nothing_written_says_so(mock_httpx):
    payload = completion("")
    payload["choices"][0]["finish_reason"] = "length"
    mock_httpx(transport_returning(200, payload))
    with pytest.raises(LlmError, match="token limit"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_a_refusal_is_reported_as_a_refusal(mock_httpx):
    mock_httpx(transport_returning(200, completion("", refusal="I will not")))
    with pytest.raises(LlmError, match="refused"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_a_non_json_body_is_a_provider_fault_not_a_crash(mock_httpx):
    mock_httpx(transport_returning(200, "<html>gateway error</html>"))
    with pytest.raises(LlmError, match="not JSON"):
        OpenRouterProvider(settings_for()).complete(ASK)


def test_a_schema_is_requested_when_one_is_given(mock_httpx):
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion('{"target": "water"}'), capture=seen))

    schema = {"type": "object", "properties": {"target": {"type": "string"}}}
    OpenRouterProvider(settings_for()).complete(ASK, schema=schema)

    body = json.loads(seen[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["model"] == "openrouter/free"


# -- extracting JSON from prose ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"target": "water"}',
        '```json\n{"target": "water"}\n```',
        '```\n{"target": "water"}\n```',
        'Here is the plan:\n{"target": "water"}\nHope that helps.',
    ],
)
def test_json_is_recovered_from_the_wrappers_models_actually_use(text):
    assert extract_json(text) == {"target": "water"}


@pytest.mark.parametrize("text", ["no object here", "[1, 2, 3]", "{broken"])
def test_unrecoverable_output_is_an_error_rather_than_a_guess(text):
    with pytest.raises(LlmError):
        extract_json(text)


# -- selection ------------------------------------------------------------


def test_a_provider_is_never_silently_substituted_for_another():
    """Asking for OpenRouter with no key gives no model, not Gemini.

    A trace recording a contract as drafted by one model when another drafted it
    would be a false record, and the trace is the product.
    """
    settings = settings_for(
        openrouter_api_key="", gemini_api_key="AIzaSyPretendThisIsARealKey123456"
    )
    assert settings.has_gemini
    assert settings.selected_provider == "none"
    assert isinstance(get_provider(settings), NullProvider)


def test_the_selected_provider_is_the_one_configured_and_keyed():
    assert settings_for().selected_provider == "openrouter"
    assert get_provider(settings_for()).name == "openrouter"


def test_gemini_is_still_selectable_but_only_when_asked_for():
    """Kept as a real second implementation: an abstraction with one provider is
    not an abstraction. Nothing reaches it unless LLM_PROVIDER says so."""
    keyed = settings_for(
        llm_provider="gemini",
        gemini_api_key="AIzaSyPretendThisIsARealKey123456",
        openrouter_api_key=FAKE_KEY,
    )
    assert keyed.selected_provider == "gemini"
    assert settings_for().selected_provider == "openrouter"


def test_no_provider_configured_is_a_supported_state_not_a_degraded_one():
    settings = settings_for(llm_provider="none", openrouter_api_key="")
    provider = get_provider(settings)
    assert isinstance(provider, NullProvider)
    assert not provider.configured

    health = provider.health()
    assert not health.reachable
    # It has to say what still works, because everything measured still does.
    assert "offline rule router" in health.detail

    with pytest.raises(LlmUnavailable, match="every measurement is unaffected"):
        provider.complete(ASK)


def test_availability_reports_names_only_and_never_a_key():
    available = available_providers(settings_for())
    assert available == {
        "mistral": False,
        "openrouter": True,
        "gemini": False,
        "none": True,
    }
    assert FAKE_KEY not in json.dumps(available)


# -- health ---------------------------------------------------------------


def test_health_generates_rather_than_listing_models(mock_httpx):
    """A catalogue entry is not evidence a model will answer: a model can be
    advertised and still return 404 on use."""
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion("ok"), capture=seen))

    health = OpenRouterProvider(settings_for()).health()

    assert health.reachable
    assert health.usable_models == ["openrouter/free"]
    assert seen and seen[0].url.path.endswith("/chat/completions")


def test_health_on_a_rejected_key_says_rejected_without_quoting_it(mock_httpx):
    mock_httpx(transport_returning(401, {"error": {"message": f"bad {FAKE_KEY}"}}))

    health = OpenRouterProvider(settings_for()).health()

    assert not health.reachable
    assert health.rejected
    assert FAKE_KEY not in health.detail
    assert "offline rule router" in health.detail


def test_health_with_no_key_is_reported_as_unconfigured(mock_httpx):
    def must_not_run(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("a request was made without a key")

    mock_httpx(httpx.MockTransport(must_not_run))
    health = OpenRouterProvider(settings_for(openrouter_api_key="")).health()

    assert not health.configured
    assert not health.reachable
    assert not health.usable_models


# -- a spent allowance is not a fault -------------------------------------


def test_every_model_rate_limited_reports_throttled_not_unreachable(mock_httpx):
    """The distinction the status light depends on.

    A free tier that has spent its daily allowance answers 429 to everything. The
    provider is reachable, the key is fine, and every measurement still comes from
    a deterministic tool. Reporting that as an error painted a working system red
    and told the operator to fix something that needs no fixing.
    """
    mock_httpx(
        transport_returning(
            429,
            {
                "error": {
                    "message": (
                        "Rate limit exceeded: free-models-per-day. Add 10 credits "
                        "to unlock 1000 free model requests per day"
                    )
                }
            },
        )
    )

    health = OpenRouterProvider(
        settings_for(openrouter_model_fallbacks=["vendor/second:free"])
    ).health()

    assert not health.reachable
    assert health.throttled
    # Not a rejected credential, which is a real fault with a real action.
    assert not health.rejected
    assert health.usable_models == []
    # It says what still works and that waiting is the remedy.
    assert "allowance" in health.detail
    assert "offline rule router" in health.detail
    assert "every measurement is unaffected" in health.detail.lower()


def test_one_model_throttled_but_another_answering_is_still_ok(mock_httpx):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion("ok", model="vendor/second:free"))

    mock_httpx(httpx.MockTransport(handler))

    health = OpenRouterProvider(
        settings_for(openrouter_model_fallbacks=["vendor/second:free"])
    ).health()

    assert health.reachable
    assert not health.throttled
    assert health.usable_models == ["vendor/second:free"]


def test_a_hard_failure_is_not_reported_as_throttling(mock_httpx):
    """A 400 is a fault. Only rate limiting earns the softer word."""
    mock_httpx(transport_returning(400, {"error": {"message": "bad request"}}))

    health = OpenRouterProvider(settings_for()).health()

    assert not health.reachable
    assert not health.throttled


def test_the_probe_maps_throttling_to_limited_rather_than_error(mock_httpx):
    """End of the chain: what the health endpoint actually reports."""
    import asyncio

    from app.core.credentials import probe_llm_provider

    mock_httpx(
        transport_returning(
            429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}}
        )
    )

    result = asyncio.run(probe_llm_provider(settings_for()))

    assert result.status == "limited"
    assert result.component == "llm_provider"
    assert FAKE_KEY not in result.detail


# -- Mistral, and a chain that crosses providers --------------------------
#
# The reason this section exists: OpenRouter meters free usage per account per day,
# so a chain of free OpenRouter models shares one allowance and expires together.
# Two models bought nothing. A second vendor is a second allowance, which is the
# only arrangement that survives a spent quota.

MISTRAL_KEY = "mstrl_" + "K" * 40


def mistral_settings(**overrides) -> Settings:
    base = {
        "llm_provider": "mistral",
        "mistral_api_key": MISTRAL_KEY,
        "openrouter_api_key": "",
    }
    base.update(overrides)
    return settings_for(**base)


def test_mistral_posts_to_its_own_endpoint_with_a_bearer_token(mock_httpx):
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion("ok"), capture=seen))

    MistralProvider(mistral_settings()).complete(ASK)

    assert len(seen) == 1
    assert str(seen[0].url) == "https://api.mistral.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {MISTRAL_KEY}"
    assert MISTRAL_KEY not in seen[0].content.decode()


def test_mistral_asks_for_the_json_mode_it_always_honours(mock_httpx):
    """Not json_schema. Mistral only supports the full schema on newer models and
    answers 422 elsewhere, which would cost the draft; json_object is accepted
    across the range and the caller validates the shape regardless."""
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion('{"a":1}'), capture=seen))

    MistralProvider(mistral_settings()).complete(
        ASK, schema={"type": "object", "properties": {"a": {"type": "integer"}}}
    )

    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "mistral-small-latest"

    # And because the request cannot carry the schema, the prompt must. Otherwise
    # the model is told to match a shape it was never shown, which is what made it
    # invent an envelope and fail validation every time.
    system = [m["content"] for m in body["messages"] if m["role"] == "system"]
    assert any('"properties"' in text for text in system)
    assert any('"a"' in text for text in system)


def test_a_provider_that_carries_the_schema_natively_does_not_repeat_it(mock_httpx):
    seen: list[httpx.Request] = []
    mock_httpx(transport_returning(200, completion('{"a":1}'), capture=seen))

    OpenRouterProvider(settings_for()).complete(
        ASK, schema={"type": "object", "properties": {"a": {"type": "integer"}}}
    )

    body = json.loads(seen[0].content)
    assert body["response_format"]["type"] == "json_schema"
    # One copy is enough; duplicating it in the prompt only spends tokens.
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_a_mistral_key_never_reaches_an_error_message(mock_httpx):
    mock_httpx(
        transport_returning(400, {"message": f"bad token {MISTRAL_KEY} supplied"})
    )
    with pytest.raises(LlmError) as raised:
        MistralProvider(mistral_settings()).complete(ASK)

    assert MISTRAL_KEY not in str(raised.value)
    assert REDACTED in str(raised.value)


def test_mistral_names_its_own_variable_when_the_key_is_missing():
    provider = MistralProvider(mistral_settings(mistral_api_key=""))
    assert not provider.configured
    with pytest.raises(LlmUnavailable, match="MISTRAL_API_KEY"):
        provider.complete(ASK)


def test_a_bare_message_body_is_read_as_the_reason(mock_httpx):
    """Mistral reports some failures as {"message": ...} rather than an error
    object, which the OpenRouter-shaped reader would have rendered as the whole
    payload."""
    mock_httpx(transport_returning(422, {"message": "model not found"}))
    with pytest.raises(LlmError, match="model not found"):
        MistralProvider(mistral_settings()).complete(ASK)


def test_content_returned_as_typed_parts_is_still_read(mock_httpx):
    """Some providers return content as a list of parts instead of a string."""
    payload = completion("")
    payload["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "the answer"},
        {"type": "text", "text": " is ok"},
    ]
    mock_httpx(transport_returning(200, payload))

    assert MistralProvider(mistral_settings()).complete(ASK).text == "the answer is ok"


# -- the chain ------------------------------------------------------------


def chained(**overrides) -> Settings:
    base = {
        "llm_provider": "mistral,openrouter",
        "mistral_api_key": MISTRAL_KEY,
        "openrouter_api_key": FAKE_KEY,
    }
    base.update(overrides)
    return settings_for(**base)


def test_a_comma_separated_provider_list_becomes_a_chain():
    settings = chained()
    assert settings.provider_chain == ["mistral", "openrouter"]
    assert settings.selected_provider == "mistral"

    provider = get_provider(settings)
    assert isinstance(provider, ChainProvider)
    assert [member.name for member in provider.members] == ["mistral", "openrouter"]
    assert provider.name == "mistral+openrouter"


def test_a_provider_without_a_key_is_dropped_from_the_chain_not_substituted():
    settings = chained(mistral_api_key="")
    assert settings.provider_chain == ["openrouter"]
    assert settings.requested_providers == ["mistral", "openrouter"]
    # One member is not wrapped: a chain of one only adds noise to every error.
    assert not isinstance(get_provider(settings), ChainProvider)
    assert get_provider(settings).name == "openrouter"


def test_a_chain_of_one_still_works_when_only_the_fallback_is_keyed(mock_httpx):
    mock_httpx(transport_returning(200, completion("ok")))
    result = get_provider(chained(mistral_api_key="")).complete(ASK)
    assert result.provider == "openrouter"


def test_the_second_provider_answers_when_the_first_is_out_of_allowance(mock_httpx):
    """The whole point. A spent Mistral allowance must not end the request."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen.append(host)
        if "mistral" in host:
            return httpx.Response(
                429, json={"message": "Rate limit exceeded: requests per day"}
            )
        return httpx.Response(200, json=completion("ok", model="vendor/x:free"))

    mock_httpx(httpx.MockTransport(handler))

    result = get_provider(chained()).complete(ASK)

    # Answered, and attributed to the member that actually did it.
    assert result.provider == "openrouter"
    assert any("mistral" in host for host in seen)
    assert any("openrouter" in host for host in seen)


def test_the_trace_records_the_provider_that_actually_answered(mock_httpx):
    mock_httpx(transport_returning(200, completion("ok")))
    result = get_provider(chained()).complete(ASK)
    # First member, so first name. A chain must never report its own name here.
    assert result.provider == "mistral"


def test_a_malformed_reply_is_not_retried_against_the_next_provider(mock_httpx):
    """Falling through is for exhaustion, not disagreement.

    A provider that answers with nonsense has a real fault behind it, most likely
    in the prompt. Asking a second vendor would hide that behind a second opinion.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text="<html>not json</html>")

    mock_httpx(httpx.MockTransport(handler))

    with pytest.raises(LlmError, match="not JSON"):
        get_provider(chained()).complete(ASK)
    assert calls["n"] == 1, "a malformed reply must not fall through"


def test_every_provider_exhausted_is_rate_limiting_and_says_what_still_works(
    mock_httpx,
):
    mock_httpx(
        transport_returning(429, {"message": "Rate limit exceeded: per-day"})
    )
    with pytest.raises(LlmRateLimited) as raised:
        get_provider(chained()).complete(ASK)

    assert "offline rule router" in str(raised.value)
    for secret in (MISTRAL_KEY, FAKE_KEY):
        assert secret not in str(raised.value)


def test_chain_health_is_reachable_when_any_member_generates(mock_httpx):
    def handler(request: httpx.Request) -> httpx.Response:
        if "mistral" in request.url.host:
            return httpx.Response(429, json={"message": "per-day limit"})
        return httpx.Response(200, json=completion("ok"))

    mock_httpx(httpx.MockTransport(handler))

    health = get_provider(chained()).health()

    assert health.reachable
    assert not health.throttled
    # Models are qualified by provider: two vendors can ship the same name.
    assert health.usable_models == ["openrouter/openrouter/free"]


def test_chain_health_is_throttled_only_when_every_member_is(mock_httpx):
    mock_httpx(transport_returning(429, {"message": "Rate limit: per-day quota"}))

    health = get_provider(chained()).health()

    assert not health.reachable
    assert health.throttled
    assert "out of allowance" in health.detail
    assert "every measurement is unaffected" in health.detail.lower()


def test_a_chain_with_no_keys_at_all_is_the_null_provider():
    settings = chained(mistral_api_key="", openrouter_api_key="")
    assert settings.provider_chain == []
    assert settings.selected_provider == "none"
    assert isinstance(get_provider(settings), NullProvider)


def test_the_cache_key_model_is_qualified_by_provider():
    """Two vendors can offer a model of the same name, and a contract cache keyed
    on the bare name would serve a plan drafted somewhere else."""
    from app.core.contract import _primary_model

    assert _primary_model(mistral_settings()) == "mistral/mistral-small-latest"
    assert _primary_model(settings_for()) == "openrouter/openrouter/free"
    assert _primary_model(chained()).startswith("mistral/")
