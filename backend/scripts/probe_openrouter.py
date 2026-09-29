"""Inspect exactly what OpenRouter returns, with the key redacted.

The health probe reported an empty completion, which is not enough to act on: an
empty body, a refusal, a content field under a different name and a model slug that
does not route all look the same from the outside. This prints the actual response
shape so the cause is visible rather than inferred.

    .\\.venv\\Scripts\\python.exe scripts\\probe_openrouter.py [model]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.llm.base import redact  # noqa: E402

FREE_HINTS = ("free", ":free")


def main() -> int:
    settings = get_settings()
    if not settings.has_openrouter:
        print("OPENROUTER_API_KEY is not set")
        return 1

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_site_url,
        "X-Title": settings.openrouter_app_name,
    }
    wanted = sys.argv[1] if len(sys.argv) > 1 else settings.openrouter_model

    with httpx.Client(timeout=40.0) as client:
        # What models does this key actually have?
        listing = client.get(
            f"{settings.openrouter_base_url.rstrip('/')}/models", headers=headers
        )
        print(f"GET /models -> {listing.status_code}")
        slugs: list[str] = []
        if listing.status_code == 200:
            data = listing.json().get("data") or []
            slugs = [str(item.get("id")) for item in data]
            print(f"  {len(slugs)} model(s) advertised")
            exact = [s for s in slugs if s == wanted]
            print(f"  '{wanted}' advertised: {bool(exact)}")
            free = [s for s in slugs if any(h in s.lower() for h in FREE_HINTS)]
            print(f"  free-tier slugs: {len(free)}")
            for slug in free[:12]:
                print(f"    {slug}")

        print(f"\nPOST /chat/completions with model={wanted!r}")
        response = client.post
        result = response(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": wanted,
                "messages": [{"role": "user", "content": "Reply with the word ok."}],
                "temperature": 0.0,
                "max_tokens": 32,
            },
        )
        print(f"  status {result.status_code}")
        try:
            payload = result.json()
        except ValueError:
            print("  body is not JSON:", redact(result.text[:400]))
            return 1

        print("  keys:", ", ".join(sorted(payload)))
        if payload.get("error"):
            print("  error:", redact(json.dumps(payload["error"])[:400]))
        for index, choice in enumerate(payload.get("choices") or []):
            message = choice.get("message") or {}
            print(f"  choice[{index}] finish={choice.get('finish_reason')!r}")
            print(f"    message keys: {', '.join(sorted(message))}")
            for field in ("content", "reasoning", "refusal"):
                value = message.get(field)
                if value:
                    print(f"    {field}: {redact(str(value)[:200])}")
        if payload.get("usage"):
            print("  usage:", payload["usage"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
