"""Probe the configured language model provider without revealing its key.

Issues a real generation rather than listing models. Listing is not evidence: a
model can appear in a catalogue and still refuse to generate, which is how the
previous provider cost a day of debugging.

    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py

Prints which provider is selected, which models actually answered, and whether a
structured request comes back as JSON. Never prints the key, and asserts as much
before exiting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.llm import Message, get_provider  # noqa: E402
from app.core.llm.base import LlmError, redact  # noqa: E402
from app.core.llm.registry import available_providers  # noqa: E402


def main() -> int:
    settings = get_settings()
    provider = get_provider()
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        print(text)

    say(f"LLM_PROVIDER      {settings.llm_provider}")
    say(f"selected          {settings.selected_provider}")
    say(f"timeout           {settings.llm_timeout_seconds:.0f}s")
    for name, ready in available_providers().items():
        say(f"  {name:<14} {'key present' if ready else 'no key'}")

    if settings.selected_provider == "openrouter":
        say(f"endpoint          {settings.openrouter_chat_url}")
        say(f"model chain       {', '.join(settings.openrouter_model_chain)}")

    say("")
    health = provider.health()
    say(f"configured        {health.configured}")
    say(f"reachable         {health.reachable}")
    if health.usable_models:
        say(f"generated         {', '.join(health.usable_models)}")
    if health.rejected:
        say("credentials       REJECTED by the provider")
    say(f"detail            {health.detail}")

    if health.reachable:
        say("")
        try:
            plain = provider.complete(
                [
                    Message(
                        role="system",
                        content="Answer in exactly one short sentence.",
                    ),
                    Message(
                        role="user",
                        content="What does a normalised difference index measure?",
                    ),
                ],
                max_tokens=60,
            )
            say(f"answered by       {plain.model}")
            say(f"sample            {plain.text[:140]}")
            if plain.prompt_tokens is not None:
                say(
                    f"tokens            {plain.prompt_tokens} in, "
                    f"{plain.completion_tokens} out"
                )
        except LlmError as exc:
            say(f"plain call failed {exc}")

        try:
            structured = provider.complete(
                [
                    Message(
                        role="system",
                        content=(
                            "Return only JSON matching the schema. No prose, no "
                            "code fences."
                        ),
                    ),
                    Message(
                        role="user",
                        content=(
                            'Return {"task":"region_grounding","target":"water"} '
                            "for the question: highlight the reservoir."
                        ),
                    ),
                ],
                schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["task", "target"],
                },
                max_tokens=120,
            )
            from app.core.llm.openrouter import extract_json

            parsed = extract_json(structured.text)
            say(f"structured ok     {parsed}")
        except LlmError as exc:
            say(f"structured failed {exc}")
            say("                  the offline rule router covers this.")

    # The point of the script is that it is safe to paste its output anywhere.
    key = settings.openrouter_api_key.strip()
    printed = "\n".join(lines)
    if key and key in printed:
        print("\nFAILED: the key appeared in this output.")
        return 2
    if key and redact(printed) != printed:
        print("\nFAILED: something key-shaped appeared in this output.")
        return 2
    print("\nno credential appears in the output above.")
    return 0 if health.reachable or not health.configured else 1


if __name__ == "__main__":
    raise SystemExit(main())
