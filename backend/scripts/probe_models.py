"""Time a handful of free OpenRouter models on a contract-sized request.

``openrouter/free`` is an auto-router. It picks from a pool that includes large
reasoning models, and on a busy morning it queued every contract request past the
timeout, so the offline router planned all six demo scenes. A short health prompt
still answered in a second, which is what makes this worth measuring: the problem
is not reachability, it is which model the router lands on for a big prompt.

This times named candidates on a prompt the size of a real contract draft, so the
fallback chain in ``.env`` is chosen from evidence rather than from reputation.

    python scripts/probe_models.py

Prints no credential. Models that fail are reported with their reason.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.llm.base import LlmError, Message  # noqa: E402
from app.core.llm.openrouter import OpenRouterProvider  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES = [
    "openrouter/free",
    "liquid/lfm-2.5-2.6b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-14b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

# Roughly the shape and size of a contract draft: a long system prompt, a
# question, and a schema to fill. Timing a two-word prompt would prove nothing,
# which is how the fast health probe hid this in the first place.
SYSTEM = (
    "You plan remote-sensing analyses. Given a question about satellite imagery "
    "and a list of available tools, return JSON naming the claim under test, the "
    "tools to run in order with their parameters, the measurements expected, and "
    "the alternative explanations that should be checked. Available tools: "
    "spectral-index-engine (indices: NDVI, NDWI, MNDWI, NDBI, NDMI; threshold "
    "methods: otsu, fixed), grounding-engine (target classes: water, vegetation, "
    "built-up, bare), sar-backscatter-engine, optical-sar-fusion, "
    "change-cva-engine, rs-landcover-probe, evidence-disagreement-engine, "
    "confounder-engine (kinds: seasonality, misregistration, radiometry, cloud, "
    "illumination), gis-measure-engine, verdict-engine. Return only JSON."
) * 2

QUESTION = (
    "Has the water spread of the Tungabhadra reservoir increased between May 2020 "
    "and May 2024? The inputs are two co-registered optical scenes."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "task_type": {"type": "string"},
        "target": {"type": "string"},
        "tools": {"type": "array", "items": {"type": "object"}},
        "confounders": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claim", "task_type", "tools"],
}


def main() -> int:
    settings = get_settings()
    if not settings.has_openrouter:
        print("OPENROUTER_API_KEY is not set, so there is nothing to time.")
        return 1

    provider = OpenRouterProvider(settings)
    budget = settings.llm_timeout_seconds
    print(f"\nbudget per attempt: {budget:.0f}s\n")

    fast: list[tuple[float, str]] = []

    for model in CANDIDATES:
        # One model at a time, bypassing the chain, so each is timed on its own.
        one = OpenRouterProvider(settings.model_copy(update={"openrouter_model": model}))
        started = time.perf_counter()
        try:
            result = one.complete(
                [
                    Message(role="system", content=SYSTEM),
                    Message(role="user", content=QUESTION),
                ],
                schema=SCHEMA,
                max_tokens=700,
            )
        except LlmError as exc:
            elapsed = time.perf_counter() - started
            print(f"  {'FAIL':<6} {model:<52} {elapsed:6.1f}s  {exc}")
            continue

        elapsed = time.perf_counter() - started
        # Did it actually produce something parseable, or just produce something?
        try:
            json.loads(
                result.text[result.text.find("{") : result.text.rfind("}") + 1] or "{}"
            )
            shape = "json"
        except ValueError:
            shape = "prose"

        fast.append((elapsed, model))
        print(
            f"  {'OK':<6} {model:<52} {elapsed:6.1f}s  "
            f"answered by {result.model} ({shape}, "
            f"{result.completion_tokens or '?'} tok)"
        )

    if not fast:
        print("\nNothing answered. The offline rule router plans every contract.")
        return 1

    fast.sort()
    print("\nfastest first:")
    for elapsed, model in fast:
        print(f"  {elapsed:6.1f}s  {model}")
    print(
        "\nPut the quick ones in OPENROUTER_MODEL_FALLBACKS, comma separated, so a "
        "queued auto-router is not the only thing standing between a question and "
        "a plan.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
