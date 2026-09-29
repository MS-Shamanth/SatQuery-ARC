"""Is the server still answering while a contract is being drafted?

Drafting waits on a remote model. If that wait happens on the event loop, every
other request queues behind it: the health pill stops responding, the sample
gallery will not load, and the whole interface looks dead while it is only
thinking. That is exactly how it behaved before the drafting work was moved into
a thread, and a 25 second freeze on the first click is not something a demo
survives.

So this measures the one thing that matters: the worst latency of a trivial
request taken repeatedly while a real draft is in flight.

    python scripts/check_responsive.py [sample]
"""

from __future__ import annotations

import statistics
import sys
import threading
import time

import httpx

BASE = "http://127.0.0.1:8000/api"

# A plain GET that touches no imagery. Anything slower than this is queueing, not
# work.
PROBE_PATH = "/health"

# Generous. A blocked loop showed tens of seconds; real serving is milliseconds.
ACCEPTABLE_WORST_MS = 2500.0


def main(sample: str = "reservoir_change") -> int:
    with httpx.Client(timeout=120.0) as client:
        manifest = client.get(f"{BASE}/samples").raise_for_status().json()
        scene = manifest["scenes"][sample]
        question = scene["suggested_queries"][0]

        session = client.post(f"{BASE}/samples/{sample}/load").raise_for_status().json()
        session_id = session["session_id"]
        print(f"\n{sample}: {question}")

        drafted: dict[str, object] = {}

        def draft() -> None:
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{BASE}/sessions/{session_id}/contract",
                    json={"query": question, "refresh": True},
                )
                drafted["status"] = response.status_code
                drafted["body"] = response.json() if response.is_success else None
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                drafted["error"] = f"{type(exc).__name__}: {exc}"
            drafted["ms"] = round((time.perf_counter() - started) * 1000)

        worker = threading.Thread(target=draft, name="draft", daemon=True)
        worker.start()

        # Poll a trivial endpoint on a separate client, so connection reuse cannot
        # make the drafting request's own socket look like the server's health.
        latencies: list[float] = []
        with httpx.Client(timeout=30.0) as probe:
            while worker.is_alive():
                started = time.perf_counter()
                try:
                    probe.get(f"{BASE}{PROBE_PATH}")
                    latencies.append((time.perf_counter() - started) * 1000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  MISS the probe failed outright: {exc}")
                    latencies.append(float("inf"))
                time.sleep(0.25)

        worker.join()

    if not latencies:
        print("  MISS drafting finished before a single probe landed")
        return 1

    worst = max(latencies)
    print(f"\n  draft took {drafted.get('ms')} ms")
    if "error" in drafted:
        print(f"  draft error: {drafted['error']}")
    else:
        body = drafted.get("body") or {}
        print(
            f"  draft status {drafted.get('status')} "
            f"source={body.get('source')} model={body.get('model') or '-'}"
        )
        if body.get("fallback_reason"):
            print(f"  fallback: {body['fallback_reason']}")

    print(
        f"  {PROBE_PATH} probed {len(latencies)}x while drafting: "
        f"median {statistics.median(latencies):.0f} ms, worst {worst:.0f} ms"
    )

    bad = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal bad
        if condition:
            print(f"  OK   {label} {detail}")
        else:
            bad += 1
            print(f"  MISS {label} {detail}")

    check(
        "the draft completed",
        "error" not in drafted and drafted.get("status") == 200,
        str(drafted.get("error") or drafted.get("status")),
    )
    check(
        "the server kept answering while the model was thinking",
        worst < ACCEPTABLE_WORST_MS,
        f"worst {worst:.0f} ms, limit {ACCEPTABLE_WORST_MS:.0f} ms",
    )
    check(
        "the draft stayed inside its own timeout budget",
        isinstance(drafted.get("ms"), int) and drafted["ms"] < 70_000,
        f"{drafted.get('ms')} ms",
    )

    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {bad} problem(s)\n")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
