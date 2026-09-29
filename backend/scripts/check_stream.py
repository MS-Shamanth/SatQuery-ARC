"""Does every streamed event carry the keys the client reads?

The panels read ``trace.disagreement``, ``trace.packet``, ``trace.verdict`` and
``trace.remedies``. They are nullable, and the client guards on ``=== null``.

The stream used to be serialised with ``exclude_none=True``, which is recursive:
while a run was still in progress those keys were absent rather than null, so the
guards passed an ``undefined`` straight through and the next line threw. Two panels
showed a red error over a run that was completely fine.

``check_wire.py`` did not catch it because it reads the plain HTTP trace. This
reads the stream, event by event, which is what the browser actually consumes.

    python scripts/check_stream.py [sample]
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8000/api"

# Nullable trace fields a panel destructures or dots into. Absent is not the same
# as null to any of them.
NULLABLE_TRACE_KEYS = (
    "verdict",
    "ledger",
    "remedies",
    "disagreement",
    "packet",
    "finished_at",
    "error",
)


def main(sample: str = "water_body") -> int:
    ok = 0
    bad = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok, bad
        if condition:
            ok += 1
            print(f"  OK   {label} {detail}")
        else:
            bad += 1
            print(f"  MISS {label} {detail}")

    with httpx.Client(timeout=180.0) as client:
        manifest = client.get(f"{BASE}/samples").raise_for_status().json()
        scene = manifest["scenes"][sample]
        question = scene["suggested_queries"][0]
        print(f"\n{scene['title']}\n  query: {question}\n")

        session = client.post(f"{BASE}/samples/{sample}/load").raise_for_status().json()
        session_id = session["session_id"]

        contract = (
            client.post(
                f"{BASE}/sessions/{session_id}/contract",
                json={"query": question, "refresh": True},
            )
            .raise_for_status()
            .json()
        )
        pending = (
            client.post(
                f"{BASE}/sessions/{session_id}/runs",
                json={"contract_hash": contract["contract_hash"]},
            )
            .raise_for_status()
            .json()
        )
        run_id = pending["run_id"]

        # The trace handed back by the start call is read by the same panels, so it
        # is held to the same contract as the stream.
        for key in NULLABLE_TRACE_KEYS:
            check(f"the pending trace declares {key}", key in pending)

        seen = 0
        traces = 0
        missing: dict[str, int] = {}
        with client.stream(
            "GET", f"{BASE}/sessions/{session_id}/runs/{run_id}/stream"
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                seen += 1

                # Every event declares its own optional slots too, for the same
                # reason: the reducer tests `event.stage`, `event.step` and so on.
                for key in ("stage", "step", "tool_run", "trace"):
                    if key not in event:
                        missing[f"event.{key}"] = missing.get(f"event.{key}", 0) + 1

                trace = event.get("trace")
                if trace is None:
                    continue
                traces += 1
                for key in NULLABLE_TRACE_KEYS:
                    if key not in trace:
                        missing[f"trace.{key}"] = missing.get(f"trace.{key}", 0) + 1

                if event.get("type") == "run.finished":
                    break

        print()
        check("the stream delivered events", seen > 0, f"{seen} event(s)")
        check("at least one event carried a trace", traces > 0, f"{traces}")
        check(
            "no event dropped a key the client reads",
            not missing,
            json.dumps(missing) if missing else "",
        )

    print(f"\n{ok} ok, {bad} miss\n")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
