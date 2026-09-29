"""Compare what the API actually sends against what the client expects.

A missing field is not a small problem in this app. The client reads
``contract.expected_outputs.length`` and the like directly, so one absent key
throws during render, React unmounts the tree, and the user gets a blank page with
nothing in it to explain why. That is how the approve-and-run flow broke.

This walks the real endpoints the Studio calls, in the order it calls them, and
diffs each payload's keys against the TypeScript interfaces in
frontend/src/lib/types.ts. It reports fields the client expects that the server
does not send, which is the direction that causes a crash, and also fields the
server sends that the client has no name for, which is usually a stale type.

    .\\.venv\\Scripts\\python.exe scripts\\check_wire.py water_body
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000/api"
TYPES = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "lib"
    / "types.ts"
)

# Which interface describes each payload, and how to reach the payload.
CHECKS: tuple[tuple[str, str], ...] = (
    ("SessionRecord", "session"),
    ("ReadinessReport", "readiness"),
    ("AnalysisContract", "contract"),
    ("RunTrace", "trace"),
    ("MapLayer", "layer"),
    ("EvidencePacket", "packet"),
    ("Verdict", "verdict"),
    ("ConfounderTest", "confounder"),
    ("Measurement", "measurement"),
    ("ToolRun", "tool_run"),
    ("TraceStage", "stage"),
    ("TraceStep", "step"),
)


def interface_fields(source: str, name: str) -> set[str] | None:
    """The property names one TypeScript interface declares.

    Deliberately simple parsing: these interfaces are flat records of
    ``name: type;`` lines, and a real parser here would be more machinery than
    the problem deserves.
    """
    match = re.search(
        rf"export interface {re.escape(name)}\s*\{{(.*?)^\}}",
        source,
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        return None

    fields: set[str] = set()
    depth = 0
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue
        # Skip the inside of nested object literals; only top-level keys matter.
        if depth == 0:
            found = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\??\s*:", stripped)
            if found:
                fields.add(found.group(1))
        depth += stripped.count("{") - stripped.count("}")
    return fields


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "water_body"
    source = TYPES.read_text(encoding="utf-8")

    payloads: dict[str, dict] = {}
    with httpx.Client(timeout=240.0) as client:
        manifest = client.get(f"{BASE}/samples").raise_for_status().json()
        scene = manifest["scenes"][key]
        question = scene["suggested_queries"][0]

        session = client.post(f"{BASE}/samples/{key}/load").raise_for_status().json()
        payloads["session"] = session
        session_id = session["session_id"]

        payloads["readiness"] = (
            client.get(f"{BASE}/sessions/{session_id}/readiness")
            .raise_for_status()
            .json()
        )
        # Deliberately offline: this check is about the shape of the payload, and
        # depending on a language model would make it fail for reasons that have
        # nothing to do with the wire format.
        contract = (
            client.post(
                f"{BASE}/sessions/{session_id}/contract",
                json={"query": question, "refresh": True, "force_offline": True},
            )
            .raise_for_status()
            .json()
        )
        payloads["contract"] = contract

        started = client.post(
            f"{BASE}/sessions/{session_id}/runs",
            json={"contract_hash": contract["contract_hash"]},
        ).raise_for_status()
        run_id = started.json()["run_id"]

        # Drain the stream so the run is finished before the trace is read.
        with client.stream(
            "GET", f"{BASE}/sessions/{session_id}/runs/{run_id}/stream"
        ) as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    if event.get("type") == "run.finished":
                        break

        trace = (
            client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}")
            .raise_for_status()
            .json()
        )
        payloads["trace"] = trace

        layers = (
            client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}/layers")
            .raise_for_status()
            .json()
        )
        if layers:
            payloads["layer"] = layers[0]

        packet = client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}/packet")
        if packet.status_code == 200:
            payloads["packet"] = packet.json()

        if trace.get("verdict"):
            payloads["verdict"] = trace["verdict"]
        if trace.get("confounders"):
            payloads["confounder"] = trace["confounders"][0]
        if trace.get("measurements"):
            payloads["measurement"] = trace["measurements"][0]
        if trace.get("tool_runs"):
            payloads["tool_run"] = trace["tool_runs"][0]
        if trace.get("stages"):
            payloads["stage"] = trace["stages"][0]
            for stage in trace["stages"]:
                if stage["steps"]:
                    payloads["step"] = stage["steps"][0]
                    break

    print(f"\nscene {key}: {question}\n")
    missing_total = 0

    for interface, payload_key in CHECKS:
        expected = interface_fields(source, interface)
        payload = payloads.get(payload_key)
        if expected is None:
            print(f"  {interface:<18} SKIP  no such interface in types.ts")
            continue
        if payload is None:
            print(f"  {interface:<18} SKIP  this run produced no {payload_key}")
            continue

        sent = set(payload)
        missing = sorted(expected - sent)
        extra = sorted(sent - expected)

        if missing:
            missing_total += len(missing)
            print(f"  {interface:<18} BREAKS THE PAGE: client reads but server "
                  f"omits {', '.join(missing)}")
        else:
            print(f"  {interface:<18} ok    {len(sent)} field(s)")
        if extra:
            print(f"  {'':<18}       server also sends {', '.join(extra)}")

    print()
    if missing_total:
        print(f"{missing_total} field(s) the client would crash on.")
        return 1
    print("every field the client reads is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
