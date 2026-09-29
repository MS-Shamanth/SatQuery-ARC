"""End-to-end check of the run pipeline against a live server.

Loads a sample scene, drafts a contract, executes it, and reads the SSE stream,
then asserts the trace is coherent. Run with the backend up:

    .venv\\Scripts\\python.exe scripts\\check_run.py water_body
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile

import httpx

BASE = "http://127.0.0.1:8000/api"

# This script prints text a language model wrote, and the Windows console defaults
# to cp1252, which cannot encode most of it. A free model emitting one en dash
# crashed the whole verification after the run had already passed, which reads as a
# product failure and is not one. Unencodable characters are replaced rather than
# raised: the point of the output is to be read, not to be a faithful byte copy.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main(key: str, query: str | None = None) -> int:
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
        scene = manifest["scenes"][key]
        question = query or scene["suggested_queries"][0]
        print(f"\n{scene['title']} ({scene['place']})\n  query: {question}\n")

        session = client.post(f"{BASE}/samples/{key}/load").raise_for_status().json()
        session_id = session["session_id"]

        # Always redraft: a cached contract predates any tool registered since,
        # and this check is meant to exercise the live planning path.
        contract = (
            client.post(
                f"{BASE}/sessions/{session_id}/contract",
                json={"query": question, "refresh": True},
            )
            .raise_for_status()
            .json()
        )
        print(f"  contract {contract['contract_hash'][:12]} via {contract['source']}")
        print(f"  claim: {contract['claim']}")
        check("contract plans tools", bool(contract["tools"]),
              f"({len(contract['tools'])})")

        started = client.post(
            f"{BASE}/sessions/{session_id}/runs",
            json={"contract_hash": contract["contract_hash"]},
        )
        check("run accepted", started.status_code == 202, f"HTTP {started.status_code}")
        if started.status_code != 202:
            print(started.text)
            return 1
        run_id = started.json()["run_id"]
        check("pending trace declares every stage",
              len(started.json()["stages"]) == 7,
              f"({len(started.json()['stages'])} stages)")

        # Read the stream to completion.
        seen: list[str] = []
        with client.stream(
            "GET", f"{BASE}/sessions/{session_id}/runs/{run_id}/stream"
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                seen.append(event["type"])
                if event["type"] == "log":
                    print(f"    | {event['message']}")
                if event["type"] == "run.finished":
                    break

        check("stream opened with run.started", seen and seen[0] == "run.started")
        check("stream closed with run.finished", bool(seen) and seen[-1] == "run.finished")
        check("stages streamed", seen.count("stage.started") >= 7,
              f"({seen.count('stage.started')})")
        check("steps streamed", seen.count("step.finished") > 0,
              f"({seen.count('step.finished')})")

        trace = (
            client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}")
            .raise_for_status()
            .json()
        )
        print(f"\n  status {trace['status']} in {trace['duration_ms']:.0f} ms")
        for stage in trace["stages"]:
            print(
                f"    {stage['status']:<10} {stage['label']:<20} "
                f"{stage['duration_ms']:>8.1f} ms  {len(stage['steps'])} step(s)"
            )

        check("run completed", trace["status"] == "completed", trace["status"])
        check("measurements recorded", len(trace["measurements"]) > 0,
              f"({len(trace['measurements'])})")
        check("masks recorded", len(trace["masks"]) > 0, f"({len(trace['masks'])})")
        check(
            "every measurement carries a formula",
            all(m["formula"] for m in trace["measurements"]),
        )
        check(
            "every measurement names its tool",
            all(m["source_tool"] and m["source_version"] for m in trace["measurements"]),
        )

        specialists = next(
            s for s in trace["stages"] if s["id"] == "run_specialists"
        )
        check("specialists ran", specialists["status"] in {"ok", "partial"},
              specialists["status"])
        confounders = next(
            s for s in trace["stages"] if s["id"] == "test_confounders"
        )
        check("confounders enumerated", len(confounders["steps"]) > 0,
              f"({len(confounders['steps'])})")
        measured = [s for s in confounders["steps"] if s["status"] == "ok"]
        print(
            f"    confounders: {len(measured)} with measured evidence, "
            f"{len(confounders['steps']) - len(measured)} named but untested"
        )
        # -- the verdict --------------------------------------------------
        stage = next(s for s in trace["stages"] if s["id"] == "resolve_verdict")
        verdict = trace.get("verdict")
        if verdict is None:
            check("no verdict is claimed when none was reached",
                  stage["status"] == "not_built", stage["status"])
        else:
            print(
                f"\n  VERDICT {verdict['label'].upper()} at "
                f"{verdict['confidence']:.0%} confidence"
            )
            print(f"    asserted: {verdict['asserted_direction']}  "
                  f"measured: {verdict['measured_direction']}")
            print(f"    {verdict['reasoning']}")
            for component in verdict["confidence_components"]:
                print(
                    f"      {component['name']:<26} "
                    f"{component['contribution']:+.3f} of {component['weight']:.2f}"
                    f"  (measured {component['measured']:.3f})"
                )
            if verdict.get("narrative"):
                print(f"\n    narrated by {verdict['narrative_source']}:")
                print(f"      {verdict['narrative']}")
            for lever in verdict["what_would_change_it"]:
                print(f"      would change it: {lever}")

            check("the verdict stage ran", stage["status"] in {"ok", "partial"},
                  stage["status"])
            check("confidence is a sum of named components",
                  len(verdict["confidence_components"]) >= 4,
                  f"({len(verdict['confidence_components'])})")
            check(
                "confidence equals the sum of its components",
                abs(
                    sum(c["contribution"] for c in verdict["confidence_components"])
                    - verdict["confidence"]
                )
                < 1e-6
                or verdict["confidence"] in (0.0, 1.0),
            )
            check("the evidence ledger reached the trace",
                  trace.get("ledger") is not None)
            if trace.get("ledger"):
                ledger = trace["ledger"]
                check("every admitted item names its source tool and formula",
                      all(i["source_tool"] and i["formula"] for i in ledger["items"]),
                      f"({len(ledger['items'])} items)")
                check("excluded measurements carry a reason",
                      all(bool(r) for r in ledger["excluded"].values()),
                      f"({len(ledger['excluded'])} excluded)")
                keys = {m["key"] for m in trace["measurements"]}
                check("every admitted item traces to a real measurement",
                      all(i["measurement_key"] in keys for i in ledger["items"]))

            audit = verdict.get("narrative_audit")
            if audit is not None:
                print(
                    f"    numeric audit: {audit['traced']}/{audit['checked']} traced, "
                    f"passed={audit['passed']}"
                )
                check(
                    "a narrative that survives the audit has no untraceable figure",
                    (not audit["passed"]) or not audit["untraceable"],
                    str(audit["untraceable"][:4]),
                )
                check(
                    "a narrative that fails the audit is not displayed",
                    audit["passed"] or verdict.get("narrative") is None,
                )

        # Every measurement key referenced by a step must exist in the trace.
        keys = {m["key"] for m in trace["measurements"]}
        dangling = [
            key
            for stage in trace["stages"]
            for step in stage["steps"]
            for key in step["measurement_keys"]
            if key not in keys and step["status"] == "ok"
        ]
        check("no step references a measurement that is absent", not dangling,
              f"{dangling[:3]}")

        # Re-running the same contract must be refused while one is in flight or
        # accepted cleanly afterwards.
        again = client.post(
            f"{BASE}/sessions/{session_id}/runs",
            json={"contract_hash": contract["contract_hash"]},
        )
        check("a finished run does not block a new one", again.status_code == 202,
              f"HTTP {again.status_code}")

        bogus = client.post(
            f"{BASE}/sessions/{session_id}/runs",
            json={"contract_hash": "0" * 32},
        )
        check("an unknown contract hash is refused", bogus.status_code == 404,
              f"HTTP {bogus.status_code}")

        # -- map layers ---------------------------------------------------
        layers = (
            client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}/layers")
            .raise_for_status()
            .json()
        )
        print()
        for layer in layers:
            west, south, east, north = layer["bounds_wgs84"]
            print(
                f"    {layer['kind']:<5} {layer['key']:<28} "
                f"{south:.4f}..{north:.4f}N {west:.4f}..{east:.4f}E"
                + (f"  {layer['area_km2']:.3f} km2" if layer.get("area_km2") else "")
            )
        check("layers were rendered", len(layers) > 0, f"({len(layers)})")
        check("a base imagery layer exists",
              any(layer["kind"] == "base" for layer in layers))
        check("at least one mask layer exists",
              any(layer["kind"] == "mask" for layer in layers))
        check(
            "every layer carries plausible WGS84 bounds",
            all(
                -180 <= layer["bounds_wgs84"][0] < layer["bounds_wgs84"][2] <= 180
                and -90 <= layer["bounds_wgs84"][1] < layer["bounds_wgs84"][3] <= 90
                for layer in layers
            ),
        )

        for layer in layers:
            png = client.get(f"http://127.0.0.1:8000{layer['png_url']}")
            check(
                f"PNG served for {layer['key']}",
                png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n",
                f"{len(png.content):,} bytes",
            )
            if layer.get("geojson_url"):
                geo = client.get(f"http://127.0.0.1:8000{layer['geojson_url']}")
                body = geo.json()
                check(
                    f"GeoJSON served for {layer['key']}",
                    geo.status_code == 200
                    and body["type"] == "FeatureCollection"
                    and len(body["features"]) > 0,
                    f"{len(body.get('features', []))} feature(s)",
                )
                coords_ok = all(
                    -180 <= point[0] <= 180 and -90 <= point[1] <= 90
                    for feature in body["features"]
                    for ring in feature["geometry"]["coordinates"]
                    for point in ring
                )
                check(f"GeoJSON for {layer['key']} is in degrees", coords_ok)

        traversal = client.get(
            f"{BASE}/sessions/{session_id}/runs/{run_id}"
            "/layers/..%2F..%2Fsession.json"
        )
        check("a traversal attempt on a layer path is refused",
              traversal.status_code == 404, f"HTTP {traversal.status_code}")

        # -- the evidence packet ------------------------------------------
        response = client.get(f"{BASE}/sessions/{session_id}/runs/{run_id}/packet")
        check("the packet manifest is served", response.status_code == 200,
              f"HTTP {response.status_code}")
        if response.status_code == 200:
            pack = response.json()
            print()
            print(
                f"    packet: {len(pack['files'])} file(s), "
                f"{sum(f['size_bytes'] for f in pack['files']) / 1024:.0f} kB"
            )
            audit = pack["audit"]
            print(
                f"    figures: {audit['traced']}/{audit['checked']} attributed, "
                f"passed={audit['passed']}"
            )
            stage = next(s for s in trace["stages"] if s["id"] == "compose_packet")
            check("the composing stage ran", stage["status"] in {"ok", "partial"},
                  stage["status"])
            check("the trace names the packet it wrote",
                  trace.get("packet") is not None)
            check("every figure in the report is attributed",
                  audit["passed"], str(audit["untraceable"][:3]))
            check("every packet file carries a checksum and a size",
                  all(f["sha256"] and f["size_bytes"] > 0 for f in pack["files"]))
            check("the packet says what it leaves out",
                  bool(pack["omissions"]), f"({len(pack['omissions'])})")
            check("a report, a trace and the measurements are all present",
                  {"report", "trace", "measurements"}
                  <= {f["role"] for f in pack["files"]})

            keys = {m["key"] for m in trace["measurements"]}
            attributed = [f for f in pack["figures"] if f["measurement_key"]]
            check(
                "every measured figure resolves to a measurement in this run",
                all(f["measurement_key"] in keys for f in attributed),
                f"({len(attributed)} measured figures)",
            )

            base_url = f"{BASE}/sessions/{session_id}/runs/{run_id}/packet"
            report = next(
                (f for f in pack["files"] if f["role"] == "report"), None
            )
            if report is not None:
                pdf = client.get(f"{base_url}/{report['filename']}")
                check(
                    "the report downloads as a PDF",
                    pdf.status_code == 200 and pdf.content[:4] == b"%PDF",
                    f"{len(pdf.content):,} bytes",
                )
                check(
                    "the served report matches the checksum in the manifest",
                    hashlib.sha256(pdf.content).hexdigest() == report["sha256"],
                )
            if pack.get("archive"):
                zipped = client.get(f"{base_url}/{pack['archive']['filename']}")
                check(
                    "the archive downloads as a zip",
                    zipped.status_code == 200 and zipped.content[:2] == b"PK",
                    f"{len(zipped.content):,} bytes",
                )
                with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
                    held = set(archive.namelist())
                check("the archive holds every file the manifest lists",
                      all(f["filename"] in held for f in pack["files"]),
                      f"({len(held)} entries)")
                check("the archive explains itself in plain text",
                      "README.txt" in held)

            refused = client.get(f"{base_url}/session.json")
            check("a file the packet does not contain is refused",
                  refused.status_code == 404, f"HTTP {refused.status_code}")
            traversal = client.get(f"{base_url}/..%2F..%2Fsession.json")
            check("a traversal attempt on a packet path is refused",
                  traversal.status_code == 404, f"HTTP {traversal.status_code}")

        # The grounded area must agree with the mask it drew.
        grounded = [
            layer for layer in layers if layer["key"].startswith("grounded")
        ]
        if grounded:
            area = next(
                (
                    m for m in trace["measurements"]
                    if m["key"] == "grounded_area_km2"
                ),
                None,
            )
            check("the grounding engine reported an area", area is not None)
            if area is not None:
                check(
                    "the drawn layer and the reported area agree",
                    abs(grounded[0]["area_km2"] - area["value"]) < 1e-6,
                    f"{grounded[0]['area_km2']:.4f} vs {area['value']:.4f} km2",
                )

    print(f"\n{ok} ok, {bad} miss\n")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
