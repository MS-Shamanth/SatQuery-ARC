"""Why a contract fell back to the offline router, for each sample question.

``check_run.py`` reports which drafter produced a plan but not why the model was
not used. That distinction matters: a rate limit is the fallback working, and a
draft that fails validation every time on one shape of question is a bug in the
prompt. Both look identical from the outside.

Runs against a live server, refreshing so the cache cannot hide the answer.

    python scripts/probe_contract_source.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000/api"

CASES = [
    ("water_body", "Highlight the water body referred to in the query"),
    ("reservoir_change", "Has the water spread increased since 2019?"),
    ("urban_growth", "Has built-up area expanded between the two dates?"),
    ("flood_urban", "Which parts of the city are flooded?"),
    ("seasonal_farmland", "Has vegetation declined between the two dates?"),
]


def post(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def main() -> int:
    failures = 0
    for sample, question in CASES:
        # Loading a sample creates the session it loads into.
        session = post(f"/samples/{sample}/load")["session_id"]
        try:
            contract = post(
                f"/sessions/{session}/contract",
                {"query": question, "refresh": True},
            )
        except urllib.error.HTTPError as exc:  # pragma: no cover - live probe
            print(f"  {sample:<26} HTTP {exc.code} {exc.read()[:160]!r}")
            failures += 1
            continue

        source = contract.get("source")
        model = contract.get("model") or "-"
        reason = contract.get("fallback_reason")
        mark = "model " if source == "language-model" else "OFFLINE"
        print(f"  {mark} {sample:<26} {source:<20} {model}")
        if reason:
            print(f"          reason: {reason}")
            failures += 1

    print()
    print("A reason above is not necessarily a fault: a 429 is the fallback doing")
    print("its job. A validation message repeating on one shape of question is.")
    return 0 if failures == 0 else 0  # informational only


if __name__ == "__main__":
    sys.exit(main())
