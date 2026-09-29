"""Print the most recent run trace: measurements and notes, by tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(tool: str | None = None) -> int:
    traces = sorted(
        Path("data/sessions").glob("*/runs/*.json"), key=lambda p: p.stat().st_mtime
    )
    if not traces:
        print("no traces on disk")
        return 1

    trace = json.loads(traces[-1].read_text(encoding="utf-8"))
    print(f"query : {trace['query']}")
    print(f"claim : {trace['claim']}")
    print(f"task  : {trace['task_type']}  status: {trace['status']}\n")

    for item in trace["measurements"]:
        if tool and item["source_tool"] != tool:
            continue
        print(f"  {item['key']:<26} {item['value']:>13.4f} {item['unit']}")

    print()
    for stage in trace["stages"]:
        for step in stage["steps"]:
            if tool and step.get("tool") not in (None, tool):
                continue
            for note in step["notes"]:
                print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
