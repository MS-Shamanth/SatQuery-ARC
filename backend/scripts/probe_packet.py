"""Write a packet for one cached sample and report what the audit found.

Run against the real sample library so the report is exercised on real
measurements, real readiness checks and a real verdict, which is where an
attribution gap actually shows up.

    .\\.venv\\Scripts\\python.exe scripts\\probe_packet.py seasonal_farmland "Did vegetation decrease?"
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.contract import generate_contract  # noqa: E402
from app.core.orchestrator import execute_run  # noqa: E402
from app.core.overlays import persist_base_layers, persist_outcome_layers  # noqa: E402
from app.core.packet import build_packet  # noqa: E402
from app.core.readiness import evaluate_readiness  # noqa: E402
from app.core.registry import get_registry  # noqa: E402
from app.core.samples import load_sample_into_session  # noqa: E402
from app.core.sessions import get_store  # noqa: E402
from app.models.trace import StageId  # noqa: E402
from app.tools.base import ToolContext  # noqa: E402


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "water_body"
    query = sys.argv[2] if len(sys.argv) > 2 else None

    store = get_store()
    session = load_sample_into_session(key, store, get_settings().samples_dir)
    readiness = evaluate_readiness(session, store)
    registry = get_registry()
    context = ToolContext(session=session, store=store, readiness=readiness)

    question = query or "Highlight the water body referred to in the query"
    contract = generate_contract(question, context, registry, refresh=True)
    context.contract = contract
    context.target_classes = list(contract.target_classes)

    directory = store.session_dir(session.session_id) / "packet"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    layers: list = []

    def compose(trace, outcomes):
        layers.extend(persist_base_layers(directory, session, store))
        for tool, outcome in outcomes.items():
            if outcome.ok:
                layers.extend(persist_outcome_layers(directory, tool, outcome))
        return build_packet(
            trace,
            outcomes,
            directory,
            session=session,
            readiness=readiness,
            layers=layers,
        )

    trace, _ = execute_run(contract, context, registry, compose=compose)

    stage = trace.stage(StageId.COMPOSE_PACKET)
    print(f"sample        : {key}")
    print(f"question      : {question}")
    print(f"run status    : {trace.status.value}")
    print(f"packet stage  : {stage.status.value if stage else 'missing'}")

    packet = trace.packet
    if packet is None:
        print("NO PACKET WAS WRITTEN")
        return 1

    print(f"files         : {len(packet.files)}  ({packet.total_bytes / 1024:.0f} kB)")
    for item in packet.files:
        print(f"    {item.filename:<34} {item.size_bytes:>9,} B  {item.role.value}")
    if packet.archive:
        print(f"    {packet.archive.filename:<34} {packet.archive.size_bytes:>9,} B  archive")

    audit = packet.audit
    print(f"figures       : {audit.traced} of {audit.checked} attributed")
    if audit.untraceable:
        print("UNTRACEABLE   :", ", ".join(audit.untraceable))
        print("\nwhere each registered figure came from:")
        for figure in packet.figures:
            print(f"    {figure.display[:60]:<62} | {figure.where}")
        return 1

    print("all figures attributed")
    print(f"\nwritten to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
