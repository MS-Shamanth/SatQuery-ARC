"""Run every method on one sample and show where they disagree.

The panel this feeds is the most interesting thing in the product, so it is worth
checking against real imagery rather than trusting it: a disagreement map that is
all green proves nothing, and one that is all red means the comparison is broken.

    .\\.venv\\Scripts\\python.exe scripts\\probe_disagreement.py urban_growth
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.adapter import load_weights  # noqa: E402
from app.core.readiness import evaluate_readiness  # noqa: E402
from app.core.samples import load_sample_into_session  # noqa: E402
from app.core.sessions import get_store  # noqa: E402
from app.models.disagreement import AGREEMENT_LABEL  # noqa: E402
from app.tools.base import ToolContext  # noqa: E402
from app.tools.disagreement import DisagreementEngine  # noqa: E402
from app.tools.grounding import GroundingEngine  # noqa: E402
from app.tools.indices import SpectralIndexEngine  # noqa: E402
from app.tools.probe import LandCoverProbe  # noqa: E402
from app.tools.sar import SarBackscatterEngine  # noqa: E402


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "water_body"
    target = sys.argv[2] if len(sys.argv) > 2 else "water"

    store = get_store()
    session = load_sample_into_session(key, store, get_settings().samples_dir)
    readiness = evaluate_readiness(session, store)
    context = ToolContext(
        session=session,
        store=store,
        readiness=readiness,
        target_classes=[target],
        parameters={"indices": ["NDVI", "NDWI", "MNDWI", "NDBI"], "target": target},
    )

    print(f"scene {key}, asking about {target}\n")

    for tool in (SpectralIndexEngine(), GroundingEngine(), SarBackscatterEngine()):
        outcome = tool.run(context)
        mark = "ran" if outcome.ok else "declined"
        print(f"  {tool.name:<28} {mark}"
              + ("" if outcome.ok else f": {outcome.skipped_reason}"))
        if outcome.ok:
            context.upstream[tool.name] = outcome

    weights = load_weights(get_settings().models_dir)
    if weights is not None:
        probe = LandCoverProbe(weights)
        outcome = probe.run(context)
        print(f"  {probe.name:<28} {'ran' if outcome.ok else 'declined'}")
        if outcome.ok:
            context.upstream[probe.name] = outcome

    context.parameters = {}
    engine = DisagreementEngine()
    ready, why = engine.upstream_ready(context)
    if not ready:
        print(f"\nthe disagreement engine declined: {why}")
        return 1

    outcome = engine.run(context)
    if not outcome.ok:
        print(f"\nthe disagreement engine failed: {outcome.skipped_reason}")
        return 1

    report = outcome.artifacts["disagreement.report"]
    print(f"\n  methods compared: {', '.join(report.methods)}")
    print(f"  candidate area    {report.candidate_area_km2:.4f} km2")
    print(f"  agree             {report.agree_area_km2:.4f} km2")
    print(f"  disagree          {report.disagree_area_km2:.4f} km2")
    print(f"  uncertain         {report.uncertain_area_km2:.4f} km2")
    print(f"  CONTESTED         {report.conflicting_fraction:.2%} of candidate area")
    print(f"  regions listed    {len(report.regions)}")

    for region in report.regions[:4]:
        print(f"\n  REGION #{region.region_id}  "
              f"[{AGREEMENT_LABEL[region.agreement]}]  {region.area_km2:.4f} km2")
        if region.centroid_wgs84:
            lon, lat = region.centroid_wgs84
            print(f"    at {lat:.4f}N {lon:.4f}E")
        for opinion in region.opinions:
            print(f"    {opinion.method:<24} -> {opinion.verdict}")
        print(f"    reason: {region.reason}")
        print(f"    action: {region.action}")

    print()
    for note in outcome.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
