"""Run the learned probe on a cached sample and print what it found.

Shows the two things worth checking: that the classifier produces a sane map of a
real scene it was never trained on, and how far it departs from the published
indices on the same pixels.

    .\\.venv\\Scripts\\python.exe scripts\\probe_landcover.py urban_growth
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.readiness import evaluate_readiness  # noqa: E402
from app.core.samples import load_sample_into_session  # noqa: E402
from app.core.sessions import get_store  # noqa: E402
from app.tools.base import ToolContext  # noqa: E402
from app.tools.indices import SpectralIndexEngine  # noqa: E402
from app.tools.probe import LandCoverProbe  # noqa: E402
from app.core.adapter import CLASSES, load_weights  # noqa: E402


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "urban_growth"

    weights = load_weights(get_settings().models_dir)
    if weights is None:
        print("the probe has not been trained; run scripts/train_probe.py first")
        return 1

    store = get_store()
    session = load_sample_into_session(key, store, get_settings().samples_dir)
    readiness = evaluate_readiness(session, store)
    context = ToolContext(
        session=session,
        store=store,
        readiness=readiness,
        target_classes=["built-up", "water", "vegetation"],
        parameters={"indices": ["NDVI", "NDWI", "MNDWI", "NDBI"]},
    )

    print(f"scene: {key}  ({session.configuration.value})")
    record = weights.training
    print(
        f"model: fitted {record.trained_at:%Y-%m-%d}, "
        f"{record.accuracy:.1%} held-out accuracy on {record.held_out} spectra"
    )

    index_outcome = SpectralIndexEngine().run(context)
    if index_outcome.ok:
        context.upstream[SpectralIndexEngine.name] = index_outcome

    outcome = LandCoverProbe(weights).run(context)
    if not outcome.ok:
        print(f"the probe declined: {outcome.skipped_reason}")
        return 1

    print()
    print(f"{'class':<14}{'area km2':>10}{'mean p':>9}{'held-out recall':>17}")
    print("-" * 50)
    for name in CLASSES:
        area = outcome.measurement(f"probe_area_km2.{name}")
        confidence = outcome.measurement(f"probe_confidence.{name}")
        recall = record.per_class_recall.get(name, 0.0)
        print(
            f"{name:<14}{(area.value if area else 0.0):>10.3f}"
            f"{(confidence.value if confidence else 0.0):>9.3f}{recall:>17.3f}"
        )

    unassigned = outcome.measurement("probe_unassigned_fraction")
    if unassigned is not None:
        print(f"\nleft unassigned: {unassigned.value:.2%}")

    print("\nagainst the published indices, on the same pixels")
    any_comparison = False
    for name in CLASSES:
        overlap = outcome.measurement(f"probe_agreement_iou.{name}")
        if overlap is None:
            continue
        any_comparison = True
        print(
            f"  {name:<12} overlap {overlap.value:>6.1%}   "
            f"probe {overlap.inputs['probe_area_km2']:.3f} km2 vs "
            f"{overlap.inputs['index_mask']} {overlap.inputs['index_area_km2']:.3f} km2"
        )
    if not any_comparison:
        print("  (no comparable index mask was produced for this scene)")

    print()
    for note in outcome.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
