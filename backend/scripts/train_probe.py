"""Train the land-cover probe and report what it can actually do.

The probe is the proof that a learned component fits this architecture: it trains,
it measures itself on data it never saw, it registers as a tool, and it runs inside
the same contract and trace as the deterministic engines while being labelled a
trained model rather than a computation.

    .\\.venv\\Scripts\\python.exe scripts\\train_probe.py

Writes data/models/landcover_probe.json. Delete that file and the probe declines to
register, with the reason shown in the capability panel. Nothing else in the system
depends on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.adapter import CLASSES, FEATURES, save_weights, train  # noqa: E402


def main() -> int:
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

    print(f"training on {per_class} synthetic spectra per class")
    print(f"features: {', '.join(FEATURES)}")
    weights = train(per_class=per_class)
    record = weights.training

    print()
    print(f"  samples fitted   {record.samples:,}")
    print(f"  held out         {record.held_out:,}")
    print(f"  final loss       {record.final_loss:.4f}")
    print(f"  held-out accuracy {record.accuracy:.4f}")
    print()
    print("  recall by class (on the held-out synthetic spectra)")
    for name in CLASSES:
        print(f"    {name:<12} {record.per_class_recall[name]:.4f}")

    print()
    print("  confusion, rows are true and columns predicted")
    header = "               " + "".join(f"{name[:9]:>11}" for name in CLASSES)
    print(header)
    for index, name in enumerate(CLASSES):
        row = "".join(f"{count:>11,}" for count in record.confusion[index])
        print(f"    {name:<11}{row}")

    print()
    print("  what pushes hardest towards each class")
    for name in CLASSES:
        top = weights.influences(name)[:3]
        rendered = ", ".join(f"{feature} {value:+.2f}" for feature, value in top)
        print(f"    {name:<12} {rendered}")

    path = save_weights(weights, get_settings().models_dir)
    print()
    print(f"written to {path}")
    print()
    for note in record.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
