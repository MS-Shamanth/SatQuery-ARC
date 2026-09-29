"""Composite a run's base layer with its mask overlays for visual inspection.

    .venv\\Scripts\\python.exe scripts\\preview_layers.py change_gain change_loss
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(*keys: str) -> int:
    from PIL import Image

    runs = sorted(
        Path("data/sessions").glob("*/runs/*/layers.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        print("no rendered layers on disk")
        return 1
    directory = runs[-1].parent

    bases = sorted(directory.glob("base__*.png"))
    if not bases:
        print("no base layer")
        return 1
    out = Image.open(bases[-1]).convert("RGBA")
    print(f"base: {bases[-1].name} {out.size}")

    wanted = keys or ("",)
    for png in sorted(directory.glob("*.png")):
        if png.name.startswith("base__"):
            continue
        if not any(key in png.name for key in wanted):
            continue
        layer = Image.open(png).convert("RGBA").resize(out.size, Image.NEAREST)
        out = Image.alpha_composite(out, layer)
        print(f"  over: {png.name}")

    destination = Path("data/cache/_layer_preview.png")
    out.save(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
