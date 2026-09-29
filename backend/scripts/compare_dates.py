"""Put the two dates side by side with a mask, to judge what the mask found.

A change mask can look plausible over one date and absurd over the other. Drawing
both with the same overlay is the cheapest way to tell a real transition from an
index tracking something else.

    .\\.venv\\Scripts\\python.exe scripts\\compare_dates.py change_gain
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(*keys: str) -> int:
    from PIL import Image, ImageDraw

    runs = sorted(
        Path("data/sessions").glob("*/runs/*/layers.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        print("no rendered layers on disk")
        return 1
    directory = runs[-1].parent

    panels: list[Image.Image] = []
    for base in sorted(directory.glob("base__*.png")):
        canvas = Image.open(base).convert("RGBA")
        for png in sorted(directory.glob("*.png")):
            if png.name.startswith("base__"):
                continue
            if keys and not any(key in png.name for key in keys):
                continue
            overlay = Image.open(png).convert("RGBA").resize(canvas.size, Image.NEAREST)
            canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        label = base.stem.replace("base__", "")
        draw.rectangle([0, 0, 90, 16], fill=(0, 0, 0, 200))
        draw.text((5, 3), label, fill=(255, 255, 255, 255))
        panels.append(canvas)
        print(f"panel {label} {canvas.size}")

    if not panels:
        print("no base layers")
        return 1

    width = sum(panel.width for panel in panels) + 8 * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    strip = Image.new("RGBA", (width, height), (10, 14, 22, 255))
    offset = 0
    for panel in panels:
        strip.paste(panel, (offset, 0))
        offset += panel.width + 8

    destination = Path("data/cache/_date_compare.png")
    strip.convert("RGB").save(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
