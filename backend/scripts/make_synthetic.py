"""Build the sample library offline, with no network access.

    python scripts/make_synthetic.py            # all presets
    python scripts/make_synthetic.py water_body

Every asset produced here is marked ``is_simulated`` in the manifest and carries
a note saying how it was generated. This exists so the application can be
demonstrated on a machine with no internet connection, and so a fresh checkout
without the cached Copernicus imagery still has something to run against.

These are not satellite measurements and the UI will say so.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.sample_sources import FetchedScene, TargetGrid, now_utc, write_scene  # noqa: E402
from app.core.samples import (  # noqa: E402
    PRESETS,
    PRESETS_BY_KEY,
    PresetDef,
    asset_filename,
    load_manifest,
    record_asset,
    save_manifest,
    simulate_sar_from_optical,
)
from app.models.schemas import ImageRole  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

SYNTHETIC_NOTE = (
    "Synthesised offline for demonstration without network access. Land-cover "
    "geometry and spectral values are generated, not observed."
)

# Reflectance x 10000 per class for (blue, green, red, nir, swir16, swir22).
SIGNATURE: dict[str, tuple[int, ...]] = {
    "water": (420, 360, 250, 115, 60, 45),
    "vegetation_dense": (330, 590, 360, 4000, 1450, 640),
    "vegetation_sparse": (520, 760, 720, 2100, 1850, 1150),
    "bare": (1150, 1340, 1720, 2180, 2900, 2480),
    "builtup": (1420, 1520, 1700, 1990, 2600, 2200),
    "cloud": (3100, 3200, 3300, 3400, 2050, 1500),
}

SCL_CODE: dict[str, int] = {
    "water": 6,
    "vegetation_dense": 4,
    "vegetation_sparse": 4,
    "bare": 5,
    "builtup": 5,
    "cloud": 9,
}

CLASS_ORDER = tuple(SIGNATURE)


def utm_grid(lon: float, lat: float, size_px: int, pixel_m: float = 10.0) -> TargetGrid:
    """A UTM grid centred on a coordinate, matching how real scenes are cut."""
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    left = xs[0] - (size_px / 2) * pixel_m
    top = ys[0] + (size_px / 2) * pixel_m
    return TargetGrid(
        crs=f"EPSG:{epsg}",
        transform=Affine(pixel_m, 0.0, left, 0.0, -pixel_m, top),
        width=size_px,
        height=size_px,
    )


def _fractal_field(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Smooth multi-octave noise, used to give land cover organic boundaries."""
    field = np.zeros(shape, dtype="float64")
    amplitude = 1.0
    for octave in (4, 8, 16, 32, 64):
        coarse = rng.random((octave, octave))
        rows = np.linspace(0, octave - 1, shape[0])
        cols = np.linspace(0, octave - 1, shape[1])
        r0 = np.clip(rows.astype(int), 0, octave - 1)
        c0 = np.clip(cols.astype(int), 0, octave - 1)
        field += amplitude * coarse[np.ix_(r0, c0)]
        amplitude *= 0.55
    field -= field.min()
    return field / max(field.max(), 1e-9)


def _field_pattern(shape: tuple[int, int], rng: np.random.Generator, cells: int = 14):
    """A blocky agricultural parcel pattern."""
    rows, cols = shape
    block_r = max(1, rows // cells)
    block_c = max(1, cols // cells)
    small = rng.random((rows // block_r + 1, cols // block_c + 1))
    return np.kron(small, np.ones((block_r, block_c)))[:rows, :cols]


def labels_for(key: str, role: ImageRole, shape: tuple[int, int]) -> np.ndarray:
    """Land-cover labels per preset and slot, as an index into CLASS_ORDER."""
    rng = np.random.default_rng(abs(hash((key, role.value))) % (2**32))
    rows, cols = shape
    terrain = _fractal_field(shape, rng)
    labels = np.full(shape, CLASS_ORDER.index("vegetation_sparse"), dtype="uint8")

    if key == "water_body":
        # A reservoir filling the lower-right, with a sinuous shoreline.
        yy, xx = np.mgrid[0:rows, 0:cols]
        shore = 0.55 * cols + 90 * np.sin(yy / 70.0) + 60 * (terrain - 0.5)
        labels[xx > shore] = CLASS_ORDER.index("water")
        labels[(xx <= shore) & (terrain > 0.62)] = CLASS_ORDER.index("vegetation_dense")
        labels[(xx <= shore) & (terrain < 0.30)] = CLASS_ORDER.index("bare")
        town = (slice(int(rows * 0.55), int(rows * 0.75)), slice(int(cols * 0.08), int(cols * 0.28)))
        labels[town] = np.where(
            rng.random(labels[town].shape) > 0.35,
            CLASS_ORDER.index("builtup"),
            labels[town],
        )

    elif key == "urban_growth":
        labels[terrain > 0.58] = CLASS_ORDER.index("bare")
        parcels = _field_pattern(shape, rng, cells=18)
        labels[parcels > 0.72] = CLASS_ORDER.index("vegetation_dense")
        # A village present at both dates.
        labels[int(rows * 0.34):int(rows * 0.42), int(cols * 0.36):int(cols * 0.46)] = (
            CLASS_ORDER.index("builtup")
        )
        if role is ImageRole.DATE_B:
            # Later date: a new grid-plan development plus an access corridor.
            labels[int(rows * 0.62):int(rows * 0.92), int(cols * 0.10):int(cols * 0.52)] = (
                CLASS_ORDER.index("builtup")
            )
            labels[int(rows * 0.20):int(rows * 0.26), :] = CLASS_ORDER.index("builtup")

    elif key == "flood_urban":
        # River across the top, city below, flood sheets on the plain.
        yy, xx = np.mgrid[0:rows, 0:cols]
        river = 0.22 * rows + 55 * np.sin(xx / 110.0) + 30 * (terrain - 0.5)
        labels[np.abs(yy - river) < 42] = CLASS_ORDER.index("water")
        city = yy > int(rows * 0.48)
        labels[city & (terrain > 0.34)] = CLASS_ORDER.index("builtup")
        labels[city & (terrain <= 0.34)] = CLASS_ORDER.index("vegetation_sparse")
        flood = (yy > int(rows * 0.30)) & (yy < int(rows * 0.52)) & (terrain < 0.42)
        labels[flood] = CLASS_ORDER.index("water")
        if role is ImageRole.OPTICAL:
            # Cloud bank over part of the city: the reason SAR is needed.
            blob = _fractal_field(shape, np.random.default_rng(99))
            cloud = (blob > 0.60) & (yy > int(rows * 0.55)) & (xx < int(cols * 0.62))
            labels[cloud] = CLASS_ORDER.index("cloud")

    elif key == "seasonal_farmland":
        parcels = _field_pattern(shape, rng, cells=16)
        if role is ImageRole.DATE_A:
            labels[:] = CLASS_ORDER.index("vegetation_dense")
            labels[parcels < 0.18] = CLASS_ORDER.index("bare")
        else:
            # Post-harvest: the same parcels, now bare.
            labels[:] = CLASS_ORDER.index("bare")
            labels[parcels > 0.86] = CLASS_ORDER.index("vegetation_sparse")
        # Villages persist across both dates.
        for _ in range(7):
            r = rng.integers(0, rows - 26)
            c = rng.integers(0, cols - 26)
            labels[r:r + 22, c:c + 22] = CLASS_ORDER.index("builtup")

    return labels


def optical_cube(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.zeros((7, *labels.shape), dtype="float64")
    for index, name in enumerate(CLASS_ORDER):
        mask = labels == index
        if not mask.any():
            continue
        for band, value in enumerate(SIGNATURE[name]):
            cube[band][mask] = value
        cube[6][mask] = SCL_CODE[name]
    cube[:6] += rng.normal(0.0, 55.0, cube[:6].shape)
    cube[:6] = np.clip(cube[:6], 1, 10000)
    return cube.astype("uint16")


def build(preset: PresetDef, samples_dir: Path) -> dict[ImageRole, object]:
    print(f"\n=== {preset.key}: {preset.title}  (synthetic)")
    grid = utm_grid(preset.lon, preset.lat, preset.size_px)
    records: dict[ImageRole, object] = {}
    shape = (preset.size_px, preset.size_px)

    base_dates = {
        ImageRole.SINGLE: datetime(2024, 1, 12, tzinfo=timezone.utc),
        ImageRole.DATE_A: datetime(2020, 2, 22, tzinfo=timezone.utc),
        ImageRole.DATE_B: datetime(2026, 3, 12, tzinfo=timezone.utc),
        ImageRole.OPTICAL: datetime(2019, 10, 18, tzinfo=timezone.utc),
        ImageRole.SAR: datetime(2019, 10, 18, tzinfo=timezone.utc),
    }
    if preset.key == "seasonal_farmland":
        base_dates[ImageRole.DATE_A] = datetime(2024, 3, 12, tzinfo=timezone.utc)
        base_dates[ImageRole.DATE_B] = datetime(2024, 5, 31, tzinfo=timezone.utc)

    grid_cube: np.ndarray | None = None
    for role in preset.optical:
        labels = labels_for(preset.key, role, shape)
        cube = optical_cube(labels, seed=abs(hash((preset.key, role.value))) % 10000)
        scene = FetchedScene(
            cube=cube,
            band_names=["blue", "green", "red", "nir", "swir16", "swir22", "scl"],
            grid=grid,
            nodata=0,
            collection="synthetic-optical",
            acquisition_date=base_dates[role],
            tags={
                "PLATFORM": "synthetic",
                "INSTRUMENT": "synthetic-MSI",
                "PRODUCT": "SYNTHETIC_L2A",
                "REFLECTANCE_SCALE": "10000",
                "SIMULATED": "true",
                "SIMULATION_NOTE": SYNTHETIC_NOTE,
            },
        )
        path = samples_dir / asset_filename(preset.key, role)
        write_scene(scene, path)
        records[role] = record_asset(preset, role, scene, path, is_simulated=True)
        if role is preset.grid_role:
            grid_cube = cube
        print(f"    wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")

    for role in preset.sar:
        if grid_cube is None:
            continue
        scene = simulate_sar_from_optical(grid_cube, grid)
        scene.acquisition_date = base_dates[role]
        path = samples_dir / asset_filename(preset.key, role)
        write_scene(scene, path)
        records[role] = record_asset(preset, role, scene, path, is_simulated=True)
        print(f"    wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keys", nargs="*", help="preset keys; default all")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="leave any already-cached scene alone",
    )
    args = parser.parse_args()

    settings = get_settings()
    samples_dir = settings.samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    selected = args.keys or [preset.key for preset in PRESETS]
    unknown = [key for key in selected if key not in PRESETS_BY_KEY]
    if unknown:
        print(f"Unknown preset(s): {', '.join(unknown)}")
        return 2

    manifest = load_manifest(samples_dir)
    for key in selected:
        preset = PRESETS_BY_KEY[key]
        expected = set(preset.optical) | set(preset.sar)
        if args.only_missing and set(manifest.scenes[key].assets) >= expected:
            print(f"\n=== {key}: already cached, leaving alone")
            continue
        manifest.scenes[key].assets = build(preset, samples_dir)  # type: ignore[assignment]

    manifest.generated_at = now_utc()
    save_manifest(manifest, samples_dir)
    print(f"\nManifest written to {samples_dir / 'manifest.json'}")
    print("All assets generated here are flagged as simulated in the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
