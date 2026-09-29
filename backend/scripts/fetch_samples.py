"""Populate the local sample cache from real Copernicus imagery.

    python scripts/fetch_samples.py                # fetch every preset
    python scripts/fetch_samples.py water_body     # fetch one
    python scripts/fetch_samples.py --force        # refetch what is cached
    python scripts/fetch_samples.py --no-sar-fallback

Optical comes from Earth Search (unauthenticated). Radar comes from the
Planetary Computer's Sentinel-1 RTC collection (anonymous SAS token). If radar
cannot be retrieved, a simulated backscatter scene is derived from the real
optical land cover and clearly labelled as simulated, unless
``--no-sar-fallback`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.sample_sources import (  # noqa: E402
    SampleFetchError,
    coverage_fraction,
    fetch_sentinel1_rtc_scene,
    fetch_sentinel2_scene,
    grid_centred_on,
    item_covers_point,
    now_utc,
    search_sentinel1_rtc,
    search_sentinel2,
    write_scene,
    S2_REFERENCE_ASSET,
)
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

logger = logging.getLogger("fetch_samples")


def fetch_preset(
    preset: PresetDef,
    client: httpx.Client,
    samples_dir: Path,
    *,
    allow_sar_fallback: bool,
) -> dict[ImageRole, object]:
    """Fetch every asset of one preset. Returns role -> SampleAssetRecord."""
    print(f"\n=== {preset.key}: {preset.title}")
    print(f"    {preset.place}  ({preset.lat:.3f}N, {preset.lon:.3f}E)")

    records: dict[ImageRole, object] = {}
    grid = None
    grid_cube = None

    # Optical first: it defines the grid that radar is cut onto.
    ordered_optical = sorted(
        preset.optical.items(), key=lambda kv: 0 if kv[0] is preset.grid_role else 1
    )

    for role, spec in ordered_optical:
        items = search_sentinel2(client, _bbox(preset), spec.date_range, spec.max_cloud)
        if not items:
            raise SampleFetchError(
                f"No Sentinel-2 item under {spec.max_cloud:.0f}% cloud in "
                f"{spec.date_range} for {preset.key}/{role.value}."
            )

        item, grid = _choose_covering_item(items, preset, grid)
        cloud = item["properties"].get("eo:cloud_cover")
        print(
            f"    {role.value:8} {item['id']}  "
            f"{item['properties']['datetime'][:10]}  cloud {cloud:.1f}%"
        )
        print(f"    grid     EPSG:{grid.epsg}  {grid.width}x{grid.height} px")

        scene = fetch_sentinel2_scene(
            client, item, preset.lon, preset.lat, preset.size_px, grid=grid
        )
        path = samples_dir / asset_filename(preset.key, role)
        write_scene(scene, path)
        records[role] = record_asset(preset, role, scene, path)
        if role is preset.grid_role:
            grid_cube = scene.cube
        print(f"    wrote    {path.name}  {path.stat().st_size / 1e6:.2f} MB")

    # Radar, cut onto the optical grid so the pair is genuinely co-registered.
    for role, spec in preset.sar.items():
        if grid is None:
            raise SampleFetchError(
                f"{preset.key} requests SAR but has no optical asset to define a grid."
            )
        record = None
        try:
            items = search_sentinel1_rtc(client, _bbox(preset), spec.date_range)
            if not items:
                raise SampleFetchError(
                    f"No Sentinel-1 RTC item in {spec.date_range} for {preset.key}."
                )
            last_error: Exception | None = None
            for item in items:
                try:
                    print(
                        f"    {role.value:8} trying {item['id']}  "
                        f"{item['properties']['datetime'][:10]}"
                    )
                    scene = fetch_sentinel1_rtc_scene(client, item, grid)
                    path = samples_dir / asset_filename(preset.key, role)
                    write_scene(scene, path)
                    record = record_asset(preset, role, scene, path)
                    print(f"    wrote    {path.name}  {path.stat().st_size / 1e6:.2f} MB")
                    break
                except SampleFetchError as exc:
                    print(f"             {exc}")
                    last_error = exc
            if record is None and last_error is not None:
                raise last_error
        except Exception as exc:  # noqa: BLE001 - fall back rather than abort
            print(f"    !! real SAR unavailable: {exc}")
            if not allow_sar_fallback:
                raise
            if grid_cube is None:
                raise SampleFetchError(
                    "Cannot simulate SAR without the optical cube."
                ) from exc
            print("    -> simulating backscatter from the real optical land cover")
            scene = simulate_sar_from_optical(grid_cube, grid)
            path = samples_dir / asset_filename(preset.key, role)
            write_scene(scene, path)
            record = record_asset(preset, role, scene, path, is_simulated=True)
            print(f"    wrote    {path.name} (SIMULATED)")

        records[role] = record

    return records


MIN_COVERAGE = 0.92


def _choose_covering_item(items, preset: PresetDef, grid):
    """Pick the first candidate that genuinely covers the target window.

    Cloud cover is not the only thing that matters. A granule can intersect the
    search box while leaving the area of interest as nodata, so each candidate is
    probed against the COG's overviews before it is accepted.
    """
    footprint_ok = [it for it in items if item_covers_point(it, preset.lon, preset.lat)]
    ordered = footprint_ok + [it for it in items if it not in footprint_ok]

    rejected: list[str] = []
    for candidate in ordered:
        href = candidate["assets"][S2_REFERENCE_ASSET]["href"]
        try:
            candidate_grid = grid or grid_centred_on(
                href, preset.lon, preset.lat, preset.size_px
            )
            covered = coverage_fraction(href, candidate_grid)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            rejected.append(f"{candidate['id']} (probe failed: {exc})")
            continue

        if covered >= MIN_COVERAGE:
            return candidate, candidate_grid

        rejected.append(f"{candidate['id']} ({covered:.0%} covered)")

    detail = "; ".join(rejected[:6]) or "no candidates"
    raise SampleFetchError(
        f"No Sentinel-2 granule covers {preset.place} in this window. Tried: {detail}"
    )


def _bbox(preset: PresetDef) -> list[float]:
    """A small search box around the preset centre, in degrees."""
    half = 0.05
    return [
        preset.lon - half,
        preset.lat - half,
        preset.lon + half,
        preset.lat + half,
    ]


def verify(path: Path) -> str:
    """Re-open a written asset and summarise what the pipeline will see."""
    with rasterio.open(path) as dataset:
        names = [d or "?" for d in dataset.descriptions]
        return (
            f"{dataset.width}x{dataset.height} {dataset.count}band "
            f"{dataset.dtypes[0]} EPSG:{dataset.crs.to_epsg()} "
            f"bands=[{','.join(names)}] "
            f"date={dataset.tags().get('ACQUISITION_DATE', '?')[:10]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keys", nargs="*", help="preset keys to fetch; default all")
    parser.add_argument("--force", action="store_true", help="refetch cached assets")
    parser.add_argument(
        "--no-sar-fallback",
        action="store_true",
        help="fail instead of simulating radar when none can be retrieved",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s | %(message)s",
    )

    settings = get_settings()
    samples_dir = settings.samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    selected = args.keys or [preset.key for preset in PRESETS]
    unknown = [key for key in selected if key not in PRESETS_BY_KEY]
    if unknown:
        print(f"Unknown preset(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(PRESETS_BY_KEY)}")
        return 2

    manifest = load_manifest(samples_dir)
    failures: list[str] = []

    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for key in selected:
            preset = PRESETS_BY_KEY[key]
            scene = manifest.scenes[key]
            expected = set(preset.optical) | set(preset.sar)
            if not args.force and set(scene.assets) >= expected:
                print(f"\n=== {key}: already cached ({len(scene.assets)} assets), skipping")
                continue
            try:
                records = fetch_preset(
                    preset,
                    client,
                    samples_dir,
                    allow_sar_fallback=not args.no_sar_fallback,
                )
                scene.assets = records  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED {key}: {exc}")
                failures.append(f"{key}: {exc}")

    manifest.generated_at = now_utc()
    save_manifest(manifest, samples_dir)

    print("\n--- cached scenes ---")
    for key, scene in manifest.scenes.items():
        if not scene.available:
            print(f"  {key:20} not cached")
            continue
        flag = " [includes SIMULATED asset]" if scene.has_simulated_asset else ""
        print(f"  {key:20} {len(scene.assets)} asset(s){flag}")
        for role, asset in scene.assets.items():
            print(f"      {role.value:8} {verify(samples_dir / asset.filename)}")

    if failures:
        print("\n--- failures ---")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"\nManifest written to {samples_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
