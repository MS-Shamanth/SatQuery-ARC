"""Sample scene library tests.

These cover the preset registry, the manifest round-trip, the honesty guarantees
around simulated imagery, and the physical plausibility of the simulated SAR
fallback. Network fetching itself is not exercised here: the providers are
verified by running scripts/fetch_samples.py, and unit tests must not depend on
Copernicus being reachable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from rasterio.transform import Affine

from app.core.sample_sources import (
    TargetGrid,
    item_covers_point,
    rfc3339_range,
)
from app.core.samples import (
    PRESETS,
    PRESETS_BY_KEY,
    SIMULATED_SIGMA0_DB,
    SIMULATION_NOTE,
    SampleNotAvailable,
    asset_filename,
    blank_manifest,
    load_manifest,
    load_sample_into_session,
    record_asset,
    save_manifest,
    simulate_sar_from_optical,
)
from app.models.schemas import ImageRole, InputConfiguration
from tests.raster_fixtures import LandCover, make_labels, optical_from_labels


def _grid(size: int = 64) -> TargetGrid:
    return TargetGrid(
        crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 2000000.0),
        width=size,
        height=size,
    )


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------


def test_there_is_one_preset_per_scripted_demo() -> None:
    """Four scripted demos, the pair that resolves the fourth, and one refusal.

    Demo 4 refuses a comparison and names the acquisition that would settle it, so
    one preset is that acquisition: the refusal can be acted on rather than only
    read.

    Demo 5 exists because the library would otherwise only contain scenes the
    system can answer. It is real construction measured with an index that cannot
    see it, kept deliberately so the failure is demonstrable rather than described.
    """
    assert len(PRESETS) == 6
    demos = sorted(preset.demo for preset in PRESETS)
    assert demos == [
        "Demo 1 - single-image grounding",
        "Demo 2 - claim investigation",
        "Demo 3 - optical and SAR debate",
        "Demo 4 - seasonal confounder",
        "Demo 4 - the corrected comparison",
        "Demo 5 - when the instrument cannot answer",
    ]


def test_every_demo_scene_is_measured_by_an_index_that_suits_it() -> None:
    """The claim-investigation demo has to rest on a trustworthy measurement.

    Reaching a supported verdict on a scene whose index cannot separate it would
    be a number contradicted by its own map, which is the one failure this system
    is built to prevent. The reservoir scene is the demo that supports a claim,
    and the urban scene is the demo that declines to.
    """
    reservoir = PRESETS_BY_KEY["reservoir_change"]
    assert reservoir.demo == "Demo 2 - claim investigation"
    assert "MNDWI" in reservoir.description or "water" in reservoir.description

    urban = PRESETS_BY_KEY["urban_growth"]
    assert urban.demo == "Demo 5 - when the instrument cannot answer"
    # The description has to say what goes wrong, or the scene reads as a bug.
    assert "cannot" in urban.description


def test_the_corrected_pair_answers_the_scene_it_corrects() -> None:
    """A remedy has to name what it settles and for which input."""
    from app.models.contract import ConfounderKind

    corrected = next(
        preset for preset in PRESETS if preset.key == "seasonal_farmland_same_season"
    )
    original = next(
        preset for preset in PRESETS if preset.key == "seasonal_farmland"
    )

    assert corrected.corrects == original.key
    assert ConfounderKind.SEASONALITY in corrected.remedies
    assert corrected.remedy_note
    # Same ground, so only the calendar differs between the two runs.
    assert (corrected.lon, corrected.lat) == (original.lon, original.lat)
    # Both of its dates sit in the same month, which is what removes phenology.
    assert all("-03-" in spec.date_range for spec in corrected.optical.values())


def test_preset_keys_are_unique_and_route_safe() -> None:
    keys = [preset.key for preset in PRESETS]
    assert len(keys) == len(set(keys))
    for key in keys:
        # Must satisfy the route pattern ^[a-z0-9_]{3,40}$.
        assert 3 <= len(key) <= 40
        assert key.replace("_", "").isalnum() and key.islower()


def test_every_preset_declares_slots_matching_its_configuration() -> None:
    for preset in PRESETS:
        roles = set(preset.optical) | set(preset.sar)
        if preset.configuration is InputConfiguration.SINGLE:
            assert roles == {ImageRole.SINGLE}, preset.key
        elif preset.configuration is InputConfiguration.CROSS_MODAL_PAIR:
            assert roles == {ImageRole.OPTICAL, ImageRole.SAR}, preset.key
        elif preset.configuration is InputConfiguration.BI_TEMPORAL_PAIR:
            assert roles == {ImageRole.DATE_A, ImageRole.DATE_B}, preset.key
        # The grid-defining role must be one this preset actually fetches.
        assert preset.grid_role in roles, preset.key


def test_cross_modal_preset_defines_its_grid_from_the_optical_slot() -> None:
    """Radar is cut onto the optical grid, which is what co-registers the pair."""
    preset = PRESETS_BY_KEY["flood_urban"]
    assert preset.grid_role is ImageRole.OPTICAL
    assert preset.sar


def test_urban_growth_pair_uses_the_same_season_on_purpose() -> None:
    """Comparing like with like is what lets a SUPPORTED verdict stand."""
    preset = PRESETS_BY_KEY["urban_growth"]
    months = []
    for spec in preset.optical.values():
        start, end = spec.date_range.split("/")
        months.append((int(start[5:7]), int(end[5:7])))
    assert months[0] == months[1]


def test_seasonal_preset_spans_the_harvest_on_purpose() -> None:
    """The kill-shot needs a real phenological gap, not a random one."""
    preset = PRESETS_BY_KEY["seasonal_farmland"]
    start_a = preset.optical[ImageRole.DATE_A].date_range.split("/")[0]
    start_b = preset.optical[ImageRole.DATE_B].date_range.split("/")[0]
    month_a, month_b = int(start_a[5:7]), int(start_b[5:7])
    assert month_a == 2 or month_a == 3
    assert month_b == 5
    assert abs(month_b - month_a) >= 2


def test_every_preset_suggests_queries_drawn_from_the_problem_statement() -> None:
    for preset in PRESETS:
        assert preset.suggested_queries, preset.key
        assert all(q.strip() for q in preset.suggested_queries)


def test_asset_filenames_are_deterministic_and_slot_scoped() -> None:
    assert asset_filename("urban_growth", ImageRole.DATE_A) == "urban_growth__date_a.tif"
    assert asset_filename("flood_urban", ImageRole.SAR) == "flood_urban__sar.tif"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_blank_manifest_advertises_every_preset_as_uncached(sample_library) -> None:
    manifest = blank_manifest()
    assert set(manifest.scenes) == set(PRESETS_BY_KEY)
    for scene in manifest.scenes.values():
        assert scene.available is False
        assert scene.assets == {}


def test_manifest_round_trips_through_disk(sample_library) -> None:
    manifest = blank_manifest()
    manifest.scenes["water_body"].suggested_queries = ["probe"]
    save_manifest(manifest, sample_library)

    reloaded = load_manifest(sample_library)
    assert reloaded.scenes["water_body"].suggested_queries == ["probe"]
    assert set(reloaded.scenes) == set(PRESETS_BY_KEY)


def test_missing_manifest_reports_nothing_cached(sample_library) -> None:
    manifest = load_manifest(sample_library)
    assert all(not scene.available for scene in manifest.scenes.values())


def test_corrupt_manifest_does_not_break_the_library(sample_library) -> None:
    (sample_library / "manifest.json").write_text("{not json", encoding="utf-8")
    manifest = load_manifest(sample_library)
    assert set(manifest.scenes) == set(PRESETS_BY_KEY)
    assert all(not scene.available for scene in manifest.scenes.values())


def test_manifest_drops_assets_whose_files_are_absent(sample_library, tmp_path) -> None:
    """A fresh checkout has the manifest but not the regenerable GeoTIFFs."""
    scene = make_optical_asset(sample_library, "water_body", ImageRole.SINGLE)
    manifest = blank_manifest()
    manifest.scenes["water_body"].assets = {ImageRole.SINGLE: scene}
    save_manifest(manifest, sample_library)

    assert load_manifest(sample_library).scenes["water_body"].available is True

    (sample_library / scene.filename).unlink()
    assert load_manifest(sample_library).scenes["water_body"].available is False


def make_optical_asset(samples_dir, key: str, role: ImageRole):
    """Write a small optical asset and return its manifest record."""
    from app.core.sample_sources import FetchedScene, write_scene

    labels = make_labels(64, 64)
    cube = optical_from_labels(labels)
    fetched = FetchedScene(
        cube=cube,
        band_names=["blue", "green", "red", "nir", "swir16", "swir22"],
        grid=_grid(64),
        nodata=0,
        collection="sentinel-2-l2a",
        item_id="S2B_TEST_20240112_0_L2A",
        item_url="https://example.invalid/items/S2B_TEST_20240112_0_L2A",
        acquisition_date=datetime(2024, 1, 12, tzinfo=timezone.utc),
        cloud_cover=0.4,
        platform="sentinel-2b",
        tags={"INSTRUMENT": "MSI"},
    )
    path = samples_dir / asset_filename(key, role)
    write_scene(fetched, path)
    return record_asset(PRESETS_BY_KEY[key], role, fetched, path)


def test_real_asset_records_its_stac_provenance(sample_library) -> None:
    record = make_optical_asset(sample_library, "water_body", ImageRole.SINGLE)
    prov = record.provenance

    assert prov.is_simulated is False
    assert prov.simulation_note is None
    assert prov.stac_item_id == "S2B_TEST_20240112_0_L2A"
    assert prov.stac_url and prov.stac_url.startswith("https://")
    assert prov.collection == "sentinel-2-l2a"
    assert prov.acquisition_date is not None
    assert prov.epsg == 32643
    assert prov.gsd_m == pytest.approx(10.0)
    assert prov.licence and "Copernicus" in prov.licence
    assert prov.attribution and "Copernicus" in prov.attribution
    assert record.sha256 and len(record.sha256) == 64
    assert record.width == 64 and record.height == 64


# ---------------------------------------------------------------------------
# Simulated SAR: must be plausible, and must be labelled
# ---------------------------------------------------------------------------


def test_simulated_sar_is_always_flagged_and_explained(sample_library) -> None:
    """The one thing that must never slip: a simulation passing as measurement."""
    from app.core.sample_sources import write_scene

    labels = make_labels(64, 64)
    optical = optical_from_labels(labels)
    scene = simulate_sar_from_optical(optical, _grid(64))
    path = sample_library / asset_filename("flood_urban", ImageRole.SAR)
    write_scene(scene, path)

    record = record_asset(
        PRESETS_BY_KEY["flood_urban"], ImageRole.SAR, scene, path, is_simulated=True
    )
    prov = record.provenance

    assert prov.is_simulated is True
    assert prov.simulation_note == SIMULATION_NOTE
    assert "not a radar measurement" in prov.simulation_note
    assert prov.stac_item_id is None
    # Simulated data carries no Copernicus licence claim.
    assert prov.licence is None
    assert prov.attribution is None


def test_collection_alone_is_enough_to_flag_a_simulation(sample_library) -> None:
    """Even without the explicit flag, the simulated collection marks itself."""
    from app.core.sample_sources import write_scene

    scene = simulate_sar_from_optical(optical_from_labels(make_labels(48, 48)), _grid(48))
    path = sample_library / "probe.tif"
    write_scene(scene, path)
    record = record_asset(PRESETS_BY_KEY["flood_urban"], ImageRole.SAR, scene, path)
    assert record.provenance.is_simulated is True


def test_simulated_sar_orders_land_cover_the_way_radar_does() -> None:
    """Water must be darkest and built-up brightest, or the demo teaches a lie."""
    labels = np.full((120, 120), int(LandCover.VEGETATION), dtype="uint8")
    labels[:40] = int(LandCover.WATER)
    labels[40:80] = int(LandCover.BUILTUP)
    optical = optical_from_labels(labels, noise_sigma=5.0)

    scene = simulate_sar_from_optical(optical, _grid(120))
    vv_db = 10.0 * np.log10(np.maximum(scene.cube[0], 1e-12))

    water = float(vv_db[:40].mean())
    builtup = float(vv_db[40:80].mean())
    vegetation = float(vv_db[80:].mean())

    assert water < vegetation < builtup
    assert builtup - water > 12.0
    assert water < -15.0


def test_simulated_sar_produces_two_polarisations_in_linear_power() -> None:
    scene = simulate_sar_from_optical(optical_from_labels(make_labels(48, 48)), _grid(48))
    assert scene.cube.shape == (2, 48, 48)
    assert scene.band_names == ["VV", "VH"]
    assert scene.cube.dtype == np.float32
    # Linear power is strictly positive; decibels would be negative.
    assert float(scene.cube.min()) > 0.0
    assert scene.tags["BACKSCATTER_UNITS"] == "gamma0 linear power"


def test_cross_polarisation_is_weaker_than_co_polarisation() -> None:
    """VH returns less than VV for every modelled class, as in real data."""
    for name, (vv, vh) in SIMULATED_SIGMA0_DB.items():
        assert vh < vv, name


def test_simulated_sar_is_reproducible() -> None:
    optical = optical_from_labels(make_labels(48, 48))
    first = simulate_sar_from_optical(optical, _grid(48), seed=5)
    second = simulate_sar_from_optical(optical, _grid(48), seed=5)
    assert np.array_equal(first.cube, second.cube)


def test_simulated_sar_speckle_varies_within_a_uniform_class() -> None:
    """Speckle is what makes radar look like radar."""
    labels = np.full((80, 80), int(LandCover.VEGETATION), dtype="uint8")
    scene = simulate_sar_from_optical(optical_from_labels(labels), _grid(80))
    assert float(scene.cube[0].std()) > 0.0


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2024-01-01/2024-03-20", "2024-01-01T00:00:00Z/2024-03-20T23:59:59Z"),
        (
            "2024-01-01T06:00:00Z/2024-03-20T18:00:00Z",
            "2024-01-01T06:00:00Z/2024-03-20T18:00:00Z",
        ),
        ("2024-01-01/..", "2024-01-01T00:00:00Z/.."),
        ("not-a-range", "not-a-range"),
    ],
)
def test_date_ranges_are_normalised_to_rfc3339(given: str, expected: str) -> None:
    """Earth Search rejects bare calendar dates, so the interval is expanded."""
    assert rfc3339_range(given) == expected


def test_item_footprint_containment_is_checked_not_just_bbox_overlap() -> None:
    """A granule can intersect the search box yet leave the target as nodata."""
    square = {
        "type": "Polygon",
        "coordinates": [[[72.0, 22.0], [73.0, 22.0], [73.0, 23.0], [72.0, 23.0], [72.0, 22.0]]],
    }
    assert item_covers_point({"geometry": square}, 72.5, 22.5) is True
    assert item_covers_point({"geometry": square}, 74.5, 22.5) is False
    # No geometry at all falls through to the empirical coverage probe.
    assert item_covers_point({}, 0.0, 0.0) is True


def test_target_grid_reports_bounds_and_epsg() -> None:
    grid = _grid(100)
    left, bottom, right, top = grid.bounds
    assert (left, top) == (600000.0, 2000000.0)
    assert right == pytest.approx(601000.0)
    assert bottom == pytest.approx(1999000.0)
    assert grid.epsg == 32643


# ---------------------------------------------------------------------------
# Loading into a session
# ---------------------------------------------------------------------------


def test_loading_a_sample_populates_a_session(sample_library, store) -> None:
    record = make_optical_asset(sample_library, "water_body", ImageRole.SINGLE)
    manifest = blank_manifest()
    manifest.scenes["water_body"].assets = {ImageRole.SINGLE: record}
    save_manifest(manifest, sample_library)

    session = load_sample_into_session("water_body", store, sample_library)

    assert session.configuration is InputConfiguration.SINGLE
    assert ImageRole.SINGLE in session.images
    assert session.images[ImageRole.SINGLE].metadata.geo.epsg == 32643
    assert any("water_body" in note for note in session.notes)
    # The cached file is copied, not moved: the library survives the load.
    assert (sample_library / record.filename).exists()


def test_loading_a_simulated_scene_says_so_in_the_session(
    sample_library, store
) -> None:
    from app.core.sample_sources import write_scene

    scene = simulate_sar_from_optical(optical_from_labels(make_labels(48, 48)), _grid(48))
    path = sample_library / asset_filename("flood_urban", ImageRole.SAR)
    write_scene(scene, path)
    sar_record = record_asset(
        PRESETS_BY_KEY["flood_urban"], ImageRole.SAR, scene, path, is_simulated=True
    )
    optical_record = make_optical_asset(sample_library, "flood_urban", ImageRole.OPTICAL)

    manifest = blank_manifest()
    manifest.scenes["flood_urban"].assets = {
        ImageRole.OPTICAL: optical_record,
        ImageRole.SAR: sar_record,
    }
    save_manifest(manifest, sample_library)

    session = load_sample_into_session("flood_urban", store, sample_library)

    assert session.configuration is InputConfiguration.CROSS_MODAL_PAIR
    assert any("simulated" in note.lower() for note in session.notes)


def test_loading_an_unknown_sample_is_refused(sample_library, store) -> None:
    with pytest.raises(SampleNotAvailable, match="No sample scene"):
        load_sample_into_session("no_such_scene", store, sample_library)


def test_loading_an_uncached_sample_explains_how_to_fix_it(
    sample_library, store
) -> None:
    with pytest.raises(SampleNotAvailable) as excinfo:
        load_sample_into_session("water_body", store, sample_library)
    assert "fetch_samples.py" in str(excinfo.value)
    assert "make_synthetic.py" in str(excinfo.value)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_samples_route_lists_every_preset(api, sample_library) -> None:
    body = api.get("/api/samples").json()
    assert set(body["scenes"]) == set(PRESETS_BY_KEY)
    scene = body["scenes"]["seasonal_farmland"]
    assert scene["demo"] == "Demo 4 - seasonal confounder"
    assert scene["place"].startswith("Irrigated cropland")
    assert scene["suggested_queries"]


def test_samples_route_reports_uncached_scenes_rather_than_hiding_them(
    api, sample_library
) -> None:
    body = api.get("/api/samples").json()
    assert body["scenes"]["water_body"]["assets"] == {}


def test_load_route_creates_a_session_from_a_cached_scene(
    api, sample_library, store
) -> None:
    record = make_optical_asset(sample_library, "water_body", ImageRole.SINGLE)
    manifest = blank_manifest()
    manifest.scenes["water_body"].assets = {ImageRole.SINGLE: record}
    save_manifest(manifest, sample_library)

    response = api.post("/api/samples/water_body/load")

    assert response.status_code == 201
    body = response.json()
    assert body["configuration"] == "single"
    assert "single" in body["images"]
    assert body["images"]["single"]["metadata"]["geo"]["epsg"] == 32643

    # The readiness gate runs against the loaded sample like any upload.
    readiness = api.get(f"/api/sessions/{body['session_id']}/readiness").json()
    assert readiness["verdict"] in ("ready", "ready_with_warnings")


def test_load_route_404s_for_an_uncached_scene(api, sample_library) -> None:
    response = api.post("/api/samples/urban_growth/load")
    assert response.status_code == 404
    assert "fetch_samples.py" in response.json()["detail"]


def test_load_route_rejects_a_malformed_key(api, sample_library) -> None:
    assert api.post("/api/samples/Bad-Key/load").status_code == 422
    assert api.post("/api/samples/../etc/load").status_code in (404, 422)
