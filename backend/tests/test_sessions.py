"""Session store and ingest API tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.sessions import InvalidUpload, SessionNotFound, safe_extension
from app.models.schemas import ImageRole, InputConfiguration, Modality
from tests.raster_fixtures import make_optical_scene, make_sar_scene


def _upload(api, session_id: str, role: str, path, filename: str | None = None):
    with path.open("rb") as handle:
        return api.post(
            f"/api/sessions/{session_id}/images",
            data={"role": role},
            files={"file": (filename or path.name, handle, "image/tiff")},
        )


def _new_session(api) -> str:
    response = api.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


# ---------------------------------------------------------------------------
# Store behaviour
# ---------------------------------------------------------------------------


def test_create_session_makes_a_durable_record(store) -> None:
    record = store.create()
    assert len(record.session_id) == 32
    assert record.configuration is InputConfiguration.INCOMPLETE
    # Reloaded from disk, not from memory.
    assert store.load(record.session_id).session_id == record.session_id


def test_malformed_session_id_is_rejected_before_touching_the_filesystem(store) -> None:
    """Guards against path traversal through the session id."""
    for bad in ("../../etc", "..", "", "nope", "0" * 31, "g" * 32):
        with pytest.raises(SessionNotFound):
            store.session_dir(bad)


def test_unsupported_extension_is_rejected() -> None:
    assert safe_extension("scene.TIF") == ".tif"
    assert safe_extension("bench.png") == ".png"
    with pytest.raises(InvalidUpload):
        safe_extension("payload.exe")
    with pytest.raises(InvalidUpload):
        safe_extension("noextension")


def test_ingest_extracts_metadata_and_renders_a_preview(store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "optical.tif", width=160, height=120)
    record = store.create()

    image = store.ingest(
        record.session_id, ImageRole.SINGLE, scene.path, "optical.tif", move=False
    )

    assert image.metadata.width == 160
    assert image.metadata.geo.gsd_m == pytest.approx(10.0)
    assert image.metadata.modality.modality is Modality.OPTICAL
    assert image.thumbnail_available is True
    assert image.thumbnail_recipe is not None
    assert len(image.sha256) == 64
    assert image.ingest_ms > 0
    # Stored under the role name, never under the client-supplied filename.
    assert image.stored_filename == "single.tif"


def test_ingest_of_an_unreadable_file_leaves_no_trace(store, tmp_path) -> None:
    from app.core.raster_io import RasterIngestError

    bogus = tmp_path / "broken.tif"
    bogus.write_bytes(b"not a raster at all")
    record = store.create()

    with pytest.raises(RasterIngestError):
        store.ingest(record.session_id, ImageRole.SINGLE, bogus, "broken.tif", move=False)

    reloaded = store.load(record.session_id)
    assert reloaded.images == {}
    assert not (store.session_dir(record.session_id) / "images" / "single.tif").exists()


def test_reuploading_a_slot_replaces_the_previous_file(store, tmp_path) -> None:
    first = make_optical_scene(tmp_path / "a.tif", width=64, height=64)
    second = make_sar_scene(tmp_path / "b.tif", width=96, height=96)
    record = store.create()

    store.ingest(record.session_id, ImageRole.SINGLE, first.path, "a.tif", move=False)
    store.ingest(record.session_id, ImageRole.SINGLE, second.path, "b.tif", move=False)

    reloaded = store.load(record.session_id)
    assert len(reloaded.images) == 1
    image = reloaded.images[ImageRole.SINGLE]
    assert image.metadata.width == 96
    assert image.metadata.modality.modality is Modality.SAR
    # Exactly one file survives in the slot: no stale copy is left behind.
    stored = list((store.session_dir(record.session_id) / "images").iterdir())
    assert [path.name for path in stored] == ["single.tif"]


def test_reupload_with_a_different_extension_removes_the_old_file(
    store, tmp_path
) -> None:
    """A slot must not end up holding two files under different extensions."""
    import numpy as np

    from tests.raster_fixtures import write_raster

    tiff = make_optical_scene(tmp_path / "a.tif", width=64, height=64)
    png = write_raster(
        tmp_path / "b.png",
        np.random.default_rng(1).integers(0, 255, (3, 64, 64), dtype="uint8"),
        epsg=None,
        driver="PNG",
    )
    record = store.create()

    store.ingest(record.session_id, ImageRole.SINGLE, tiff.path, "a.tif", move=False)
    store.ingest(record.session_id, ImageRole.SINGLE, png, "b.png", move=False)

    stored = sorted(
        path.name for path in (store.session_dir(record.session_id) / "images").iterdir()
    )
    assert stored == ["single.png"]


@pytest.mark.parametrize(
    ("roles", "expected"),
    [
        ([ImageRole.SINGLE], InputConfiguration.SINGLE),
        ([ImageRole.OPTICAL, ImageRole.SAR], InputConfiguration.CROSS_MODAL_PAIR),
        ([ImageRole.DATE_A, ImageRole.DATE_B], InputConfiguration.BI_TEMPORAL_PAIR),
        ([ImageRole.DATE_A], InputConfiguration.INCOMPLETE),
        ([ImageRole.OPTICAL], InputConfiguration.INCOMPLETE),
    ],
)
def test_configuration_is_derived_from_filled_slots(
    store, tmp_path, roles, expected
) -> None:
    record = store.create()
    for index, role in enumerate(roles):
        scene = make_optical_scene(tmp_path / f"s{index}.tif", width=48, height=48)
        store.ingest(record.session_id, role, scene.path, f"s{index}.tif", move=False)

    assert store.load(record.session_id).configuration is expected


def test_remove_image_clears_the_slot_and_its_preview(store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=48, height=48)
    record = store.create()
    store.ingest(record.session_id, ImageRole.OPTICAL, scene.path, "a.tif", move=False)

    assert store.remove_image(record.session_id, ImageRole.OPTICAL) is True
    assert store.load(record.session_id).images == {}
    assert not store.thumbnail_path(record.session_id, ImageRole.OPTICAL).exists()
    assert store.remove_image(record.session_id, ImageRole.OPTICAL) is False


def test_delete_session_removes_everything(store, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=48, height=48)
    record = store.create()
    store.ingest(record.session_id, ImageRole.SINGLE, scene.path, "a.tif", move=False)

    assert store.delete(record.session_id) is True
    assert not store.session_dir(record.session_id).exists()
    with pytest.raises(SessionNotFound):
        store.load(record.session_id)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def test_post_session_returns_a_new_id(api) -> None:
    body = api.post("/api/sessions").json()
    assert len(body["session_id"]) == 32
    assert body["images"] == {}
    assert body["configuration"] == "incomplete"


def test_upload_returns_measured_metadata(api, tmp_path) -> None:
    scene = make_optical_scene(
        tmp_path / "scene.tif",
        width=240,
        height=180,
        acquisition_date=datetime(2020, 3, 14, tzinfo=timezone.utc),
    )
    session_id = _new_session(api)

    response = _upload(api, session_id, "single", scene.path)
    assert response.status_code == 201

    body = response.json()
    meta = body["metadata"]
    assert meta["width"] == 240 and meta["height"] == 180
    assert meta["band_count"] == 6
    assert meta["geo"]["epsg"] == 32643
    assert meta["geo"]["gsd_m"] == pytest.approx(10.0)
    assert meta["geo"]["gsd_method"] == "projected-linear-unit"
    assert meta["geo"]["area_km2"] == pytest.approx(240 * 10 * 180 * 10 / 1e6)
    assert meta["modality"]["modality"] == "optical"
    assert meta["acquisition_date"].startswith("2020-03-14")
    assert meta["day_of_year"] == 74
    assert [band["role"] for band in meta["bands"]] == [
        "blue", "green", "red", "nir", "swir16", "swir22",
    ]


def test_uploading_both_modalities_yields_a_cross_modal_configuration(
    api, tmp_path
) -> None:
    optical = make_optical_scene(tmp_path / "opt.tif", width=96, height=96)
    sar = make_sar_scene(tmp_path / "sar.tif", width=96, height=96)
    session_id = _new_session(api)

    assert _upload(api, session_id, "optical", optical.path).status_code == 201
    assert _upload(api, session_id, "sar", sar.path).status_code == 201

    body = api.get(f"/api/sessions/{session_id}").json()
    assert body["configuration"] == "cross_modal_pair"
    assert body["images"]["sar"]["metadata"]["modality"]["modality"] == "sar"
    assert body["images"]["optical"]["metadata"]["modality"]["modality"] == "optical"


def test_unsupported_file_type_returns_415(api, tmp_path) -> None:
    blob = tmp_path / "notes.txt"
    blob.write_text("hello")
    session_id = _new_session(api)

    response = _upload(api, session_id, "single", blob, filename="notes.txt")
    assert response.status_code == 415
    assert "Accepted" in response.json()["detail"]


def test_non_raster_content_returns_422(api, tmp_path) -> None:
    fake = tmp_path / "fake.tif"
    fake.write_bytes(b"still not a raster")
    session_id = _new_session(api)

    response = _upload(api, session_id, "single", fake)
    assert response.status_code == 422
    assert "fake.tif" in response.json()["detail"]
    # Nothing was retained.
    assert api.get(f"/api/sessions/{session_id}").json()["images"] == {}


def test_empty_upload_returns_422(api, tmp_path) -> None:
    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"")
    session_id = _new_session(api)

    response = _upload(api, session_id, "single", empty)
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_invalid_role_is_rejected(api, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=48, height=48)
    session_id = _new_session(api)
    response = _upload(api, session_id, "not_a_role", scene.path)
    assert response.status_code == 422


def test_upload_to_unknown_session_returns_404(api, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=48, height=48)
    response = _upload(api, "0" * 32, "single", scene.path)
    assert response.status_code == 404


def test_malformed_session_id_in_path_is_rejected(api) -> None:
    """The route pattern rejects anything that is not 32 hex characters.

    A traversal attempt is refused either by the path pattern (422) or by route
    matching after the client decodes the escapes (404). Both are safe; what
    matters is that it never reaches the filesystem.
    """
    assert api.get("/api/sessions/not-a-valid-id").status_code == 422
    assert api.get("/api/sessions/" + "0" * 31).status_code == 422
    assert api.get("/api/sessions/..%2F..%2Fetc").status_code in (404, 422)
    assert api.get("/api/sessions/../../etc/passwd").status_code in (404, 422)


def test_thumbnail_is_served_as_png(api, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=128, height=128)
    session_id = _new_session(api)
    _upload(api, session_id, "single", scene.path)

    response = api.get(f"/api/sessions/{session_id}/thumb/single")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_thumbnail_for_empty_slot_returns_404(api) -> None:
    session_id = _new_session(api)
    assert api.get(f"/api/sessions/{session_id}/thumb/sar").status_code == 404


def test_delete_image_then_session(api, tmp_path) -> None:
    scene = make_optical_scene(tmp_path / "a.tif", width=48, height=48)
    session_id = _new_session(api)
    _upload(api, session_id, "date_a", scene.path)

    assert api.delete(f"/api/sessions/{session_id}/images/date_a").status_code == 200
    assert api.get(f"/api/sessions/{session_id}").json()["images"] == {}
    assert api.delete(f"/api/sessions/{session_id}").status_code == 200
    assert api.get(f"/api/sessions/{session_id}").status_code == 404
