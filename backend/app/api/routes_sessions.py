"""Session and image-ingest routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, UploadFile
from fastapi.responses import FileResponse

from app.core.raster_io import RasterIngestError
from app.core.sessions import (
    MAX_UPLOAD_BYTES,
    InvalidUpload,
    SessionNotFound,
    get_store,
    safe_extension,
)
from app.models.schemas import (
    ApiMessage,
    ImageRole,
    IngestedImage,
    SessionRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

SESSION_ID = PathParam(
    description="32-character hexadecimal session id", pattern="^[0-9a-f]{32}$"
)


@router.post("", response_model=SessionRecord, status_code=201)
async def create_session() -> SessionRecord:
    return get_store().create()


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str = SESSION_ID) -> SessionRecord:
    try:
        return get_store().load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{session_id}", response_model=ApiMessage)
async def delete_session(session_id: str = SESSION_ID) -> ApiMessage:
    removed = get_store().delete(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No such session: {session_id}")
    return ApiMessage(message=f"Session {session_id} deleted")


@router.post("/{session_id}/images", response_model=IngestedImage, status_code=201)
async def upload_image(
    session_id: str = SESSION_ID,
    role: ImageRole = Form(description="Which input slot this image occupies"),
    file: UploadFile = File(description="GeoTIFF or TIFF; PNG/JPEG for benchmarks"),
) -> IngestedImage:
    """Ingest one raster into a session slot.

    The upload is streamed to a temporary file with a hard size cap, then read by
    rasterio. If it cannot be interpreted as a raster the request fails with 422
    and nothing is retained.
    """
    store = get_store()

    try:
        store.load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    original_filename = file.filename or "upload"
    try:
        extension = safe_extension(original_filename)
    except InvalidUpload as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            written = 0
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"'{original_filename}' exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                        ),
                    )
                temp_handle.write(chunk)

        if written == 0:
            raise HTTPException(
                status_code=422, detail=f"'{original_filename}' is empty."
            )

        return store.ingest(
            session_id=session_id,
            role=role,
            source=temp_path,
            original_filename=original_filename,
            content_type=file.content_type,
            move=True,
        )
    except RasterIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidUpload as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.delete("/{session_id}/images/{role}", response_model=ApiMessage)
async def remove_image(role: ImageRole, session_id: str = SESSION_ID) -> ApiMessage:
    try:
        removed = get_store().remove_image(session_id, role)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"No image in slot '{role.value}'"
        )
    return ApiMessage(message=f"Removed image from slot '{role.value}'")


@router.get("/{session_id}/thumb/{role}", response_class=FileResponse)
async def get_thumbnail(role: ImageRole, session_id: str = SESSION_ID) -> FileResponse:
    """Serve the rendered preview PNG for one slot."""
    store = get_store()
    try:
        store.load(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = store.thumbnail_path(session_id, role)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"No preview available for slot '{role.value}'"
        )
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
