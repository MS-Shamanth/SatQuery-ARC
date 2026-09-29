"""Disk-backed session storage.

One session holds the one, two, or three rasters that make up an input
configuration, plus their extracted metadata. State lives on disk rather than in
memory so that a backend restart mid-demo does not discard an upload.

Path handling is deliberately defensive: the session id is validated against a
UUID pattern and stored filenames are derived from the role enum, never from
user-supplied text. No part of a client-provided name reaches the filesystem.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.raster_io import RasterIngestError, build_thumbnail, read_metadata, sha256_of
from app.models.schemas import (
    ImageRole,
    IngestedImage,
    SessionRecord,
    derive_configuration,
)

logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# GeoTIFF/TIFF for geospatial imagery; PNG and JPEG are permitted only for the
# prescribed public benchmark datasets, per the problem statement.
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 400 * 1024 * 1024

# Windows file locking, typically from a OneDrive sync pass, can briefly deny the
# atomic replace of session.json. These control how long to keep trying.
SAVE_REPLACE_ATTEMPTS = 5
SAVE_RETRY_DELAY_SECONDS = 0.08


class SessionNotFound(KeyError):
    pass


class InvalidUpload(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def safe_extension(filename: str) -> str:
    """Return a validated lowercase extension, or raise.

    Only the extension is taken from the client's filename; it is checked
    against an allowlist and never joined to a path directly.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise InvalidUpload(
            f"Unsupported file type '{suffix or filename}'. Accepted: {allowed}."
        )
    return suffix


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------
    @staticmethod
    def validate_id(session_id: str) -> str:
        if not SESSION_ID_PATTERN.match(session_id or ""):
            raise SessionNotFound(f"Malformed session id: {session_id!r}")
        return session_id

    def session_dir(self, session_id: str) -> Path:
        return self.root / self.validate_id(session_id)

    def _record_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def image_path(self, session_id: str, role: ImageRole, extension: str) -> Path:
        return self.session_dir(session_id) / "images" / f"{role.value}{extension}"

    def thumbnail_path(self, session_id: str, role: ImageRole) -> Path:
        return self.session_dir(session_id) / "thumbs" / f"{role.value}.png"

    def find_image_path(self, session_id: str, role: ImageRole) -> Path | None:
        record = self.load(session_id)
        image = record.images.get(role)
        if image is None:
            return None
        candidate = self.session_dir(session_id) / "images" / image.stored_filename
        return candidate if candidate.exists() else None

    # -- lifecycle --------------------------------------------------------
    def create(self) -> SessionRecord:
        session_id = uuid.uuid4().hex
        directory = self.session_dir(session_id)
        (directory / "images").mkdir(parents=True, exist_ok=True)
        (directory / "thumbs").mkdir(parents=True, exist_ok=True)
        now = _now()
        record = SessionRecord(session_id=session_id, created_at=now, updated_at=now)
        self.save(record)
        logger.info("session created %s", session_id)
        return record

    def load(self, session_id: str) -> SessionRecord:
        path = self._record_path(session_id)
        if not path.exists():
            raise SessionNotFound(f"No such session: {session_id}")
        return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: SessionRecord) -> None:
        record.updated_at = _now()
        record.configuration = derive_configuration(set(record.images.keys()))
        path = self._record_path(record.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump_json(indent=2)

        # Write to a sibling then replace, so a crash cannot leave a truncated
        # session.json behind.
        #
        # On Windows the replace can fail with "Access is denied" when a sync
        # client such as OneDrive momentarily holds the target open. This project
        # lives inside a OneDrive folder, so that is a routine occurrence rather
        # than an exceptional one. Retry briefly, then fall back to writing in
        # place: losing atomicity is preferable to losing the session.
        temp = path.with_suffix(".json.tmp")
        temp.write_text(payload, encoding="utf-8")

        last_error: OSError | None = None
        for attempt in range(SAVE_REPLACE_ATTEMPTS):
            try:
                temp.replace(path)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(SAVE_RETRY_DELAY_SECONDS * (attempt + 1))

        logger.warning(
            "atomic replace of %s failed after %d attempts (%s); writing in place",
            path.name,
            SAVE_REPLACE_ATTEMPTS,
            last_error,
        )
        try:
            path.write_text(payload, encoding="utf-8")
        finally:
            temp.unlink(missing_ok=True)

    def delete(self, session_id: str) -> bool:
        directory = self.session_dir(session_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    def list_ids(self) -> list[str]:
        return sorted(
            child.name
            for child in self.root.iterdir()
            if child.is_dir() and SESSION_ID_PATTERN.match(child.name)
        )

    # -- ingest -----------------------------------------------------------
    def ingest(
        self,
        session_id: str,
        role: ImageRole,
        source: Path,
        original_filename: str,
        content_type: str | None = None,
        move: bool = True,
    ) -> IngestedImage:
        """Attach a raster to a session slot, extracting metadata and a preview.

        The file is read before being accepted: an unreadable raster raises and
        leaves no trace in the session, so a session never holds an entry whose
        metadata could not be derived.
        """
        started = time.perf_counter()
        record = self.load(session_id)
        extension = safe_extension(original_filename)

        destination = self.image_path(session_id, role, extension)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Clear any previous file for this slot, including a different extension.
        existing = record.images.get(role)
        if existing is not None:
            previous = destination.parent / existing.stored_filename
            if previous.exists() and previous != destination:
                previous.unlink(missing_ok=True)

        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))

        try:
            metadata = read_metadata(destination, original_filename=original_filename)
        except RasterIngestError:
            destination.unlink(missing_ok=True)
            raise

        recipe: str | None = None
        try:
            recipe = build_thumbnail(
                destination, metadata, self.thumbnail_path(session_id, role)
            )
        except Exception as exc:  # noqa: BLE001 - preview is not load-bearing
            logger.warning("thumbnail failed for %s/%s: %s", session_id, role.value, exc)
            metadata.ingest_notes.append(
                "Preview could not be rendered; analysis is unaffected."
            )

        image = IngestedImage(
            role=role,
            stored_filename=destination.name,
            original_filename=original_filename,
            content_type=content_type,
            sha256=sha256_of(destination),
            ingested_at=_now(),
            metadata=metadata,
            thumbnail_available=self.thumbnail_path(session_id, role).exists(),
            thumbnail_recipe=recipe,
            ingest_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )

        record.images[role] = image
        self.save(record)
        logger.info(
            "ingested %s/%s %s %dx%d %s in %.0f ms",
            session_id,
            role.value,
            metadata.driver,
            metadata.width,
            metadata.height,
            metadata.modality.modality.value,
            image.ingest_ms,
        )
        return image

    def remove_image(self, session_id: str, role: ImageRole) -> bool:
        record = self.load(session_id)
        image = record.images.pop(role, None)
        if image is None:
            return False
        (self.session_dir(session_id) / "images" / image.stored_filename).unlink(
            missing_ok=True
        )
        self.thumbnail_path(session_id, role).unlink(missing_ok=True)
        self.save(record)
        return True


_store: SessionStore | None = None


def get_store() -> SessionStore:
    """Process-wide session store, rooted at the configured sessions directory."""
    global _store
    if _store is None:
        from app.config import get_settings

        _store = SessionStore(get_settings().sessions_dir)
    return _store


def reset_store_for_tests(root: Path) -> SessionStore:
    """Point the store at a temporary directory. Test-only seam."""
    global _store
    _store = SessionStore(root)
    return _store
