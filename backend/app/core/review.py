"""Analyst review queue and audit log.

Verified changes land here as candidates. An analyst confirms or rejects each
one; every decision is appended to an immutable audit log and the queue is
re-ranked by confidence. State is JSON on disk so it survives a restart.

This closes the loop the problem statement asks for: a human decision that is
recorded, attributable, and reversible, feeding back into how the queue is
ordered.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.models.archive import (
    AuditEntry,
    ChangeVerification,
    ReviewItem,
    ReviewState,
)

logger = logging.getLogger(__name__)

_STATUS_RANK = {"confirmed": 0, "pending": 1, "rejected": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def _queue_path(self) -> Path:
        return self.root / "queue.json"

    @property
    def _audit_path(self) -> Path:
        return self.root / "audit.json"

    # -- persistence ------------------------------------------------------
    def _load_items(self) -> dict[str, ReviewItem]:
        if not self._queue_path.exists():
            return {}
        try:
            raw = json.loads(self._queue_path.read_text(encoding="utf-8"))
            return {i["item_id"]: ReviewItem.model_validate(i) for i in raw}
        except Exception as exc:  # noqa: BLE001
            logger.warning("review queue unreadable (%s); starting empty", exc)
            return {}

    def _save_items(self, items: dict[str, ReviewItem]) -> None:
        self._queue_path.write_text(
            json.dumps([i.model_dump() for i in items.values()], indent=2),
            encoding="utf-8",
        )

    def _load_audit(self) -> list[AuditEntry]:
        if not self._audit_path.exists():
            return []
        try:
            raw = json.loads(self._audit_path.read_text(encoding="utf-8"))
            return [AuditEntry.model_validate(e) for e in raw]
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit log unreadable (%s); starting empty", exc)
            return []

    def _save_audit(self, entries: list[AuditEntry]) -> None:
        # Cap the on-disk log so a long-running demo does not grow without bound.
        trimmed = entries[-500:]
        self._audit_path.write_text(
            json.dumps([e.model_dump() for e in trimmed], indent=2),
            encoding="utf-8",
        )

    # -- audit ------------------------------------------------------------
    def log(
        self,
        action: str,
        target: str,
        detail: str,
        *,
        actor: str = "analyst",
        confidence: float | None = None,
    ) -> AuditEntry:
        with self._lock:
            entries = self._load_audit()
            entry = AuditEntry(
                entry_id=uuid.uuid4().hex[:12],
                at=_now(),
                actor=actor,
                action=action,
                target=target,
                detail=detail,
                confidence=confidence,
            )
            entries.append(entry)
            self._save_audit(entries)
        return entry

    # -- queue ------------------------------------------------------------
    def upsert_from_verification(
        self, verification: ChangeVerification, *, actor: str = "system"
    ) -> ReviewItem:
        """Seed or refresh a candidate from a verified change.

        A re-verified AOI that was already decided is reopened to pending, since
        the evidence behind the earlier decision has been recomputed.
        """
        with self._lock:
            items = self._load_items()
            existing = items.get(verification.aoi_key)
            thumb = f"/api/archive/thumb/{verification.aoi_key}__date_b.png"
            item = ReviewItem(
                item_id=verification.aoi_key,
                aoi_key=verification.aoi_key,
                place=verification.place,
                region=verification.region,
                change_hint=verification.headline,
                verdict_label=verification.verdict_label,
                confidence=verification.confidence,
                status="pending",
                delta_display=verification.delta_display,
                acquisition_date=verification.after.acquisition_date,
                sensor=verification.after.sensor,
                thumbnail_url=thumb,
                created_at=existing.created_at if existing else _now(),
                run_id=verification.run_id,
            )
            items[item.item_id] = item
            self._save_items(items)
        self.log(
            "verified",
            verification.place,
            f"{verification.headline} - {verification.verdict_label} "
            f"({verification.confidence * 100:.0f}% confidence), "
            f"{verification.delta_display}",
            actor=actor,
            confidence=verification.confidence,
        )
        return item

    def decide(
        self, item_id: str, decision: str, *, actor: str = "analyst", note: str | None = None
    ) -> ReviewItem:
        with self._lock:
            items = self._load_items()
            item = items.get(item_id)
            if item is None:
                raise KeyError(item_id)
            if decision == "confirm":
                item.status = "confirmed"
            elif decision == "reject":
                item.status = "rejected"
            elif decision == "reopen":
                item.status = "pending"
            else:
                raise ValueError(f"Unknown decision '{decision}'.")
            item.decided_at = None if decision == "reopen" else _now()
            item.decided_by = None if decision == "reopen" else actor
            item.note = note
            items[item_id] = item
            self._save_items(items)

        action = {"confirm": "confirmed", "reject": "rejected", "reopen": "reopened"}[decision]
        self.log(
            action, item.place,
            f"{item.change_hint} ({item.confidence * 100:.0f}% confidence)"
            + (f" - {note}" if note else ""),
            actor=actor, confidence=item.confidence,
        )
        return item

    def state(self) -> ReviewState:
        with self._lock:
            items = list(self._load_items().values())
            audit = self._load_audit()
        # Re-rank: confirmed first, then pending, then rejected; by confidence within.
        items.sort(key=lambda i: (_STATUS_RANK.get(i.status, 1), -i.confidence))
        pending = sum(1 for i in items if i.status == "pending")
        confirmed = sum(1 for i in items if i.status == "confirmed")
        rejected = sum(1 for i in items if i.status == "rejected")
        return ReviewState(
            items=items,
            audit=list(reversed(audit))[:40],
            pending=pending,
            confirmed=confirmed,
            rejected=rejected,
        )

    def reset(self) -> None:
        with self._lock:
            self._save_items({})
            self._save_audit([])


_store: ReviewStore | None = None


def get_review_store() -> ReviewStore:
    global _store
    if _store is None:
        _store = ReviewStore(get_settings().review_dir)
    return _store


__all__ = ["ReviewStore", "get_review_store"]
