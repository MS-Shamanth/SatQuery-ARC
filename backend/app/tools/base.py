"""The tool protocol.

Every specialist is a Tool. A Tool declares what it needs, decides for itself
whether it can run on a given input, and returns measurements that each carry the
formula and the inputs that produced them.

Two layers are kept apart deliberately:

* ``ToolOutcome`` and ``MaskLayer`` hold numpy arrays and live only in memory
  during a run.
* ``ToolRun`` and ``MaskSummary`` are the pydantic records that reach the API,
  the execution trace, and the evidence packet.

Nothing that reaches the client carries a raw array, and nothing that reaches the
client carries a number without provenance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from app.core.sessions import SessionStore
from app.models.schemas import (
    BandRole,
    ImageRole,
    InputConfiguration,
    MaskSummary,
    Measurement,
    Modality,
    RasterMetadata,
    ReadinessReport,
    SessionRecord,
    ToolDescriptor,
    ToolImplementation,
    ToolRequirement,
    ToolRun,
)

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """Raised when a tool cannot complete for a reason worth reporting."""


@dataclass
class MaskLayer:
    """A boolean raster a tool produced, with the geometry to place it."""

    key: str
    label: str
    array: np.ndarray
    transform: Affine
    crs: CRS | None
    description: str = ""
    applies_to: list[ImageRole] = field(default_factory=list)
    threshold: float | None = None
    threshold_method: str | None = None
    separability: float | None = None
    # Pixels that were unusable (nodata, cloud) and so are neither in nor out.
    invalid: np.ndarray | None = None
    stored_filename: str | None = None

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.array))

    @property
    def valid_pixels(self) -> int:
        if self.invalid is None:
            return int(self.array.size)
        return int(self.array.size - np.count_nonzero(self.invalid))

    def pixel_area_m2(self) -> float | None:
        if self.crs is None:
            return None
        return abs(self.transform.a) * abs(self.transform.e)

    def area_km2(self) -> float | None:
        area = self.pixel_area_m2()
        if area is None:
            return None
        return self.pixel_count * area / 1_000_000.0

    def bounds(self) -> tuple[float, float, float, float]:
        height, width = self.array.shape
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * width
        bottom = top + self.transform.e * height
        return (left, min(bottom, top), right, max(bottom, top))

    def bounds_wgs84(self) -> list[float] | None:
        if self.crs is None:
            return None
        try:
            from rasterio.warp import transform_bounds

            return list(transform_bounds(self.crs, "EPSG:4326", *self.bounds()))
        except Exception:  # noqa: BLE001
            return None

    def summary(self) -> MaskSummary:
        valid = self.valid_pixels
        epsg = None
        if self.crs is not None:
            try:
                epsg = self.crs.to_epsg()
            except Exception:  # noqa: BLE001
                epsg = None
        return MaskSummary(
            key=self.key,
            label=self.label,
            description=self.description,
            pixel_count=self.pixel_count,
            total_pixels=valid,
            coverage_fraction=(self.pixel_count / valid) if valid else 0.0,
            area_km2=self.area_km2(),
            epsg=epsg,
            bounds_wgs84=self.bounds_wgs84(),
            stored_filename=self.stored_filename,
            applies_to=list(self.applies_to),
            threshold=self.threshold,
            threshold_method=self.threshold_method,
            separability=self.separability,
        )


@dataclass
class ToolOutcome:
    """In-memory result of one tool execution."""

    tool: str
    version: str
    implementation: ToolImplementation
    ok: bool = True
    skipped_reason: str | None = None
    measurements: list[Measurement] = field(default_factory=list)
    masks: list[MaskLayer] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    # Free-form values later stages consume without them being user-facing
    # measurements, for example per-index arrays kept for the debate stage.
    artifacts: dict[str, Any] = field(default_factory=dict)

    def mask(self, key: str) -> MaskLayer | None:
        for layer in self.masks:
            if layer.key == key:
                return layer
        return None

    def measurement(self, key: str) -> Measurement | None:
        for item in self.measurements:
            if item.key == key:
                return item
        return None

    def to_record(self) -> ToolRun:
        """The serialisable record for the trace and the evidence packet."""
        return ToolRun(
            tool=self.tool,
            version=self.version,
            implementation=self.implementation,
            ok=self.ok,
            skipped_reason=self.skipped_reason,
            parameters=self.parameters,
            measurements=list(self.measurements),
            masks=[layer.summary() for layer in self.masks],
            notes=list(self.notes),
            duration_ms=self.duration_ms,
        )

    @classmethod
    def skipped(
        cls, tool: Tool, reason: str, parameters: dict[str, Any] | None = None
    ) -> ToolOutcome:
        """A tool that could not run says why, and that reaches the UI.

        An unavailable input showing as "skipped, band B11 absent" is more
        informative than a silently missing row.
        """
        return cls(
            tool=tool.name,
            version=tool.version,
            implementation=tool.implementation,
            ok=False,
            skipped_reason=reason,
            parameters=parameters or {},
        )


@dataclass
class ToolContext:
    """Everything a tool is allowed to look at."""

    session: SessionRecord
    store: SessionStore
    readiness: ReadinessReport | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    target_classes: list[str] = field(default_factory=list)
    # The approved contract, when a tool's job depends on what was claimed. The
    # verdict engine needs the claim and the direction it asserts; measuring tools
    # do not and should not look at it.
    contract: Any = None
    # Outcomes of tools that already ran in this execution, keyed by tool name.
    # A tool that builds on another's mask reads it from here rather than
    # recomputing it, which keeps one definition of each derived layer.
    upstream: dict[str, ToolOutcome] = field(default_factory=dict)

    def upstream_mask(self, tool: str, key: str) -> MaskLayer | None:
        """A mask an earlier tool produced in this same run, if it exists."""
        outcome = self.upstream.get(tool)
        return outcome.mask(key) if outcome is not None else None

    # -- input access ------------------------------------------------------
    @property
    def configuration(self) -> InputConfiguration:
        return self.session.configuration

    @property
    def roles(self) -> list[ImageRole]:
        return [image.role for image in self.session.image_list()]

    def metadata(self, role: ImageRole) -> RasterMetadata:
        image = self.session.images.get(role)
        if image is None:
            raise ToolError(f"No image in slot '{role.value}'.")
        return image.metadata

    def path(self, role: ImageRole) -> Path:
        resolved = self.store.find_image_path(self.session.session_id, role)
        if resolved is None:
            raise ToolError(f"Image file for slot '{role.value}' is missing on disk.")
        return resolved

    def primary_role(self) -> ImageRole:
        """The slot a single-image operation should act on.

        For a cross-modal pair the optical image is primary, because spectral
        indices are defined on reflectance and not on backscatter.
        """
        for candidate in (
            ImageRole.SINGLE,
            ImageRole.OPTICAL,
            ImageRole.DATE_A,
            ImageRole.SAR,
            ImageRole.DATE_B,
        ):
            if candidate in self.session.images:
                return candidate
        raise ToolError("The session holds no imagery.")

    def optical_roles(self) -> list[ImageRole]:
        return [
            role
            for role in self.roles
            if self.metadata(role).modality.modality is Modality.OPTICAL
        ]

    def sar_roles(self) -> list[ImageRole]:
        return [
            role
            for role in self.roles
            if self.metadata(role).modality.modality is Modality.SAR
        ]

    def temporal_roles(self) -> tuple[ImageRole, ImageRole] | None:
        if {ImageRole.DATE_A, ImageRole.DATE_B} <= set(self.session.images):
            return (ImageRole.DATE_A, ImageRole.DATE_B)
        return None

    def param(self, key: str, default: Any = None) -> Any:
        return self.parameters.get(key, default)


@runtime_checkable
class Tool(Protocol):
    """A specialist the orchestrator can select and run."""

    name: str
    version: str
    implementation: ToolImplementation
    summary: str

    def requirement(self) -> ToolRequirement: ...

    def depends_on(self) -> tuple[str, ...]: ...

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]: ...

    def parameter_spec(self) -> dict[str, Any]: ...

    def produces(self) -> list[str]: ...

    def can_run(self, context: ToolContext) -> tuple[bool, str]: ...

    def run(self, context: ToolContext) -> ToolOutcome: ...


class BaseTool:
    """Shared plumbing: requirement checking, timing, and error capture."""

    name: str = "unnamed-tool"
    version: str = "0.0.0"
    implementation: ToolImplementation = ToolImplementation.DETERMINISTIC
    summary: str = ""

    def requirement(self) -> ToolRequirement:
        return ToolRequirement()

    def depends_on(self) -> tuple[str, ...]:
        """Tools whose output this one reads.

        Declared rather than discovered because ``can_run`` is asked two different
        questions. At execution time it must refuse when the upstream output is
        absent, or the tool would fail on a missing key. At planning time that same
        refusal would be wrong: the dependency has not run *yet*, and the planner's
        job is to schedule both in order. Naming the dependency lets the planner
        tell those two situations apart.
        """
        return ()

    def parameter_spec(self) -> dict[str, Any]:
        return {}

    def produces(self) -> list[str]:
        return []

    def descriptor(self, registered: bool = True) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name,
            version=self.version,
            implementation=self.implementation,
            summary=self.summary,
            requirement=self.requirement(),
            parameters=self.parameter_spec(),
            produces=self.produces(),
            registered=registered,
        )

    # -- requirement checking ---------------------------------------------
    def can_run(self, context: ToolContext) -> tuple[bool, str]:
        """Decide whether this tool can act on the given input.

        The returned reason is shown to the user when the answer is no, so it
        names the specific band or configuration that is missing.
        """
        requirement = self.requirement()

        if requirement.configurations and (
            context.configuration not in requirement.configurations
        ):
            allowed = ", ".join(c.value for c in requirement.configurations)
            return False, (
                f"needs a {allowed} input, but this session is "
                f"{context.configuration.value}"
            )

        candidates = self._candidate_roles(context)
        if not candidates:
            wanted = ", ".join(m.value for m in requirement.modalities) or "any"
            return False, f"no {wanted} image is present in this session"

        for role in candidates:
            meta = context.metadata(role)

            if requirement.requires_crs and meta.geo.epsg is None:
                return False, (
                    f"'{role.value}' has no coordinate reference system, so "
                    "geospatial measurement is not possible"
                )

            missing = meta.missing_roles(*requirement.band_roles)
            if missing:
                names = ", ".join(role_.value for role_ in missing)
                return False, f"'{role.value}' is missing band(s) {names}"

            if requirement.any_of_band_roles:
                satisfied = any(
                    meta.has_roles(*group) for group in requirement.any_of_band_roles
                )
                if not satisfied:
                    options = " or ".join(
                        "+".join(r.value for r in group)
                        for group in requirement.any_of_band_roles
                    )
                    return False, f"'{role.value}' provides none of {options}"

        return True, "requirements satisfied"

    def upstream_ready(self, context: ToolContext) -> tuple[bool, str]:
        """Whether the outputs this tool reads are present yet.

        Deliberately separate from ``can_run``. The two answer different
        questions, and conflating them broke the planner: ``can_run`` asks whether
        this *input* can support the tool at all, which the planner needs before
        anything has run, while this asks whether the run has reached the point
        where the tool has something to work on. A tool whose dependency has not
        executed is not unavailable, it is merely early.
        """
        return True, ""

    def _candidate_roles(self, context: ToolContext) -> list[ImageRole]:
        """Slots this tool would act on, filtered by the modalities it needs."""
        requirement = self.requirement()
        if not requirement.modalities:
            return context.roles
        wanted = set(requirement.modalities)
        return [
            role
            for role in context.roles
            if context.metadata(role).modality.modality in wanted
        ]

    # -- execution --------------------------------------------------------
    def run(self, context: ToolContext) -> ToolOutcome:
        started = time.perf_counter()
        allowed, reason = self.can_run(context)
        if allowed:
            allowed, reason = self.upstream_ready(context)
        if not allowed:
            outcome = ToolOutcome.skipped(self, reason, dict(context.parameters))
            outcome.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info("tool %s skipped: %s", self.name, reason)
            return outcome

        try:
            outcome = self.execute(context)
        except ToolError as exc:
            outcome = ToolOutcome.skipped(self, str(exc), dict(context.parameters))
            logger.warning("tool %s failed: %s", self.name, exc)
        except Exception as exc:  # noqa: BLE001 - one tool must not abort the run
            outcome = ToolOutcome.skipped(
                self, f"unexpected failure: {exc}", dict(context.parameters)
            )
            logger.exception("tool %s raised", self.name)

        outcome.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "tool %s %s in %.1f ms (%d measurements, %d masks)",
            self.name,
            "ok" if outcome.ok else "skipped",
            outcome.duration_ms,
            len(outcome.measurements),
            len(outcome.masks),
        )
        return outcome

    def execute(self, context: ToolContext) -> ToolOutcome:  # pragma: no cover
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def measurement(
        self,
        key: str,
        label: str,
        value: float,
        unit: str,
        formula: str,
        inputs: dict[str, Any],
        *,
        method: str | None = None,
        applies_to: list[ImageRole] | None = None,
        precision: int = 2,
    ) -> Measurement:
        return Measurement(
            key=key,
            label=label,
            value=float(value),
            unit=unit,
            formula=formula,
            inputs=inputs,
            source_tool=self.name,
            source_version=self.version,
            method=method,
            applies_to=applies_to or [],
            precision=precision,
        )

    def outcome(self, **kwargs: Any) -> ToolOutcome:
        return ToolOutcome(
            tool=self.name,
            version=self.version,
            implementation=self.implementation,
            **kwargs,
        )


__all__ = [
    "BaseTool",
    "MaskLayer",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolOutcome",
]
