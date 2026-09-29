"""The tool registry.

The registry is the list of things the system can actually do. Two consequences
follow, and both matter:

* The Analysis Contract is validated against it. If the language model names a
  tool that is not registered, the contract is rejected rather than executed, so
  a hallucinated capability can never appear in an execution trace.
* Registration is conditional. A tool whose prerequisites are absent, such as the
  land-cover probe with no trained weights on disk, is simply not registered, and
  the orchestrator degrades without special-casing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.models.schemas import ToolDescriptor
from app.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ToolNotRegistered(KeyError):
    """Raised when a name does not correspond to a registered tool."""


@dataclass
class Availability:
    """Whether a registered tool can run on a particular input, and why not."""

    tool_name: str
    available: bool
    reason: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._declined: dict[str, str] = {}

    # -- registration -----------------------------------------------------
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if tool.name in self._tools and not replace:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        self._declined.pop(tool.name, None)
        logger.info(
            "registered tool %s v%s (%s)",
            tool.name,
            tool.version,
            tool.implementation.value,
        )
        return tool

    def register_if(
        self, condition: bool, tool: Tool, *, reason: str = "prerequisite not met"
    ) -> Tool | None:
        """Register a tool only when its prerequisites hold.

        A declined tool is remembered so the UI can explain its absence instead
        of the capability silently not existing.
        """
        if condition:
            return self.register(tool)
        self.record_declined(tool.name, reason=reason)
        return None

    def record_declined(self, name: str, *, reason: str) -> None:
        """Note a capability that is absent, without needing an instance of it.

        ``register_if`` needs a constructed tool, and some tools cannot be built at
        all without the thing they are missing: the learned probe needs its fitted
        weights to exist before it can be instantiated. An absence still has to be
        explainable, so it is recorded directly.
        """
        self._declined[name] = reason
        logger.info("tool %s not registered: %s", name, reason)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear(self) -> None:
        self._tools.clear()
        self._declined.clear()

    # -- lookup -----------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def declined(self) -> dict[str, str]:
        return dict(self._declined)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotRegistered(
                f"'{name}' is not a registered tool. Registered: "
                f"{', '.join(self.names()) or 'none'}."
            ) from exc

    def resolve(self, names: list[str]) -> tuple[list[Tool], list[str]]:
        """Split requested names into resolved tools and unknown names.

        Used by the contract validator, which must reject unknown names rather
        than skipping them quietly.
        """
        resolved: list[Tool] = []
        unknown: list[str] = []
        for name in names:
            if name in self._tools:
                resolved.append(self._tools[name])
            else:
                unknown.append(name)
        return resolved, unknown

    # -- description ------------------------------------------------------
    def describe(self) -> list[ToolDescriptor]:
        """Descriptors for every registered tool, plus any that declined."""
        descriptors = [
            tool.descriptor()  # type: ignore[attr-defined]
            for tool in (self._tools[name] for name in self.names())
        ]
        return descriptors

    def availability(self, context: ToolContext) -> list[Availability]:
        """Which registered tools can run on this specific input."""
        results: list[Availability] = []
        for name in self.names():
            tool = self._tools[name]
            try:
                ok, reason = tool.can_run(context)
            except Exception as exc:  # noqa: BLE001 - a broken tool is reported
                ok, reason = False, f"requirement check failed: {exc}"
            results.append(Availability(tool_name=name, available=ok, reason=reason))
        return results

    def available_names(self, context: ToolContext) -> list[str]:
        return [item.tool_name for item in self.availability(context) if item.available]


def build_default_registry() -> ToolRegistry:
    """Assemble the registry from the tools that ship with the system.

    Imports are local so that adding a tool with a heavy optional dependency
    cannot break registry construction for everything else.
    """
    from app.tools.change import ChangeCvaEngine
    from app.tools.confounders import ConfounderEngine
    from app.tools.disagreement import DisagreementEngine
    from app.tools.fusion import OpticalSarFusion
    from app.tools.gis import GisMeasureEngine
    from app.tools.grounding import GroundingEngine
    from app.tools.indices import SpectralIndexEngine
    from app.tools.sar import SarBackscatterEngine
    from app.tools.verdict import VerdictEngine

    registry = ToolRegistry()
    registry.register(SpectralIndexEngine())
    registry.register(GroundingEngine())
    registry.register(SarBackscatterEngine())
    # Reads both sensors' results, so it is registered after both producers.
    registry.register(OpticalSarFusion())
    registry.register(ChangeCvaEngine())

    # The one learned tool, and the only optional one. It registers only if weights
    # have been fitted here, and declines with a reason if not, which the capability
    # panel shows. Nothing else depends on it: every scripted demonstration runs
    # identically whether it is present or absent, which is the condition for
    # adding a learned component to a system whose promise is that its numbers are
    # traceable.
    _register_probe(registry)

    # Compares what every method above concluded, so it goes after all of them.
    registry.register(DisagreementEngine())

    # Order matters for anything that reads another tool's output: the confounder
    # engine tests the change engine's finding, so it is planned after it.
    registry.register(ConfounderEngine())
    registry.register(GisMeasureEngine())
    # Last, and deliberately so: it weighs what everything else produced.
    registry.register(VerdictEngine())
    return registry


def _register_probe(registry: ToolRegistry) -> None:
    """Add the learned probe when it has been trained, and say so when it has not."""
    from app.config import get_settings
    from app.core.adapter import WEIGHTS_FILENAME, load_weights
    from app.tools.probe import LandCoverProbe

    try:
        weights = load_weights(get_settings().models_dir)
    except Exception as exc:  # noqa: BLE001 - an optional model cannot break startup
        logger.warning("could not load the land-cover probe: %s", exc)
        weights = None

    if weights is None:
        registry.record_declined(
            "rs-landcover-probe",
            reason=(
                "no fitted weights are present. Run scripts/train_probe.py to fit "
                f"the probe; it writes data/models/{WEIGHTS_FILENAME}. Until then "
                "every measurement in this build comes from a published formula."
            ),
        )
        return

    registry.register(LandCoverProbe(weights))


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """The process-wide registry."""
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def reset_registry_for_tests(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Replace the process-wide registry. Test-only seam."""
    global _registry
    _registry = registry if registry is not None else build_default_registry()
    return _registry


__all__ = [
    "Availability",
    "ToolNotRegistered",
    "ToolRegistry",
    "build_default_registry",
    "get_registry",
    "reset_registry_for_tests",
]
