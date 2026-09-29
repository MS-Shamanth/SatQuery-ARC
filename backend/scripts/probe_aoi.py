"""Choose a demo AOI by measurement rather than by eye.

The first `urban_growth` centre was picked from a map and it measured a *decrease*
in built-up area, because that AOI is dry farmland where NDBI tracks field state
rather than construction. The scene name promised growth and the system, correctly,
refused to agree. Guessing again would be the same mistake with different
coordinates.

So this probes candidates with the same change engine the demo will run, on the
same imagery the demo will load, and reports the figures that decide whether a
claim of increase can be supported:

* the measured direction, which has to match the claim,
* `threshold_sensitivity`, which is what confidence now rests on: if nudging the
  class boundary by a twentieth of the index range moves the answer, the answer
  is about the threshold and not about the ground,
* whether a second index corroborates the first,
* and the seasonal offset, since two dates in different parts of the growing
  season make the seasonality confounder survive and cap the verdict regardless
  of how clean the measurement is.

    .\\.venv\\Scripts\\python.exe scripts\\probe_aoi.py            # every candidate
    .\\.venv\\Scripts\\python.exe scripts\\probe_aoi.py noida      # one of them

Nothing here writes to the sample library. A winning candidate is promoted by
editing the preset in app/core/samples.py and re-running fetch_samples.py.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.readiness import evaluate_readiness  # noqa: E402
from app.core.samples import OpticalSpec, PresetDef  # noqa: E402
from app.core.sessions import get_store  # noqa: E402
from app.models.confounders import ConfounderVerdict  # noqa: E402
from app.models.contract import (  # noqa: E402
    AnalysisContract,
    ChangeDirection,
    ConfounderKind,
    ConfounderPlan,
    ContractSource,
    ContractTaskType,
    ToolPlan,
)
from app.models.schemas import ImageRole, InputConfiguration  # noqa: E402
from app.tools.base import ToolContext  # noqa: E402
from app.tools.change import ChangeCvaEngine  # noqa: E402
from app.tools.confounders import ConfounderEngine  # noqa: E402
from app.tools.verdict import VerdictEngine  # noqa: E402

from fetch_samples import fetch_preset  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")


@dataclass(frozen=True)
class Candidate:
    """One place and one pair of date windows to measure."""

    key: str
    place: str
    lon: float
    lat: float
    # Why this location is worth measuring. Recorded so a rejected candidate still
    # tells the next reader something.
    rationale: str
    before: str
    after: str
    # The index whose change is being claimed, and the class it measures.
    index: str = "NDBI"
    target: str = "built-up"
    # Which direction the demo's claim asserts.
    want: str = "increase"
    max_cloud: float = 8.0


# Both windows sit in the same part of the dry season on purpose. A built-up claim
# compared across seasons cannot be settled: the seasonality confounder survives
# and caps the verdict no matter how clean the measurement is. Holding the calendar
# fixed is the only way an increase in built-up area can be attributed to building.
DRY_BEFORE = "2019-12-01/2020-02-28"
DRY_AFTER = "2025-12-01/2026-02-28"

CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="noida_ext",
        place="Greater Noida West, Uttar Pradesh",
        lon=77.438,
        lat=28.601,
        rationale=(
            "High-rise residential build-out on former farmland, and the after "
            "state is finished construction rather than cleared ground."
        ),
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
    Candidate(
        key="rajarhat",
        place="New Town Rajarhat, North 24 Parganas, West Bengal",
        lon=88.470,
        lat=22.600,
        rationale="Planned township expanding into wetland and farmland.",
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
    Candidate(
        key="ulwe",
        place="Ulwe and the Navi Mumbai airport corridor, Maharashtra",
        lon=73.028,
        lat=18.993,
        rationale=(
            "Airport earthworks plus dense residential towers; strong built-up "
            "signal against a coastal backdrop."
        ),
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
    Candidate(
        key="kokapet",
        place="Kokapet and the Financial District, Hyderabad, Telangana",
        lon=78.345,
        lat=17.400,
        rationale=(
            "Commercial towers on rocky scrub. Bare rock is the confusing case "
            "for NDBI, so this candidate tests the index's weakness directly."
        ),
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
    Candidate(
        key="devanahalli",
        place="Devanahalli, Bengaluru airport corridor, Karnataka",
        lon=77.706,
        lat=13.222,
        rationale="Airport-corridor build-out over plantation and farmland.",
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
    Candidate(
        key="mohali",
        place="New Chandigarh, Sahibzada Ajit Singh Nagar, Punjab",
        lon=76.690,
        lat=30.760,
        rationale="Sector-grid expansion; the same dry season each year.",
        before=DRY_BEFORE,
        after=DRY_AFTER,
    ),
)

# Reservoir drawdown, measured on MNDWI rather than NDBI.
#
# The built-up candidates above all measured NDBI separability at or near zero:
# at 10 m in these landscapes NDBI cannot tell new concrete from dry farmland, so
# a supported verdict on a built-up claim would be a number whose map contradicts
# it. Open water is the opposite case. It is specular in SWIR and MNDWI splits it
# from everything else cleanly, which is why the same index measured the Ukai
# reservoir at a separability of 0.87.
#
# Both dates are taken from the same weeks before the monsoon, when storage is at
# its lowest and inter-annual differences are real rather than seasonal. Holding
# the calendar fixed is what lets the seasonality test rule phenology out instead
# of merely flagging it.
#
# These were first written expecting a decrease, on the assumption that a six-year
# gap would show drawdown. All four measured an increase instead: pre-monsoon
# storage in 2025 was higher than in 2019 at every one of them. The claim direction
# here follows the measurement rather than the expectation, which is the same rule
# the rest of the system runs on.
PRE_MONSOON_BEFORE = "2019-05-01/2019-06-10"
PRE_MONSOON_AFTER = "2025-05-01/2025-06-10"

WATER_CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="ukai_drawdown",
        place="Ukai reservoir, Tapi basin, Gujarat",
        lon=73.586,
        lat=21.245,
        rationale=(
            "Already proven ground for this index: MNDWI separated this reservoir "
            "at 0.87, and the Tapi basin draws down hard before the monsoon."
        ),
        before=PRE_MONSOON_BEFORE,
        after=PRE_MONSOON_AFTER,
        index="MNDWI",
        target="water",
        want="increase",
    ),
    Candidate(
        key="nagarjuna",
        place="Nagarjuna Sagar, Krishna basin, Telangana and Andhra Pradesh",
        lon=79.312,
        lat=16.575,
        rationale="Large storage with severe inter-annual swings in the Krishna basin.",
        before=PRE_MONSOON_BEFORE,
        after=PRE_MONSOON_AFTER,
        index="MNDWI",
        target="water",
        want="increase",
    ),
    Candidate(
        key="tungabhadra",
        place="Tungabhadra reservoir, Koppal, Karnataka",
        lon=76.335,
        lat=15.268,
        rationale="Shallow gradient shoreline, so a level change moves a lot of area.",
        before=PRE_MONSOON_BEFORE,
        after=PRE_MONSOON_AFTER,
        index="MNDWI",
        target="water",
        want="increase",
    ),
    Candidate(
        key="hirakud",
        place="Hirakud reservoir, Mahanadi basin, Odisha",
        lon=83.870,
        lat=21.530,
        rationale="Very large surface area with a wide seasonal band exposed.",
        before=PRE_MONSOON_BEFORE,
        after=PRE_MONSOON_AFTER,
        index="MNDWI",
        target="water",
        want="increase",
    ),
)

# Earlier after-windows, because the first attempt at this demo compared a 2019
# Sentinel-2A scene against a 2025 Sentinel-2C one and the radiometry test called
# that the likely explanation: surfaces that should not have changed differed by
# -0.066 reflectance, which is larger than the change being claimed. The test was
# right, so the fix is to compare acquisitions from the same generation of the
# constellation rather than to weaken the test.
YEAR_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("2019", "2019-05-01/2019-06-10", "2023-05-01/2023-06-10"),
    ("2020", "2020-05-01/2020-06-10", "2024-05-01/2024-06-10"),
    ("2021", "2021-05-01/2021-06-10", "2024-05-01/2024-06-10"),
)

WATER_VARIANTS: tuple[Candidate, ...] = tuple(
    Candidate(
        key=f"{base.key}_{tag}",
        place=base.place,
        lon=base.lon,
        lat=base.lat,
        rationale=base.rationale,
        before=before,
        after=after,
        index=base.index,
        target=base.target,
        want=base.want,
    )
    for base in WATER_CANDIDATES
    if base.key in {"tungabhadra", "hirakud"}
    for tag, before, after in YEAR_PAIRS
)

CANDIDATES = CANDIDATES + WATER_CANDIDATES + WATER_VARIANTS
CANDIDATES_BY_KEY = {candidate.key: candidate for candidate in CANDIDATES}

# What a candidate has to do to carry DEMO 2. The direction has to match the claim
# and the answer has to survive the threshold moving; everything else is reported
# but not required.
WANT_DIRECTION = "increase"
MAX_THRESHOLD_SENSITIVITY = 0.35
# The index has to actually divide the scene into two populations. Matching the
# engine's own WEAK_SEPARABILITY so the probe and the run agree on what counts.
MIN_SEPARABILITY = 0.35


def as_preset(candidate: Candidate) -> PresetDef:
    """A throwaway preset, so the probe fetches exactly as the library does."""
    return PresetDef(
        key=f"probe_{candidate.key}",
        title=candidate.place,
        description=candidate.rationale,
        demo="AOI probe",
        place=candidate.place,
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        lon=candidate.lon,
        lat=candidate.lat,
        suggested_queries=(
            "Has the built-up area increased, decreased, or remained unchanged?",
        ),
        optical={
            ImageRole.DATE_A: OpticalSpec(candidate.before, candidate.max_cloud),
            ImageRole.DATE_B: OpticalSpec(candidate.after, candidate.max_cloud),
        },
        grid_role=ImageRole.DATE_A,
    )


def _contract(candidate: Candidate, session_id: str) -> AnalysisContract:
    """The claim the verdict engine will be asked to rule on.

    Built here rather than drafted, so the probe tests the scene and not the
    planner. The claim asserts the direction the demo would assert.
    """
    asserted = (
        ChangeDirection.INCREASED
        if candidate.want == "increase"
        else ChangeDirection.DECREASED
    )
    return AnalysisContract(
        session_id=session_id,
        query=f"Has the {candidate.target} area {candidate.want}d?",
        claim=(
            f"The {candidate.target} area {candidate.want}d between the two dates."
        ),
        task_type=ContractTaskType.CLAIM_INVESTIGATION,
        configuration=InputConfiguration.BI_TEMPORAL_PAIR,
        acts_on=[ImageRole.DATE_A, ImageRole.DATE_B],
        target_classes=[candidate.target],
        change_direction=asserted,
        confounders=[
            ConfounderPlan(
                kind=kind,
                label=kind.value.replace("_", " "),
                question=f"Could {kind.value.replace('_', ' ')} explain this?",
            )
            for kind in (
                ConfounderKind.SEASONALITY,
                ConfounderKind.MISREGISTRATION,
                ConfounderKind.RADIOMETRY,
                ConfounderKind.CLOUD_SHADOW,
            )
        ],
        tools=[
            ToolPlan(
                tool=ChangeCvaEngine.name,
                version=ChangeCvaEngine.version,
                rationale="measure the change",
                parameters={"index": candidate.index},
                order=0,
            ),
            ToolPlan(
                tool=ConfounderEngine.name,
                version=ConfounderEngine.version,
                rationale="try to explain it away",
                order=1,
            ),
            ToolPlan(
                tool=VerdictEngine.name,
                version=VerdictEngine.version,
                rationale="rule on the claim",
                order=2,
            ),
        ],
        source=ContractSource.OFFLINE_RULE_ROUTER,
        contract_hash="0" * 32,
    )


@dataclass
class Result:
    key: str
    place: str
    ok: bool
    note: str = ""
    index: str = ""
    want: str = WANT_DIRECTION
    before_km2: float | None = None
    after_km2: float | None = None
    gain_km2: float | None = None
    loss_km2: float | None = None
    net_km2: float | None = None
    separability: float | None = None
    sensitivity: float | None = None
    corroborated_km2: float | None = None
    observable: float | None = None
    month_delta: int | None = None
    day_delta: int | None = None
    # The only figure that actually decides the question.
    verdict: str | None = None
    confidence: float | None = None
    worst_confounder: str = ""
    ruled_out: int = 0
    tested: int = 0

    @property
    def direction(self) -> str:
        if self.net_km2 is None:
            return "unmeasured"
        if abs(self.net_km2) < 0.01:
            return "unchanged"
        return "increase" if self.net_km2 > 0 else "decrease"

    @property
    def carries_demo_two(self) -> bool:
        """Whether the verdict engine actually supports the claim on this scene.

        This used to be a set of proxies: direction, separability, threshold
        stability, seasonal offset. Every proxy was individually reasonable and
        together they still certified a reservoir pair that the confounder engine
        refused, because the two dates came from different satellites and the
        radiometric offset between them was larger than the change. The only
        honest test is the one the demo will run.
        """
        return self.ok and self.verdict == "supported"


def measure(candidate: Candidate, probe_dir: Path, client: httpx.Client) -> Result:
    result = Result(
        key=candidate.key,
        place=candidate.place,
        ok=False,
        index=candidate.index,
        want=candidate.want,
    )
    preset = as_preset(candidate)

    try:
        fetch_preset(preset, client, probe_dir, allow_sar_fallback=False)
    except Exception as exc:  # noqa: BLE001 - a candidate that cannot be fetched is out
        result.note = f"fetch failed: {exc}"
        return result

    store = get_store()
    session = store.create()
    for role in (ImageRole.DATE_A, ImageRole.DATE_B):
        source = probe_dir / f"{preset.key}__{role.value}.tif"
        # Copied rather than moved, so a re-probe does not have to refetch.
        store.ingest(
            session_id=session.session_id,
            role=role,
            source=source,
            original_filename=source.name,
            move=False,
        )
    session = store.load(session.session_id)

    readiness = evaluate_readiness(session, store)
    result.month_delta = readiness.month_of_year_delta
    result.day_delta = readiness.day_delta

    context = ToolContext(
        session=session,
        store=store,
        readiness=readiness,
        parameters={"index": candidate.index},
        target_classes=[candidate.target],
        contract=_contract(candidate, session.session_id),
    )
    outcome = ChangeCvaEngine().run(context)
    if not outcome.ok:
        result.note = outcome.skipped_reason or "the change engine declined"
        return result

    # The change engine alone was not enough to choose a scene. It certified a
    # reservoir pair that the confounder engine then killed on radiometry, because
    # the two dates came from different satellites. So the probe runs the whole
    # pipeline and reports the verdict, which is the only figure that actually
    # answers "can this scene support its claim".
    context.upstream[ChangeCvaEngine.name] = outcome
    confounders = ConfounderEngine().run(context)
    if confounders.ok:
        context.upstream[ConfounderEngine.name] = confounders
        report = confounders.artifacts.get("confounders.report")
        if report is not None:
            surviving = sorted(
                report.surviving, key=lambda test: -test.penalty
            )
            if surviving:
                result.worst_confounder = (
                    f"{surviving[0].kind.value} "
                    f"({surviving[0].verdict.value.replace('_', ' ')})"
                )
            result.ruled_out = len(report.ruled_out)
            result.tested = len(report.tests)

    decision = VerdictEngine().run(context)
    if decision.ok:
        verdict = decision.artifacts.get("verdict.result")
        if verdict is not None:
            result.verdict = verdict.label.value
            result.confidence = verdict.confidence

    def value(key: str) -> float | None:
        measurement = outcome.measurement(key)
        return None if measurement is None else measurement.value

    result.ok = True
    result.before_km2 = value("area_km2_before")
    result.after_km2 = value("area_km2_after")
    result.gain_km2 = value("gain_km2")
    result.loss_km2 = value("loss_km2")
    result.net_km2 = value("net_change_km2")
    result.separability = value("class_separability")
    result.sensitivity = value("threshold_sensitivity")
    result.corroborated_km2 = value("corroborated_gain_km2")
    result.observable = value("observable_fraction")
    return result


def report(results: list[Result]) -> None:
    print("\n" + "=" * 78)
    print("MEASURED, not guessed. A candidate carries DEMO 2 when the direction")
    print("matches the claim, the answer survives the threshold moving, and the")
    print("dates sit in the same part of the year.")
    print("=" * 78)

    header = (
        f"{'candidate':<16} {'index':<6} {'net km2':>9} {'sens':>6} {'sep':>6} "
        f"{'mo':>3} {'verdict':<13} {'conf':>5}  what is still standing"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for item in sorted(
        results, key=lambda r: (not r.carries_demo_two, -(r.confidence or 0.0))
    ):
        if not item.ok:
            print(f"{item.key:<16} {item.index:<6} {'':>9} {'':>6} {'':>6} {'':>3} "
                  f"{'OUT':<13} {'':>5}  {item.note[:34]}")
            continue
        print(
            f"{item.key:<16} {item.index:<6} {item.net_km2:>9.3f} "
            f"{item.sensitivity:>6.3f} {item.separability:>6.3f} "
            f"{item.month_delta or 0:>3} {(item.verdict or 'none'):<13} "
            f"{(item.confidence or 0.0):>5.2f}  "
            + (item.worst_confounder or f"{item.ruled_out} of {item.tested} ruled out")
        )

    winners = [item for item in results if item.carries_demo_two]
    print()
    for item in winners:
        print(f"  {item.key}: {item.place}")
        print(f"    SUPPORTED at {(item.confidence or 0.0):.0%} confidence, "
              f"{item.ruled_out} of {item.tested} alternative explanations ruled out")
        print(f"    area {item.before_km2:.3f} -> {item.after_km2:.3f} km2 "
              f"({item.net_km2:+.3f} km2)")
        print(f"    gain {item.gain_km2:.3f} km2, of which NDVI corroborates "
              f"{item.corroborated_km2 or 0:.3f} km2")
        print(f"    threshold sensitivity {item.sensitivity:.3f} "
              f"(a nudge of 0.05 moves the answer this much)")
        print(f"    {item.day_delta} days apart, "
              f"{item.month_delta} month(s) of seasonal offset")
        print(f"    observable on both dates: {(item.observable or 0):.1%}")

    if not winners:
        print("  No candidate carries DEMO 2. Do not promote any of them.")
        print("  A scene that cannot support the claim it is named for is worse")
        print("  than no scene: the demo would end in a refusal the audience")
        print("  reads as a bug.")


def main() -> int:
    wanted = sys.argv[1:]
    chosen = (
        [CANDIDATES_BY_KEY[key] for key in wanted if key in CANDIDATES_BY_KEY]
        if wanted
        else list(CANDIDATES)
    )
    unknown = [key for key in wanted if key not in CANDIDATES_BY_KEY]
    if unknown:
        print(f"unknown candidate(s): {', '.join(unknown)}")
        print(f"known: {', '.join(CANDIDATES_BY_KEY)}")
        return 2

    probe_dir = get_settings().cache_dir / "aoi_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for candidate in chosen:
            results.append(measure(candidate, probe_dir, client))

    report(results)
    return 0 if any(item.carries_demo_two for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
