"""Evidence disagreement engine tests.

This panel is the one place the system shows its own uncertainty as the headline,
so the things worth testing are not "does it produce a number" but the four rules
that stop the number being meaningless:

1. Methods are compared only about the same class. Pooling every mask made a
   vegetation index "disagree" with a water index and reported the whole scene as
   contested, which is true of no scene.
2. A method that could not see is not dissenting. Cloud is not an opinion.
3. The listed opinions describe ground the region actually covers. A centroid
   falls outside any non-convex region, and sampling there produced regions where
   every method agreed, printed under a heading that said they did not.
4. No winner is declared anywhere in the output.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.models.disagreement import Agreement
from app.models.schemas import ImageRole
from app.tools.base import MaskLayer, ToolError
from app.tools.disagreement import MIN_CONFLICT_AREA_M2, DisagreementEngine
from app.tools.indices import SpectralIndexEngine
from tests.raster_fixtures import (
    LandCover,
    make_labels,
    make_optical_scene,
)

WATER_RECT = (20, 150, 80, 90)


@pytest.fixture
def engine():
    return DisagreementEngine()


@pytest.fixture
def scene(tmp_path, make_context):
    """One optical scene with water, built-up, bare soil and vegetation."""
    labels = make_labels(water=WATER_RECT)
    optical = make_optical_scene(tmp_path / "opt.tif", labels=labels, seed=3)
    context = make_context(
        {ImageRole.SINGLE: optical.path}, target_classes=["water"]
    )
    return context, labels


def indices(context, names):
    outcome = SpectralIndexEngine().run(replace(context, parameters={"indices": names}))
    assert outcome.ok, outcome.skipped_reason
    return outcome


def geometry(outcome):
    """A transform and CRS borrowed from a real upstream mask."""
    layer = outcome.masks[0]
    return layer.transform, layer.crs


def synthetic_upstream(
    outcome,
    *,
    entries: list[tuple[str, str, str, np.ndarray, np.ndarray | None, float | None]],
):
    """Replace an outcome's masks with hand-built ones.

    Lets a test state the exact pixel-level agreement it wants to assert on,
    rather than hoping a threshold lands somewhere useful.
    """
    transform, crs = geometry(outcome)
    built = {}
    for tool, key, label, array, invalid, separability in entries:
        holder = built.setdefault(tool, replace(outcome, masks=[]))
        holder.masks.append(
            MaskLayer(
                key=key,
                label=label,
                array=array,
                transform=transform,
                crs=crs,
                description=label,
                applies_to=[ImageRole.SINGLE],
                invalid=invalid,
                separability=separability,
            )
        )
    return built


# -- rule 1: only the same class is compared ------------------------------


def test_two_methods_are_needed_before_anything_is_reported(scene, engine):
    context, _ = scene
    only_one = indices(context, ["NDWI"])

    probe = replace(context, upstream={"spectral-index-engine": only_one})
    ready, reason = engine.upstream_ready(probe)
    assert not ready
    assert "fewer than two" in reason

    with pytest.raises(ToolError, match="no disagreement to report|Fewer than two"):
        engine.execute(probe)


def test_masks_about_different_classes_are_not_treated_as_a_conflict(scene, engine):
    """A water index and a vegetation index have not been asked the same question.

    This is the bug that made the first version useless: pooling the two produced
    a contested fraction near 1.0 on every scene, because almost no pixel is both
    water and vegetation.
    """
    context, _ = scene
    both = indices(context, ["NDWI", "NDVI"])

    probe = replace(context, upstream={"spectral-index-engine": both})
    with pytest.raises(ToolError, match="No class was assessed by two different"):
        engine.execute(probe)


def test_two_water_indices_are_compared_and_a_vegetation_index_is_ignored(
    scene, engine
):
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI", "NDVI"])

    probe = replace(context, upstream={"spectral-index-engine": outcome})
    result = engine.run(probe)
    assert result.ok, result.skipped_reason

    report = result.artifacts["disagreement.report"]
    # Both water indices are witnesses. The vegetation index is not, because no
    # second method assessed vegetation.
    assert any("NDWI" in method for method in report.methods)
    assert any("MNDWI" in method for method in report.methods)
    assert not any("NDVI" in method for method in report.methods)


def test_the_contested_fraction_is_of_the_candidate_area_not_the_scene(scene, engine):
    context, labels = scene
    outcome = indices(context, ["NDWI", "MNDWI"])

    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )
    report = result.artifacts["disagreement.report"]

    scene_km2 = labels.size * 100.0 / 1_000_000.0
    # The candidate area is the ground some method put forward, which on this
    # scene is the reservoir and not the whole 256x256 grid.
    assert 0.0 < report.candidate_area_km2 < scene_km2 * 0.5
    assert report.conflicting_fraction == pytest.approx(
        (report.disagree_area_km2 + report.uncertain_area_km2)
        / report.candidate_area_km2,
        abs=1e-6,
    )
    assert 0.0 <= report.conflicting_fraction <= 1.0


def test_the_three_states_partition_the_candidate_area(scene, engine):
    """Every candidate pixel is in exactly one of agree, disagree, uncertain."""
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )
    report = result.artifacts["disagreement.report"]

    agree = result.mask("evidence_agree")
    disagree = result.mask("evidence_disagree")
    uncertain = result.mask("evidence_uncertain")
    assert agree is not None and disagree is not None and uncertain is not None

    # Precedence is conflict, then uncertainty, then agreement: the map must never
    # show green over ground that is in dispute.
    assert not np.any(agree.array & disagree.array)
    assert not np.any(agree.array & uncertain.array)
    assert not np.any(disagree.array & uncertain.array)

    assert report.agree_area_km2 + report.disagree_area_km2 + report.uncertain_area_km2 == (
        pytest.approx(report.candidate_area_km2, abs=1e-6)
    )


# -- rule 2: not seeing is not dissenting ---------------------------------


def test_a_method_that_could_not_see_is_excluded_rather_than_counted_as_dissent(
    scene, engine
):
    context, _ = scene
    base = indices(context, ["NDWI"])
    shape = base.masks[0].array.shape

    # Two methods that would disagree over a band of pixels, except that the
    # second one is blind exactly there.
    first = np.zeros(shape, dtype=bool)
    first[40:80, 40:120] = True
    second = np.zeros(shape, dtype=bool)
    second[40:80, 40:80] = True
    blind = np.zeros(shape, dtype=bool)
    blind[40:80, 80:120] = True

    upstream = synthetic_upstream(
        base,
        entries=[
            ("spectral-index-engine", "ndwi_mask", "NDWI water", first, None, 0.9),
            ("sar-backscatter-engine", "sar_water", "Radar water", second, blind, None),
        ],
    )

    result = engine.run(replace(context, upstream=upstream))
    assert result.ok, result.skipped_reason
    report = result.artifacts["disagreement.report"]

    disagree = result.mask("evidence_disagree")
    assert disagree is not None
    # The band only one method could see is not conflict.
    assert not disagree.array[40:80, 80:120].any()
    assert report.disagree_area_km2 == pytest.approx(0.0, abs=1e-9)
    assert any("not dissent" in note for note in report.notes)


def test_a_blind_method_is_labelled_as_unable_to_see_rather_than_disagreeing(
    scene, engine
):
    context, _ = scene
    base = indices(context, ["NDWI"])
    shape = base.masks[0].array.shape

    first = np.zeros(shape, dtype=bool)
    first[40:90, 40:120] = True
    second = np.zeros(shape, dtype=bool)
    second[40:90, 40:70] = True
    # Blind over part of the ground the two genuinely split on, so a region is
    # still produced and one of its witnesses could not see.
    blind = np.zeros(shape, dtype=bool)
    blind[60:90, 70:120] = True

    upstream = synthetic_upstream(
        base,
        entries=[
            ("spectral-index-engine", "ndwi_mask", "NDWI water", first, None, 0.9),
            ("sar-backscatter-engine", "sar_water", "Radar water", second, blind, None),
        ],
    )
    report = engine.run(
        replace(context, upstream=upstream)
    ).artifacts["disagreement.report"]

    blind_opinions = [
        opinion
        for region in report.regions
        for opinion in region.opinions
        if opinion.could_not_see
    ]
    for opinion in blind_opinions:
        assert opinion.verdict == "could not see it"
    # And a region whose split is explained by blindness says so.
    for region in report.regions:
        if any(opinion.could_not_see for opinion in region.opinions):
            assert "could not observe" in region.reason


# -- rule 3: the sampled pixel is inside the region -----------------------


def test_the_listed_opinions_describe_ground_the_region_actually_covers(
    scene, engine
):
    """A ring: its centroid is in the hole, where every method agrees.

    Sampling the centroid reported "not water" from every witness under a heading
    that said they disagreed. The pixel has to come from inside the region.
    """
    context, _ = scene
    base = indices(context, ["NDWI"])
    shape = base.masks[0].array.shape

    ring = np.zeros(shape, dtype=bool)
    ring[60:160, 60:160] = True
    ring[85:135, 85:135] = False  # the hole the centroid lands in

    first = ring.copy()
    second = np.zeros(shape, dtype=bool)  # dissents everywhere on the ring

    upstream = synthetic_upstream(
        base,
        entries=[
            ("spectral-index-engine", "ndwi_mask", "NDWI water", first, None, 0.9),
            ("sar-backscatter-engine", "sar_water", "Radar water", second, None, None),
        ],
    )
    report = engine.run(
        replace(context, upstream=upstream)
    ).artifacts["disagreement.report"]

    assert report.regions, "a ring of contested pixels should produce a region"
    region = report.regions[0]
    assert region.agreement is Agreement.DISAGREE
    # The whole point: the witnesses must actually be split about the sampled
    # pixel, which only holds if the pixel is inside the ring.
    verdicts = {opinion.verdict for opinion in region.opinions}
    assert "WATER" in verdicts
    assert "not water" in verdicts
    assert len(region.dissenting_methods) >= 2


def test_regions_below_the_review_threshold_are_not_listed(scene, engine):
    context, _ = scene
    base = indices(context, ["NDWI"])
    shape = base.masks[0].array.shape

    # 3x3 at ten metres is 900 m2, well under the threshold.
    speck = np.zeros(shape, dtype=bool)
    speck[10:13, 10:13] = True

    upstream = synthetic_upstream(
        base,
        entries=[
            ("spectral-index-engine", "ndwi_mask", "NDWI water", speck, None, 0.9),
            (
                "sar-backscatter-engine",
                "sar_water",
                "Radar water",
                np.zeros(shape, dtype=bool),
                None,
                None,
            ),
        ],
    )
    report = engine.run(
        replace(context, upstream=upstream)
    ).artifacts["disagreement.report"]

    assert MIN_CONFLICT_AREA_M2 > 900.0
    assert report.regions == []
    # The area is still counted; only the work queue is filtered.
    assert report.disagree_area_km2 > 0.0


# -- rule 4: no winner ----------------------------------------------------


def test_no_region_declares_a_winning_method(scene, engine):
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )
    report = result.artifacts["disagreement.report"]

    banned = ("correct", "wrong", "true value", "ground truth", "more accurate")
    for region in report.regions:
        assert region.action
        assert "review" in region.action.lower() or "observation" in region.action.lower()
        for word in banned:
            assert word not in region.reason.lower()
            assert word not in region.action.lower()

    assert any("No method is treated as correct" in note for note in report.notes)


def test_every_opinion_names_the_tool_and_version_that_produced_it(scene, engine):
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    report = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    ).artifacts["disagreement.report"]

    for region in report.regions:
        assert region.opinions, "a region with no witnesses explains nothing"
        for opinion in region.opinions:
            assert opinion.source_tool
            assert opinion.source_version
            assert opinion.basis


def test_a_reason_naming_separability_carries_the_measured_value(scene, engine):
    """"Low spectral separability" is only written where it was measured."""
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    report = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    ).artifacts["disagreement.report"]

    for region in report.regions:
        if "separability" in region.reason.lower():
            assert region.reason_measurement_key == "class_separability"
            assert region.reason_value is not None
            assert 0.0 <= region.reason_value <= 1.0


# -- measurements and labelling -------------------------------------------


def test_the_headline_figures_are_measurements_with_formulas(scene, engine):
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )

    for key in (
        "conflicting_area_km2",
        "conflicting_fraction",
        "agreeing_area_km2",
        "uncertain_area_km2",
        "methods_compared",
        "conflict_regions",
    ):
        measurement = result.measurement(key)
        assert measurement is not None, f"{key} was not reported"
        assert measurement.formula, f"{key} carries no formula"
        assert measurement.source_tool == engine.name
        assert measurement.source_version == engine.version

    report = result.artifacts["disagreement.report"]
    assert result.measurement("conflicting_fraction").value == pytest.approx(
        report.conflicting_fraction, abs=1e-6
    )
    assert result.measurement("methods_compared").value == float(len(report.methods))


def test_one_tool_holding_two_opinions_is_labelled_by_index_not_by_tool(scene, engine):
    """MNDWI and NDWI both come from the index engine and disagree over cities.

    Labelling both "Spectral index" would present the most interesting conflict
    this panel can show as a tool arguing with itself.
    """
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    report = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    ).artifacts["disagreement.report"]

    assert "Spectral index (NDWI)" in report.methods
    assert "Spectral index (MNDWI)" in report.methods
    # Named once each, not once per mask.
    assert len(report.methods) == len(set(report.methods))


def test_masks_on_a_different_grid_are_dropped_rather_than_compared(scene, engine):
    """Comparing mismatched grids would map the resampling, not the evidence."""
    context, _ = scene
    base = indices(context, ["NDWI"])
    shape = base.masks[0].array.shape

    matching_a = np.zeros(shape, dtype=bool)
    matching_a[40:90, 40:120] = True
    matching_b = np.zeros(shape, dtype=bool)
    matching_b[40:90, 40:70] = True
    odd = np.zeros((shape[0] // 2, shape[1] // 2), dtype=bool)
    odd[5:20, 5:20] = True

    upstream = synthetic_upstream(
        base,
        entries=[
            ("spectral-index-engine", "ndwi_mask", "NDWI water", matching_a, None, 0.9),
            ("sar-backscatter-engine", "sar_water", "Radar water", matching_b, None, None),
            ("rs-landcover-probe", "probe_water", "Probe water", odd, None, None),
        ],
    )
    report = engine.run(
        replace(context, upstream=upstream)
    ).artifacts["disagreement.report"]

    assert len(report.methods) == 2
    assert not any("probe" in method.lower() for method in report.methods)


def test_the_summary_line_states_the_contested_share_and_the_method_count(
    scene, engine
):
    context, _ = scene
    outcome = indices(context, ["NDWI", "MNDWI"])
    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )
    report = result.artifacts["disagreement.report"]

    line = report.summary_line()
    assert "conflicting evidence" in line
    assert f"{len(report.methods)} methods" in line
    # The trace shows this first, so it has to lead the notes.
    assert result.notes[0].startswith(line)


def test_cloud_is_reported_as_unseen_rather_than_as_a_third_opinion(
    tmp_path, make_context, engine
):
    """The engine's own invalid layer must carry the union of what was unseen."""
    labels = make_labels(water=WATER_RECT, cloud=(0, 0, 40, 256))
    optical = make_optical_scene(
        tmp_path / "cloudy.tif", labels=labels, seed=3, with_scl=True
    )
    context = make_context({ImageRole.SINGLE: optical.path}, target_classes=["water"])

    outcome = indices(context, ["NDWI", "MNDWI"])
    result = engine.run(
        replace(context, upstream={"spectral-index-engine": outcome})
    )
    assert result.ok, result.skipped_reason

    for key in ("evidence_agree", "evidence_disagree", "evidence_uncertain"):
        layer = result.mask(key)
        assert layer is not None
        assert layer.invalid is not None
        # Nothing is claimed about ground no method could see.
        assert not np.any(layer.array & layer.invalid)

    assert int(np.count_nonzero(labels == int(LandCover.CLOUD))) > 0
