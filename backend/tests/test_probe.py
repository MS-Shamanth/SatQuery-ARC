"""The learned probe.

Three things are worth testing and the rest is arithmetic.

It has to actually learn: a classifier that cannot separate water from vegetation
on its own training distribution is not evidence that the adaptation path works.

It has to be honest about what it does not know: the figures it reports are held-out
results on synthetic spectra, and nothing in its output may read as a measurement of
accuracy on a real scene.

It has to be optional. Every scripted demonstration runs identically with it absent,
which is the condition for putting a fitted model inside a system whose promise is
that its numbers are traceable.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.adapter import (
    BAND_FEATURES,
    CLASSES,
    CLASS_SPECTRA,
    FEATURES,
    ProbeWeights,
    build_features,
    load_weights,
    predict,
    predict_proba,
    sample_training_spectra,
    save_weights,
    softmax,
    train,
    weights_path,
)
from app.models.schemas import ImageRole, ToolImplementation
from app.tools.indices import SpectralIndexEngine
from app.tools.probe import LandCoverProbe
from tests.raster_fixtures import make_labels, make_optical_scene


@pytest.fixture(scope="module")
def fitted() -> ProbeWeights:
    # Small but enough to converge; the shipped model uses far more.
    return train(per_class=700, epochs=350)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_the_feature_order_is_fixed() -> None:
    """A model whose inputs can be reordered silently is not reproducible."""
    assert FEATURES[:BAND_FEATURES] == (
        "blue",
        "green",
        "red",
        "nir",
        "swir16",
        "swir22",
    )
    assert len(FEATURES) == BAND_FEATURES + 5


def test_features_are_built_from_the_same_code_at_training_and_inference() -> None:
    bands = {
        name: np.array([[mean]], dtype="float64")
        for name, (mean, _) in zip(FEATURES[:BAND_FEATURES], CLASS_SPECTRA["water"])
    }
    row = build_features(bands)
    assert row.shape == (1, len(FEATURES))
    # Water: bright in green relative to the shortwave infrared, dark in the NIR.
    ndvi = row[0][FEATURES.index("ndvi")]
    mndwi = row[0][FEATURES.index("mndwi")]
    assert ndvi < 0.0
    assert mndwi > 0.0


def test_an_undefined_ratio_is_zero_rather_than_masked() -> None:
    """A feature present for some pixels and absent for others is not a feature."""
    bands = {name: np.zeros((2, 2)) for name in FEATURES[:BAND_FEATURES]}
    row = build_features(bands)
    assert np.isfinite(row).all()
    assert (row == 0.0).all()


def test_a_missing_band_is_refused_rather_than_filled_in() -> None:
    bands = {name: np.zeros((2, 2)) for name in FEATURES[:BAND_FEATURES][:-1]}
    with pytest.raises(ValueError, match="missing band"):
        build_features(bands)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_softmax_returns_a_distribution() -> None:
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_training_samples_are_balanced_across_classes() -> None:
    _, labels = sample_training_spectra(50)
    counts = np.bincount(labels, minlength=len(CLASSES))
    assert set(counts.tolist()) == {50}


def test_the_probe_separates_the_easy_classes_almost_perfectly(fitted) -> None:
    """Water and vegetation are spectrally distinct; failing them would mean a bug."""
    assert fitted.training.per_class_recall["water"] > 0.95
    assert fitted.training.per_class_recall["vegetation"] > 0.95


def test_the_probe_beats_chance_on_the_classes_that_overlap(fitted) -> None:
    """Built-up against bare is the actual task, and it is not solved.

    The training spectra overlap on purpose. What matters is that the classifier
    does substantially better than the one-in-four a coin would give, and that the
    figure is reported rather than rounded up.
    """
    for name in ("built-up", "bare"):
        recall = fitted.training.per_class_recall[name]
        assert recall > 0.6, f"{name} recall {recall:.3f} is no better than guessing"
        assert recall < 0.99, (
            f"{name} recall {recall:.3f} is suspiciously high for classes that "
            "overlap by construction"
        )


def test_the_confusion_is_between_built_up_and_bare(fitted) -> None:
    """The errors have to be where the physics says they should be."""
    matrix = fitted.training.confusion
    built = CLASSES.index("built-up")
    bare = CLASSES.index("bare")
    water = CLASSES.index("water")

    assert matrix[built][bare] > 0
    assert matrix[bare][built] > 0
    # Water is not confused with dry ground.
    assert matrix[water][built] + matrix[water][bare] < matrix[built][bare]


def test_training_is_reproducible(fitted) -> None:
    """A proof of adaptation that fits a different model each run is unpointable."""
    again = train(per_class=700, epochs=350)
    assert np.allclose(again.matrix(), fitted.matrix())
    assert again.training.accuracy == pytest.approx(fitted.training.accuracy)


def test_the_coefficients_agree_with_the_physics(fitted) -> None:
    """A linear model was chosen so this could be checked at all."""
    water = dict(fitted.influences("water"))
    vegetation = dict(fitted.influences("vegetation"))

    assert water["nir"] < 0, "water is dark in the near infrared"
    assert water["mndwi"] > 0, "water is what MNDWI is for"
    assert vegetation["ndvi"] > 0, "vegetation is what NDVI is for"


def test_the_held_out_set_is_not_the_training_set(fitted) -> None:
    record = fitted.training
    assert record.held_out > 0
    assert record.samples > record.held_out
    assert any("Held out before fitting" in note for note in record.notes)


def test_the_record_refuses_to_claim_field_accuracy(fitted) -> None:
    """The one figure a reader will quote has to carry its own caveat."""
    joined = " ".join(fitted.training.notes).lower()
    assert "synthetic" in joined
    assert "not a claim about field accuracy" in joined


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_weights_round_trip(fitted, tmp_path) -> None:
    save_weights(fitted, tmp_path)
    restored = load_weights(tmp_path)
    assert restored is not None
    assert np.allclose(restored.matrix(), fitted.matrix())
    assert restored.training.accuracy == pytest.approx(fitted.training.accuracy)


def test_an_untrained_checkout_has_no_probe(tmp_path) -> None:
    assert load_weights(tmp_path) is None


def test_a_corrupt_model_is_treated_as_a_missing_one(tmp_path) -> None:
    weights_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    weights_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert load_weights(tmp_path) is None


def test_weights_from_a_different_feature_set_are_refused(fitted, tmp_path) -> None:
    """Storing the feature order is pointless unless a mismatch is rejected."""
    stale = fitted.model_copy(deep=True)
    stale.features = list(FEATURES[:-1])
    save_weights(stale, tmp_path)
    assert load_weights(tmp_path) is None


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_context(make_context, tmp_path):
    labels = make_labels(96, 96, water=(8, 8, 30, 30), builtup=(50, 50, 30, 30))
    scene = make_optical_scene(
        tmp_path / "probe.tif", width=96, height=96, labels=labels
    )
    return make_context({ImageRole.SINGLE: scene.path}, with_readiness=True)


def test_the_probe_declares_itself_as_learned(fitted) -> None:
    tool = LandCoverProbe(fitted)
    assert tool.implementation is ToolImplementation.LEARNED
    assert "fitted on" in tool.summary


def test_the_probe_maps_a_real_scene_it_never_trained_on(fitted, scene_context) -> None:
    outcome = LandCoverProbe(fitted).run(scene_context)
    assert outcome.ok, outcome.skipped_reason

    keys = {mask.key for mask in outcome.masks}
    assert keys == {f"probe_{name.replace('-', '_')}" for name in CLASSES}

    # The synthetic scene is mostly vegetation with a water block and a built block,
    # so those two classes have to be found.
    water = outcome.measurement("probe_area_km2.water")
    built = outcome.measurement("probe_area_km2.built-up")
    assert water is not None and water.value > 0.0
    assert built is not None and built.value > 0.0


def test_every_probe_measurement_names_the_model_behind_it(
    fitted, scene_context
) -> None:
    outcome = LandCoverProbe(fitted).run(scene_context)
    accuracy = outcome.measurement("probe_held_out_accuracy")
    assert accuracy is not None
    assert accuracy.value == pytest.approx(fitted.training.accuracy)
    assert "held out before fitting" in (accuracy.method or "").lower()
    assert accuracy.inputs["held_out_samples"] == fitted.training.held_out
    # Every area measurement carries the model's accuracy in its method, so a
    # figure quoted from the probe cannot be separated from what produced it.
    area = outcome.measurement("probe_area_km2.water")
    assert "logistic regression" in (area.method or "")


def test_pixels_the_model_is_unsure_about_are_not_assigned(
    fitted, scene_context
) -> None:
    """A complete map with hidden guesses in it is worse than an honest gap."""
    scene_context.parameters = {"min_confidence": 0.99}
    outcome = LandCoverProbe(fitted).run(scene_context)
    unassigned = outcome.measurement("probe_unassigned_fraction")
    assert unassigned is not None and unassigned.value > 0.0

    scene_context.parameters = {"min_confidence": 0.3}
    relaxed = LandCoverProbe(fitted).run(scene_context)
    relaxed_unassigned = relaxed.measurement("probe_unassigned_fraction")
    assert relaxed_unassigned.value < unassigned.value


def test_the_probe_compares_itself_to_the_indices_without_scoring_either(
    fitted, scene_context
) -> None:
    """No labelled reference exists, so a winner would be an invented figure."""
    scene_context.target_classes = ["water"]
    scene_context.parameters = {"indices": ["NDVI", "MNDWI"]}
    index_outcome = SpectralIndexEngine().run(scene_context)
    assert index_outcome.ok
    scene_context.upstream[SpectralIndexEngine.name] = index_outcome
    scene_context.parameters = {}

    outcome = LandCoverProbe(fitted).run(scene_context)
    overlap = outcome.measurement("probe_agreement_iou.water")
    assert overlap is not None
    assert 0.0 <= overlap.value <= 1.0
    assert "not a score" in (overlap.method or "")
    assert overlap.inputs["index_mask"].startswith("mndwi_mask")

    joined = " ".join(outcome.notes).lower()
    assert "cannot be settled" in joined
    for forbidden in ("more accurate", "better than", "outperforms"):
        assert forbidden not in joined


def test_the_probe_states_its_limits_on_every_run(fitted, scene_context) -> None:
    outcome = LandCoverProbe(fitted).run(scene_context)
    joined = " ".join(outcome.notes).lower()
    assert "fitted weights, not a published formula" in joined
    assert "synthetic" in joined
    assert "not a measurement of its accuracy on this scene" in joined


def test_the_probe_refuses_a_scene_missing_a_band_it_was_fitted_on(
    fitted, make_context, tmp_path
) -> None:
    scene = make_optical_scene(
        tmp_path / "two.tif",
        width=64,
        height=64,
        labels=make_labels(64, 64),
        band_names=("blue", "green"),
    )
    context = make_context({ImageRole.SINGLE: scene.path})
    runnable, reason = LandCoverProbe(fitted).can_run(context)
    assert not runnable
    assert "swir16" in reason or "band" in reason.lower()


def test_the_probabilities_are_a_distribution_over_the_classes(
    fitted, scene_context
) -> None:
    bands = {
        name: np.array([[mean]], dtype="float64")
        for name, (mean, _) in zip(
            FEATURES[:BAND_FEATURES], CLASS_SPECTRA["built-up"]
        )
    }
    probabilities = predict_proba(build_features(bands), fitted)
    assert probabilities.shape == (1, len(CLASSES))
    assert probabilities.sum() == pytest.approx(1.0)

    chosen, confidence = predict(build_features(bands), fitted)
    assert CLASSES[chosen[0]] in {"built-up", "bare"}
    assert confidence[0] == pytest.approx(probabilities.max())
