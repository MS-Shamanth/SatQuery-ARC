"""A small learned land-cover classifier, and the honest case for it.

Everything else in this system measures with published formulas and declared
thresholds. That is deliberate, and it has a known limit: a global threshold on a
single index cannot separate classes whose spectra overlap. The clearest example
is in this repository's own sample library, where the built-up index scores zero at
splitting a real construction scene because at ten metres concrete and dry soil sit
in the same part of the histogram.

A classifier does not need a single global threshold. It can weigh every band and
every index at once. So this module exists to prove that the adaptation path works
end to end: a learned component trains, reports its own held-out performance,
registers as a tool, runs inside the same contract and trace as everything else,
and is labelled in the interface as a trained model rather than a computation.

What it is not:

* It is **not trained on ground truth.** There is no labelled reference for these
  scenes in this repository. It is trained on synthetic spectra drawn from
  published Sentinel-2 surface reflectance ranges per class, with noise. That is
  enough to prove the mechanism and not enough to claim field accuracy.
* Its agreement with the index-based masks on a real scene is therefore **not
  evidence that either is correct.** Both are reported, the disagreement is drawn,
  and neither is scored, because scoring without labels would be inventing a
  number. This is the same rule the optical-versus-radar comparison follows.

The model is multinomial logistic regression, fitted with gradient descent in
numpy. A linear model is the right choice here for reasons beyond convenience: its
coefficients are readable, so "why did it say built-up" has an answer in terms of
bands, and a proof of adaptation that cannot be inspected proves less.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WEIGHTS_FILENAME = "landcover_probe.json"

# The classes the probe distinguishes, in a fixed order. Bare ground is included
# specifically because it is what the built-up index confuses built-up with; a
# classifier that could not name it would not be addressing the actual failure.
CLASSES: tuple[str, ...] = ("water", "vegetation", "built-up", "bare")

# The feature vector, in a fixed order. Six reflectance bands plus the five indices
# the rest of the system already computes, so the learned tool and the
# deterministic tools are reading the same quantities and a disagreement between
# them is about the decision rule rather than about the inputs.
FEATURES: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir16",
    "swir22",
    "ndvi",
    "ndwi",
    "mndwi",
    "ndbi",
    "ndmi",
)

# Typical Sentinel-2 L2A surface reflectance per class, as fractions, with the
# spread actually seen within a class. Drawn from the ranges reported for Sentinel-2
# land-cover spectra rather than invented: water is dark everywhere and darkest in
# the infrared, vegetation has the red-edge jump, built-up is bright in the
# shortwave infrared, and bare soil is bright and flat.
#
# Built-up and bare are deliberately close. That overlap is the real problem this
# probe is meant to address, and a training set that separated them cleanly would
# be proving something easier than the task.
CLASS_SPECTRA: dict[str, tuple[tuple[float, float], ...]] = {
    #            blue          green         red           nir           swir16        swir22
    "water": (
        (0.055, 0.020), (0.070, 0.022), (0.045, 0.018), (0.025, 0.012),
        (0.015, 0.008), (0.010, 0.006),
    ),
    "vegetation": (
        (0.035, 0.012), (0.065, 0.018), (0.040, 0.015), (0.320, 0.070),
        (0.185, 0.045), (0.090, 0.030),
    ),
    "built-up": (
        (0.140, 0.035), (0.160, 0.038), (0.185, 0.045), (0.225, 0.050),
        (0.290, 0.060), (0.265, 0.058),
    ),
    "bare": (
        (0.130, 0.032), (0.165, 0.038), (0.215, 0.048), (0.285, 0.055),
        (0.345, 0.062), (0.295, 0.060),
    ),
}

BAND_FEATURES = 6


class TrainingRecord(BaseModel):
    """How the probe was trained, so its metrics can be checked rather than taken.

    Every figure the interface shows about this model comes from here, and every
    one of them is a held-out result on the training distribution. The record says
    so, because a reported accuracy without its provenance is indistinguishable
    from a claim.
    """

    trained_at: datetime
    generator: str
    seed: int
    samples: int
    held_out: int
    epochs: int
    learning_rate: float
    l2: float
    classes: list[str]
    features: list[str]

    # Held-out performance. On the training distribution, which is synthetic.
    accuracy: float
    per_class_recall: dict[str, float]
    # Rows are the true class, columns the predicted one, in ``classes`` order.
    confusion: list[list[int]]
    final_loss: float
    notes: list[str] = Field(default_factory=list)

    @property
    def summary_line(self) -> str:
        return (
            f"{self.accuracy:.1%} held-out accuracy on {self.held_out} synthetic "
            f"spectra across {len(self.classes)} classes"
        )


class ProbeWeights(BaseModel):
    """The fitted model, small enough to read."""

    classes: list[str]
    features: list[str]
    # Shape (n_classes, n_features).
    coefficients: list[list[float]]
    intercepts: list[float]
    # Standardisation, so the coefficients are comparable to one another.
    feature_mean: list[float]
    feature_scale: list[float]
    training: TrainingRecord

    def matrix(self) -> np.ndarray:
        return np.asarray(self.coefficients, dtype="float64")

    def influences(self, class_name: str) -> list[tuple[str, float]]:
        """The features that push hardest towards a class, largest first.

        This is why a linear model was chosen. "Built-up because the shortwave
        infrared is high and the vegetation index is low" is an explanation; a
        probability from an opaque model is not.
        """
        index = self.classes.index(class_name)
        pairs = list(zip(self.features, self.coefficients[index]))
        return sorted(pairs, key=lambda item: -abs(item[1]))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _ratio(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """A normalised difference that returns zero where it is undefined.

    Zero rather than a mask: this feeds a classifier, and a feature that is absent
    for some pixels and present for others would make the model's input depend on
    which pixels happened to be usable.
    """
    denominator = positive + negative
    out = np.zeros_like(denominator, dtype="float64")
    usable = np.abs(denominator) > 1e-9
    out[usable] = (positive[usable] - negative[usable]) / denominator[usable]
    return out


def build_features(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the feature vector for every pixel.

    ``bands`` holds reflectance fractions keyed by the six band names. Returns an
    array of shape (pixels, len(FEATURES)) in the declared feature order, because a
    model whose inputs can be reordered silently is not reproducible.
    """
    missing = [name for name in FEATURES[:BAND_FEATURES] if name not in bands]
    if missing:
        raise ValueError(f"missing band(s) for the probe: {', '.join(missing)}")

    blue = np.asarray(bands["blue"], dtype="float64").ravel()
    green = np.asarray(bands["green"], dtype="float64").ravel()
    red = np.asarray(bands["red"], dtype="float64").ravel()
    nir = np.asarray(bands["nir"], dtype="float64").ravel()
    swir16 = np.asarray(bands["swir16"], dtype="float64").ravel()
    swir22 = np.asarray(bands["swir22"], dtype="float64").ravel()

    return np.column_stack(
        [
            blue,
            green,
            red,
            nir,
            swir16,
            swir22,
            _ratio(nir, red),      # NDVI
            _ratio(green, nir),    # NDWI
            _ratio(green, swir16), # MNDWI
            _ratio(swir16, nir),   # NDBI
            _ratio(nir, swir16),   # NDMI
        ]
    )


def sample_training_spectra(
    per_class: int, *, seed: int = 20260921
) -> tuple[np.ndarray, np.ndarray]:
    """Draw labelled spectra from the documented per-class ranges.

    Gaussian around the class mean, clipped to the physically possible [0, 1], then
    passed through the same feature builder the tool uses at inference. Using one
    feature builder for training and inference is what stops the two drifting apart,
    which is the most common way a small model like this silently breaks.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for index, class_name in enumerate(CLASSES):
        spectra = CLASS_SPECTRA[class_name]
        draws = {
            name: np.clip(
                rng.normal(mean, spread, per_class), 0.0, 1.0
            )
            for name, (mean, spread) in zip(FEATURES[:BAND_FEATURES], spectra)
        }
        rows.append(build_features(draws))
        labels.extend([index] * per_class)

    return np.vstack(rows), np.asarray(labels, dtype="int64")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    classes: int,
    epochs: int = 600,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Multinomial logistic regression by full-batch gradient descent.

    Full batch and a fixed number of epochs rather than early stopping, so the same
    inputs give the same weights every time. A proof of adaptation that trains to a
    different model each run cannot be pointed at.
    """
    samples, dimensions = features.shape
    weights = np.zeros((classes, dimensions), dtype="float64")
    bias = np.zeros(classes, dtype="float64")
    targets = np.zeros((samples, classes), dtype="float64")
    targets[np.arange(samples), labels] = 1.0

    loss = float("nan")
    for _ in range(epochs):
        probabilities = softmax(features @ weights.T + bias)
        error = probabilities - targets
        weights -= learning_rate * ((error.T @ features) / samples + l2 * weights)
        bias -= learning_rate * error.mean(axis=0)
        loss = float(
            -np.log(np.clip(probabilities[np.arange(samples), labels], 1e-12, 1.0)).mean()
        )

    return weights, bias, loss


def predict_proba(features: np.ndarray, weights: ProbeWeights) -> np.ndarray:
    mean = np.asarray(weights.feature_mean, dtype="float64")
    scale = np.asarray(weights.feature_scale, dtype="float64")
    standardised = (features - mean) / np.where(scale > 1e-12, scale, 1.0)
    return softmax(standardised @ weights.matrix().T + np.asarray(weights.intercepts))


def predict(features: np.ndarray, weights: ProbeWeights) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class index per row, and the probability it was given."""
    probabilities = predict_proba(features, weights)
    chosen = probabilities.argmax(axis=1)
    return chosen, probabilities[np.arange(len(chosen)), chosen]


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, classes: int) -> list[list[int]]:
    matrix = np.zeros((classes, classes), dtype="int64")
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1
    return matrix.tolist()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def weights_path(models_dir: Path) -> Path:
    return Path(models_dir) / WEIGHTS_FILENAME


def save_weights(weights: ProbeWeights, models_dir: Path) -> Path:
    path = weights_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(weights.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_weights(models_dir: Path) -> ProbeWeights | None:
    """The trained probe, or None when it has not been trained here.

    None rather than an exception: an untrained probe is a tool that declines to
    register, which the capability panel then reports with its reason. A missing
    optional model must not stop the system starting.
    """
    path = weights_path(models_dir)
    if not path.exists():
        return None
    try:
        weights = ProbeWeights.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt model is a missing model
        logger.warning("ignoring unreadable probe weights at %s: %s", path, exc)
        return None

    if list(weights.features) != list(FEATURES) or list(weights.classes) != list(CLASSES):
        # Refusing a stale model is the whole point of storing the feature order.
        logger.warning(
            "probe weights at %s were trained on a different feature or class set, "
            "so they are ignored",
            path,
        )
        return None
    return weights


def train(
    *,
    per_class: int = 4000,
    held_out_fraction: float = 0.25,
    epochs: int = 600,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
    seed: int = 20260921,
) -> ProbeWeights:
    """Fit the probe and measure it on data it never saw."""
    features, labels = sample_training_spectra(per_class, seed=seed)

    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(labels))
    features, labels = features[order], labels[order]
    cut = int(len(labels) * (1.0 - held_out_fraction))
    train_x, train_y = features[:cut], labels[:cut]
    test_x, test_y = features[cut:], labels[cut:]

    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    weights, bias, loss = fit_logistic(
        (train_x - mean) / scale,
        train_y,
        classes=len(CLASSES),
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )

    fitted = ProbeWeights(
        classes=list(CLASSES),
        features=list(FEATURES),
        coefficients=weights.tolist(),
        intercepts=bias.tolist(),
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        training=TrainingRecord(
            trained_at=datetime.now(timezone.utc),
            generator=(
                "synthetic spectra drawn from published Sentinel-2 L2A surface "
                "reflectance ranges per land-cover class, with Gaussian spread"
            ),
            seed=seed,
            samples=int(len(train_y)),
            held_out=int(len(test_y)),
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            classes=list(CLASSES),
            features=list(FEATURES),
            accuracy=0.0,
            per_class_recall={},
            confusion=[],
            final_loss=loss,
            notes=[],
        ),
    )

    predicted, _ = predict(test_x, fitted)
    matrix = confusion_matrix(test_y, predicted, len(CLASSES))
    recalls: dict[str, float] = {}
    for index, name in enumerate(CLASSES):
        total = sum(matrix[index])
        recalls[name] = (matrix[index][index] / total) if total else 0.0

    fitted.training.accuracy = float((predicted == test_y).mean())
    fitted.training.per_class_recall = recalls
    fitted.training.confusion = matrix
    fitted.training.notes = [
        "Held out before fitting, so these figures are not measured on the data "
        "the model learned from.",
        "The training distribution is synthetic. These figures describe the model's "
        "behaviour on that distribution and are not a claim about field accuracy on "
        "any real scene.",
        "Built-up and bare ground overlap in the training spectra on purpose, "
        "because that overlap is the failure this probe exists to address. Their "
        "recall is the number worth reading.",
    ]
    return fitted


__all__ = [
    "BAND_FEATURES",
    "CLASSES",
    "CLASS_SPECTRA",
    "FEATURES",
    "WEIGHTS_FILENAME",
    "ProbeWeights",
    "TrainingRecord",
    "build_features",
    "confusion_matrix",
    "fit_logistic",
    "load_weights",
    "predict",
    "predict_proba",
    "sample_training_spectra",
    "save_weights",
    "softmax",
    "train",
    "weights_path",
]
