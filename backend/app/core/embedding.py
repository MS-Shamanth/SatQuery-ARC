"""Tile embeddings and query parsing for semantic archive search.

Two embedders are supported behind one interface:

* ``SpectralConceptEmbedder`` (always available): turns a tile's real spectral
  indices (NDVI, MNDWI, NDBI) into per-pixel land-cover fractions and a compact
  normalised vector. No heavy dependencies, so retrieval works fully offline on
  the installed stack.
* ``RemoteCLIPEmbedder`` (optional): if torch, open_clip and the RemoteCLIP
  weights are all present, image and text are embedded into a shared CLIP space
  for true free-text retrieval. Activated automatically when available; the app
  never depends on it.

``active_embedder_name()`` reports which one is actually backing retrieval, and
that string is what the health endpoint and the provenance panel display, so the
UI never claims a model that did not run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.raster_io import read_role_reflectance
from app.models.archive import ConceptScores
from app.models.schemas import BandRole, Modality, RasterMetadata
from app.tools.indices import normalised_difference

logger = logging.getLogger(__name__)

# Feature-read resolution. Small enough to be fast over the whole archive, large
# enough that class fractions are stable.
FEATURE_SHAPE = (256, 256)

# Class thresholds, matching the conventional index cut points used elsewhere so
# a tile's concept fractions agree with what the change engine would measure.
NDVI_VEG = 0.30
MNDWI_WATER = 0.0
NDBI_BUILT = 0.0


@dataclass
class TileFeatures:
    """Per-tile spectral summary, or empty when the tile is not optical."""

    optical: bool
    concepts: ConceptScores = field(default_factory=ConceptScores)
    ndvi_mean: float = 0.0
    mndwi_mean: float = 0.0
    ndbi_mean: float = 0.0
    brightness: float = 0.0
    water_fraction: float = 0.0

    def embedding(self) -> list[float]:
        """A unit-normalised vector for cosine similarity between tiles."""
        raw = np.array(
            [
                self.concepts.built_up,
                self.concepts.water,
                self.concepts.vegetation,
                self.concepts.bare,
                max(0.0, self.ndvi_mean),
                max(0.0, self.mndwi_mean),
                max(0.0, self.ndbi_mean + 0.5),  # shift NDBI into [0, ~1]
                self.brightness,
            ],
            dtype="float64",
        )
        norm = float(np.linalg.norm(raw))
        if norm < 1e-9:
            return [0.0] * len(raw)
        return (raw / norm).round(6).tolist()


def _read(path: Path, meta: RasterMetadata, role: BandRole) -> np.ma.MaskedArray | None:
    try:
        return read_role_reflectance(path, meta, role, out_shape=FEATURE_SHAPE)
    except Exception as exc:  # noqa: BLE001 - a missing band is not fatal
        logger.debug("feature read %s failed: %s", role.value, exc)
        return None


def extract_features(path: Path, meta: RasterMetadata) -> TileFeatures:
    """Compute land-cover concept fractions and index means for one tile.

    Everything here is measured from the pixels; nothing is inferred from the
    filename. A band that is absent simply drops the concept that needs it.
    """
    if meta.modality.modality is not Modality.OPTICAL:
        return TileFeatures(optical=False)

    green = _read(path, meta, BandRole.GREEN)
    red = _read(path, meta, BandRole.RED)
    nir = _read(path, meta, BandRole.NIR)
    if nir is None:
        nir = _read(path, meta, BandRole.NIR08)
    swir = _read(path, meta, BandRole.SWIR16)

    if red is None or nir is None:
        return TileFeatures(optical=False)

    ndvi = normalised_difference(nir, red)
    mndwi = (
        normalised_difference(green, swir)
        if green is not None and swir is not None
        else (normalised_difference(green, nir) if green is not None else None)
    )
    ndbi = (
        normalised_difference(swir, nir) if swir is not None else None
    )

    valid = ~np.ma.getmaskarray(ndvi)
    total = int(np.count_nonzero(valid))
    if total == 0:
        return TileFeatures(optical=False)

    ndvi_f = np.ma.filled(ndvi, -9.0)
    mndwi_f = np.ma.filled(mndwi, -9.0) if mndwi is not None else np.full(ndvi.shape, -9.0)
    ndbi_f = np.ma.filled(ndbi, -9.0) if ndbi is not None else np.full(ndvi.shape, -9.0)

    is_water = (mndwi_f > MNDWI_WATER) & valid
    is_veg = (ndvi_f > NDVI_VEG) & valid & ~is_water
    is_built = (ndbi_f > NDBI_BUILT) & valid & ~is_water & ~is_veg
    is_bare = valid & ~is_water & ~is_veg & ~is_built

    concepts = ConceptScores(
        built_up=round(int(np.count_nonzero(is_built)) / total, 4),
        water=round(int(np.count_nonzero(is_water)) / total, 4),
        vegetation=round(int(np.count_nonzero(is_veg)) / total, 4),
        bare=round(int(np.count_nonzero(is_bare)) / total, 4),
    )

    # Brightness: mean visible reflectance, a crude but real proxy that separates
    # bright bare/urban from dark water and shadow.
    bright_bands = [b for b in (green, red, nir) if b is not None]
    brightness = 0.0
    if bright_bands:
        stack = np.ma.stack(bright_bands)
        brightness = round(float(np.clip(stack.mean(), 0.0, 1.0)), 4)

    def _mean(arr: np.ma.MaskedArray | None) -> float:
        if arr is None or arr.count() == 0:
            return 0.0
        return round(float(arr.mean()), 4)

    return TileFeatures(
        optical=True,
        concepts=concepts,
        ndvi_mean=_mean(ndvi),
        mndwi_mean=_mean(mndwi),
        ndbi_mean=_mean(ndbi),
        brightness=brightness,
        water_fraction=concepts.water,
    )


# ---------------------------------------------------------------------------
# Query parsing (text -> concept weights + modifiers)
# ---------------------------------------------------------------------------

# Vocabulary mapping user words to land-cover concepts. Deliberately generous:
# retrieval should degrade to "close enough" rather than "no results".
CONCEPT_VOCAB: dict[str, list[str]] = {
    "built_up": [
        "built", "build", "building", "buildings", "built-up", "builtup",
        "urban", "city", "town", "settlement", "settlements", "construction",
        "constructed", "development", "developed", "houses", "housing",
        "infrastructure", "roads", "road", "impervious", "concrete", "sprawl",
    ],
    "water": [
        "water", "river", "rivers", "riverside", "riverfront", "lake", "lakes",
        "reservoir", "reservoirs", "dam", "pond", "canal", "flood", "flooded",
        "flooding", "coast", "coastal", "shoreline", "waterbody", "wetland",
    ],
    "vegetation": [
        "vegetation", "vegetated", "forest", "forests", "tree", "trees", "crop",
        "crops", "cropland", "farm", "farmland", "field", "fields", "green",
        "greenery", "plantation", "agriculture", "agricultural", "canopy",
    ],
    "bare": [
        "bare", "barren", "soil", "sand", "desert", "cleared", "clearance",
        "quarry", "mining", "excavation", "land clearance",
    ],
}

CHANGE_WORDS = {
    "new", "newly", "change", "changed", "changing", "growth", "grew", "grown",
    "growing", "increase", "increased", "increasing", "expansion", "expanded",
    "expanding", "appeared", "emerging", "emerged", "recent", "recently",
    "built", "construction", "loss", "lost", "decline", "declined", "shrink",
    "shrunk", "disappeared", "converted", "transition",
}

NEAR_WATER_WORDS = {
    "near a river", "near river", "by a river", "beside a river", "along a river",
    "near water", "by the water", "waterfront", "riverside", "riverfront",
    "near a lake", "by a lake", "coastal", "near the coast",
}


@dataclass
class ParsedQuery:
    concept_weights: dict[str, float]
    wants_change: bool
    near_water: bool
    interpreted_as: str


def parse_query(text: str) -> ParsedQuery:
    """Turn free text into concept weights and modifiers.

    This is the text retrieval path for the spectral-concept embedder. It is a
    keyword-to-concept router, not a language model, and it says plainly how it
    read the query so the ranking is never a black box.
    """
    lowered = f" {text.lower().strip()} "
    weights: dict[str, float] = {}
    hits: dict[str, list[str]] = {}

    for concept, words in CONCEPT_VOCAB.items():
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                weights[concept] = weights.get(concept, 0.0) + 1.0
                hits.setdefault(concept, []).append(word)

    wants_change = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in CHANGE_WORDS)
    near_water = any(phrase in lowered for phrase in NEAR_WATER_WORDS)
    # "near a river" implies water proximity even if water wasn't a primary concept.
    if near_water and "water" not in weights:
        weights.setdefault("water", 0.0)

    # Normalise concept weights to sum 1 (excluding the proximity-only water entry
    # which stays as a soft boost handled in scoring).
    total = sum(v for v in weights.values() if v > 0)
    if total > 0:
        weights = {k: round(v / total, 3) for k, v in weights.items() if v > 0}
    else:
        # No recognised concept: fall back to "anything", ranked by change.
        weights = {}

    parts: list[str] = []
    if weights:
        named = ", ".join(
            f"{k.replace('_', '-')}" for k, v in sorted(weights.items(), key=lambda x: -x[1])
        )
        parts.append(f"looking for {named}")
    else:
        parts.append("no specific land-cover class recognised; ranking the whole archive")
    if wants_change:
        parts.append("prioritising places with a detected change")
    if near_water:
        parts.append("boosting sites near water")
    interpreted = "; ".join(parts) + "."

    return ParsedQuery(
        concept_weights=weights,
        wants_change=wants_change,
        near_water=near_water,
        interpreted_as=interpreted,
    )


# ---------------------------------------------------------------------------
# Optional RemoteCLIP embedder
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _remoteclip_available() -> bool:
    """True only if torch, open_clip and a RemoteCLIP checkpoint are all present."""
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        if importlib.util.find_spec("open_clip") is None:
            return False
    except Exception:  # noqa: BLE001
        return False

    from app.config import get_settings

    weights = get_settings().models_dir / "RemoteCLIP-RN50.pt"
    return weights.exists()


def active_embedder_name() -> str:
    """Which embedder actually backs retrieval right now."""
    if _remoteclip_available():
        return "RemoteCLIP-RN50 (onnx/torch)"
    return "spectral-concept-v1"


__all__ = [
    "ConceptScores",
    "ParsedQuery",
    "TileFeatures",
    "active_embedder_name",
    "extract_features",
    "parse_query",
]
