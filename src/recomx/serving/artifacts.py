"""Validated, cached artifact loading for RecomX serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RankingArtifact:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: float
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        width = len(self.feature_names)
        if width == 0:
            raise ValueError("Ranking artifact has no feature names.")
        if self.weights.shape != (width,):
            raise ValueError("Ranking weights do not match feature schema.")
        if self.mean.shape != (width,) or self.scale.shape != (width,):
            raise ValueError("Ranking normalization vectors do not match feature schema.")
        if np.any(self.scale <= 0):
            raise ValueError("Ranking scale values must be positive.")
        if not np.isfinite(self.weights).all():
            raise ValueError("Ranking weights contain non-finite values.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Ranking normalization contains non-finite values.")
        if not np.isfinite(self.bias):
            raise ValueError("Ranking bias must be finite.")


@lru_cache(maxsize=4)
def load_ranking_artifact(path: str) -> RankingArtifact:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing ranking artifact: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as payload:
        required = {"weights", "mean", "scale", "bias", "feature_names"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Ranking artifact missing arrays: {sorted(missing)}")
        artifact = RankingArtifact(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            bias=float(np.asarray(payload["bias"]).reshape(-1)[0]),
            feature_names=tuple(str(value) for value in payload["feature_names"].tolist()),
        )
    artifact.validate()
    return artifact
