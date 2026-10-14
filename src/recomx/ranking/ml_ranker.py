"""Dependency-light ML ranker for RecomX Day-05."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recomx.ranking.features import FEATURE_COLUMNS


@dataclass
class LogisticRanker:
    """Standardized logistic ranker trained with deterministic batch gradient descent."""

    learning_rate: float = 0.1
    epochs: int = 80
    l2: float = 1e-3
    weights_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(self, frame: pd.DataFrame) -> "LogisticRanker":
        x = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = frame["label"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) < 2:
            raise ValueError("ML ranker requires both positive and negative labels.")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        x = (x - self.mean_) / self.scale_
        self.weights_ = np.zeros(x.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            logits = np.clip(x @ self.weights_ + self.bias_, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = probabilities - y
            gradient = (x.T @ error) / len(x) + self.l2 * self.weights_
            self.weights_ -= self.learning_rate * gradient
            self.bias_ -= self.learning_rate * float(error.mean())
        return self

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("ML ranker must be fitted before scoring.")
        x = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        x = (x - self.mean_) / self.scale_
        logits = np.clip(x @ self.weights_ + self.bias_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def rank_with_model(
    frame: pd.DataFrame, model: LogisticRanker, top_k: int = 20
) -> pd.DataFrame:
    """Score and deterministically rank each user's frozen candidate set."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    ranked = frame.copy()
    ranked["ranking_score"] = model.predict_score(ranked).astype("float32")
    ranked = ranked.sort_values(
        ["userId", "ranking_score", "movieId"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["recommendation_rank"] = ranked.groupby("userId").cumcount() + 1
    ranked = ranked.loc[ranked["recommendation_rank"].le(top_k)].copy()
    ranked["recommendation_rank"] = ranked["recommendation_rank"].astype("int16")
    return ranked.reset_index(drop=True)
