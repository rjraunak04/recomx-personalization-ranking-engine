"""Deterministic non-ML ranking baseline for RecomX Day-05."""

from __future__ import annotations

import pandas as pd


def rank_retrieval_baseline(features: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Rank candidates using frozen hybrid retrieval score before ML ranking."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    required = {"userId", "movieId", "retrieval_score"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Features missing columns: {sorted(missing)}")
    if features.duplicated(["userId", "movieId"]).any():
        raise ValueError("Features contain duplicate user-item pairs.")

    ranked = features.sort_values(
        ["userId", "retrieval_score", "movieId"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["ranking_score"] = ranked["retrieval_score"].astype("float32")
    ranked["recommendation_rank"] = ranked.groupby("userId").cumcount() + 1
    ranked = ranked.loc[ranked["recommendation_rank"].le(top_k)].copy()
    ranked["recommendation_rank"] = ranked["recommendation_rank"].astype("int16")
    return ranked.reset_index(drop=True)
