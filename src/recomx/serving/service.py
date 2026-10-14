"""Framework-independent recommendation orchestration for RecomX."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.serving.artifacts import RankingArtifact
from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


def _predict_scores(frame: pd.DataFrame, artifact: RankingArtifact) -> np.ndarray:
    missing = set(artifact.feature_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Serving candidates missing ranking features: {sorted(missing)}")
    if artifact.feature_names != tuple(FEATURE_COLUMNS):
        raise ValueError("Serving artifact feature schema differs from ranking schema.")
    x = frame.loc[:, artifact.feature_names].to_numpy(dtype=np.float64)
    x = (x - artifact.mean) / artifact.scale
    logits = np.clip(x @ artifact.weights + artifact.bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def recommend_known_user(
    request: RecommendationRequest,
    candidates: pd.DataFrame,
    artifact: RankingArtifact,
    seen_movie_ids: set[int] | frozenset[int],
) -> RecommendationResponse:
    """Rank a known user's precomputed candidate pool and filter seen items."""
    request.validate()
    artifact.validate()

    required = {"userId", "movieId", *FEATURE_COLUMNS}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Serving candidates missing columns: {sorted(missing)}")

    frame = candidates.loc[candidates["userId"].eq(request.user_id)].copy()
    if frame.empty:
        raise ValueError(f"No personalized candidates for known user {request.user_id}.")

    frame = frame.loc[~frame["movieId"].isin(seen_movie_ids)].copy()
    if frame.empty:
        raise ValueError(f"No unseen candidates remain for known user {request.user_id}.")

    frame["serving_score"] = _predict_scores(frame, artifact)
    frame = frame.sort_values(
        ["serving_score", "movieId"],
        ascending=[False, True],
        kind="mergesort",
    ).drop_duplicates("movieId", keep="first")
    frame = frame.head(request.top_k).reset_index(drop=True)

    items = tuple(
        RecommendationItem(
            movie_id=int(row.movieId),
            rank=rank,
            score=float(row.serving_score),
            reason="ml_ranked_personalized",
        )
        for rank, row in enumerate(frame.itertuples(index=False), start=1)
    )
    response = RecommendationResponse(
        user_id=request.user_id,
        strategy="personalized",
        recommendations=items,
    )
    response.validate()
    return response
