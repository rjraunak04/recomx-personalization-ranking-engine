"""Deterministic cold-start fallback for RecomX serving."""

from __future__ import annotations

import pandas as pd

from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


def recommend_cold_user(
    request: RecommendationRequest,
    popularity: pd.DataFrame,
) -> RecommendationResponse:
    """Serve deterministic global-popularity recommendations to an unseen user."""
    request.validate()
    required = {"movieId", "popularity_score"}
    missing = required.difference(popularity.columns)
    if missing:
        raise ValueError(f"Popularity fallback missing columns: {sorted(missing)}")

    frame = popularity.loc[:, ["movieId", "popularity_score"]].copy()
    frame = frame.dropna(subset=["movieId", "popularity_score"])
    frame = frame.sort_values(
        ["popularity_score", "movieId"],
        ascending=[False, True],
        kind="mergesort",
    ).drop_duplicates("movieId", keep="first")
    frame = frame.head(request.top_k).reset_index(drop=True)
    if len(frame) < request.top_k:
        raise ValueError(
            f"Popularity fallback has {len(frame)} items; {request.top_k} required."
        )

    items = tuple(
        RecommendationItem(
            movie_id=int(row.movieId),
            rank=rank,
            score=float(row.popularity_score),
            reason="global_popularity_cold_start",
        )
        for rank, row in enumerate(frame.itertuples(index=False), start=1)
    )
    response = RecommendationResponse(
        user_id=request.user_id,
        strategy="cold_start_popularity",
        recommendations=items,
    )
    response.validate()
    return response
