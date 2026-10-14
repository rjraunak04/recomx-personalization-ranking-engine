"""Movie metadata enrichment and user-facing recommendation explanations."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from recomx.serving.contract import RecommendationResponse

_REASON_TEXT = {
    "ml_ranked_personalized": "Ranked for you from your personalized candidate set.",
    "global_popularity_cold_start": "Popular with users while we learn your preferences.",
}


def build_movie_metadata(movies: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Build a validated movie metadata lookup from MovieLens movies."""
    required = {"movieId", "title", "genres"}
    missing = required.difference(movies.columns)
    if missing:
        raise ValueError(f"Movie metadata missing columns: {sorted(missing)}")
    frame = movies.loc[:, ["movieId", "title", "genres"]].drop_duplicates("movieId")
    return {
        int(row.movieId): {"title": str(row.title), "genres": str(row.genres)}
        for row in frame.itertuples(index=False)
    }


def enrich_response(
    response: RecommendationResponse,
    metadata: dict[int, dict[str, str]],
) -> dict[str, object]:
    """Attach display metadata without changing rank, score, or strategy."""
    response.validate()
    recommendations: list[dict[str, object]] = []
    for item in response.recommendations:
        record = asdict(item)
        movie = metadata.get(item.movie_id, {})
        record["title"] = movie.get("title", "Unknown title")
        record["genres"] = movie.get("genres", "(no genres listed)")
        record["explanation"] = _REASON_TEXT.get(item.reason, item.reason)
        recommendations.append(record)
    return {
        "user_id": response.user_id,
        "strategy": response.strategy,
        "recommendations": recommendations,
    }
