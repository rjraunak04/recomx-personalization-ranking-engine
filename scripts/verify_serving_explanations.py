"""Verify Day-06 CP-06 metadata and explanation enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.serving.contract import RecommendationItem, RecommendationResponse
from recomx.serving.explanations import build_movie_metadata, enrich_response

MOVIES = Path("data/interim/movies.parquet")


def main() -> None:
    if not MOVIES.is_file():
        raise FileNotFoundError(f"Missing prerequisite artifact: {MOVIES}")
    movies = pd.read_parquet(MOVIES, columns=["movieId", "title", "genres"])
    metadata = build_movie_metadata(movies)
    movie_ids = movies["movieId"].astype(int).head(3).tolist()
    response = RecommendationResponse(
        user_id=7,
        strategy="personalized",
        recommendations=tuple(
            RecommendationItem(
                movie_id=movie_id,
                rank=rank,
                score=1.0 / rank,
                reason="ml_ranked_personalized",
            )
            for rank, movie_id in enumerate(movie_ids, start=1)
        ),
    )
    enriched = enrich_response(response, metadata)
    recommendations = enriched["recommendations"]
    summary = {
        "recommendations": len(recommendations),
        "metadata_lookup_items": len(metadata),
        "titles_present": all(item["title"] != "Unknown title" for item in recommendations),
        "genres_present": all("genres" in item for item in recommendations),
        "ranking_preserved": [item["rank"] for item in recommendations] == [1, 2, 3],
        "scores_preserved": [item["score"] for item in recommendations]
        == [1.0, 0.5, 1.0 / 3.0],
        "locked_test_used": False,
    }
    if not all(
        [
            summary["titles_present"],
            summary["genres_present"],
            summary["ranking_preserved"],
            summary["scores_preserved"],
        ]
    ):
        raise ValueError(f"Explanation verification failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-06 EXPLANATION & METADATA: PASS")


if __name__ == "__main__":
    main()
