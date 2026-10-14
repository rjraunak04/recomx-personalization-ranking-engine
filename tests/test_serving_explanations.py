from __future__ import annotations

import pandas as pd
import pytest

from recomx.serving.contract import RecommendationItem, RecommendationResponse
from recomx.serving.explanations import build_movie_metadata, enrich_response


def _response(reason: str = "ml_ranked_personalized") -> RecommendationResponse:
    return RecommendationResponse(
        user_id=7,
        strategy="personalized",
        recommendations=(
            RecommendationItem(movie_id=10, rank=1, score=0.9, reason=reason),
            RecommendationItem(movie_id=20, rank=2, score=0.8, reason=reason),
        ),
    )


def test_metadata_enrichment_preserves_ranking() -> None:
    metadata = {10: {"title": "A", "genres": "Drama"}, 20: {"title": "B", "genres": "Comedy"}}
    result = enrich_response(_response(), metadata)
    assert [item["movie_id"] for item in result["recommendations"]] == [10, 20]
    assert [item["score"] for item in result["recommendations"]] == [0.9, 0.8]
    assert result["recommendations"][0]["title"] == "A"


def test_personalized_explanation_is_user_facing() -> None:
    item = enrich_response(_response(), {})["recommendations"][0]
    assert item["explanation"] == "Ranked for you from your personalized candidate set."


def test_cold_start_explanation() -> None:
    response = RecommendationResponse(
        user_id=99,
        strategy="cold_start_popularity",
        recommendations=(
            RecommendationItem(
                movie_id=10,
                rank=1,
                score=100.0,
                reason="global_popularity_cold_start",
            ),
        ),
    )
    item = enrich_response(response, {})["recommendations"][0]
    assert item["explanation"] == "Popular with users while we learn your preferences."


def test_missing_metadata_has_safe_defaults() -> None:
    item = enrich_response(_response(), {})["recommendations"][0]
    assert item["title"] == "Unknown title"
    assert item["genres"] == "(no genres listed)"


def test_metadata_schema_validation() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_movie_metadata(pd.DataFrame({"movieId": [1], "title": ["A"]}))
