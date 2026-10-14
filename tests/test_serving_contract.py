from __future__ import annotations
import pytest
from recomx.serving.contract import RecommendationItem, RecommendationRequest, RecommendationResponse

def test_default_request_is_valid() -> None:
    request = RecommendationRequest(user_id=42)
    request.validate()
    assert request.top_k == 20

@pytest.mark.parametrize("top_k", [0, 21])
def test_request_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        RecommendationRequest(user_id=42, top_k=top_k).validate()

def test_request_rejects_invalid_user() -> None:
    with pytest.raises(ValueError, match="user_id"):
        RecommendationRequest(user_id=0).validate()

def test_personalized_response_contract() -> None:
    response = RecommendationResponse(
        user_id=42,
        strategy="personalized",
        recommendations=(
            RecommendationItem(movie_id=10, rank=1, score=0.9, reason="ranked_for_user"),
            RecommendationItem(movie_id=20, rank=2, score=0.8, reason="ranked_for_user"),
        ),
    )
    response.validate()

def test_response_rejects_duplicate_movies() -> None:
    response = RecommendationResponse(
        user_id=42,
        strategy="cold_start_popularity",
        recommendations=(
            RecommendationItem(movie_id=10, rank=1, score=1.0, reason="popular"),
            RecommendationItem(movie_id=10, rank=2, score=0.9, reason="popular"),
        ),
    )
    with pytest.raises(ValueError, match="unique"):
        response.validate()
