from __future__ import annotations

import pandas as pd
import pytest

from recomx.serving.contract import RecommendationRequest
from recomx.serving.fallback import recommend_cold_user


def _popularity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [30, 20, 10, 40, 20],
            "popularity_score": [0.8, 0.9, 0.9, 0.7, 0.9],
        }
    )


def test_cold_user_receives_popularity_top_k() -> None:
    response = recommend_cold_user(
        RecommendationRequest(user_id=999_999, top_k=3),
        _popularity(),
    )
    assert response.strategy == "cold_start_popularity"
    assert [item.movie_id for item in response.recommendations] == [10, 20, 30]
    assert [item.rank for item in response.recommendations] == [1, 2, 3]


def test_fallback_is_deterministic() -> None:
    request = RecommendationRequest(user_id=999_999, top_k=4)
    first = recommend_cold_user(request, _popularity())
    second = recommend_cold_user(request, _popularity())
    assert first == second


def test_fallback_requires_enough_unique_items() -> None:
    with pytest.raises(ValueError, match="required"):
        recommend_cold_user(
            RecommendationRequest(user_id=999_999, top_k=5),
            _popularity(),
        )


def test_fallback_validates_schema() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        recommend_cold_user(
            RecommendationRequest(user_id=999_999),
            pd.DataFrame({"movieId": [1, 2]}),
        )
