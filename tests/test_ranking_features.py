from __future__ import annotations

import pandas as pd

from recomx.ranking.features import FEATURE_COLUMNS, add_ranking_features


def test_ranking_features_are_train_derived_and_complete() -> None:
    ranking = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
            "score": [0.5, 0.25],
            "candidate_rank": [1, 2],
            "sources": ["item_knn|popularity", "latent_svd"],
            "label": [1, 0],
        }
    )
    train = pd.DataFrame(
        {
            "userId": [1, 1, 3],
            "movieId": [10, 11, 20],
            "rating": [4.0, 2.0, 5.0],
        }
    )
    result = add_ranking_features(ranking, train)
    assert not result[list(FEATURE_COLUMNS)].isna().any().any()
    first = result.iloc[0]
    assert first["user_history_count"] == 2
    assert first["user_mean_rating"] == 3.0
    assert first["item_interaction_count"] == 1
    assert first["source_item_knn"] == 1
    assert first["source_popularity"] == 1
    assert first["source_count"] == 2


def test_cold_user_and_item_statistics_are_zero_filled() -> None:
    ranking = pd.DataFrame(
        {
            "userId": [99],
            "movieId": [999],
            "score": [0.1],
            "candidate_rank": [5],
            "sources": ["popularity"],
            "label": [0],
        }
    )
    train = pd.DataFrame({"userId": [1], "movieId": [10], "rating": [4.0]})
    result = add_ranking_features(ranking, train).iloc[0]
    assert result["user_history_count"] == 0
    assert result["user_mean_rating"] == 0.0
    assert result["item_interaction_count"] == 0
    assert result["item_mean_rating"] == 0.0
