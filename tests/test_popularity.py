"""Tests for the RecomX popularity baseline."""

import pandas as pd
import pytest

from recomx.models.popularity import (
    PopularityModel,
    build_seen_item_map,
    recommend_for_users,
)


@pytest.fixture
def train() -> pd.DataFrame:
    """Small explicit-feedback training set."""

    return pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                5,
            ],
            "movieId": [
                10,
                20,
                10,
                30,
                10,
                40,
                20,
                50,
            ],
            "rating": [
                5.0,
                4.0,
                4.5,
                5.0,
                4.0,
                2.0,
                3.0,
                4.5,
            ],
        }
    )


def test_fit_uses_only_relevant_feedback(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    scores = model.item_scores.set_index("movieId")

    assert scores.loc[10, "popularity_score"] == 3
    assert scores.loc[20, "popularity_score"] == 1
    assert 40 not in scores.index


def test_popularity_order_is_deterministic(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    assert model.item_scores["movieId"].tolist() == [
        10,
        20,
        30,
        50,
    ]


def test_popularity_rank_starts_at_one(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    assert model.item_scores["popularity_rank"].tolist() == [1, 2, 3, 4]


def test_seen_items_are_filtered(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    recommendations = model.recommend(
        user_id=1,
        seen_items={10, 20},
        k=2,
    )

    assert recommendations["movieId"].tolist() == [
        30,
        50,
    ]


def test_recommendation_rank_is_contiguous(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    recommendations = model.recommend(
        user_id=1,
        seen_items={10},
        k=3,
    )

    assert recommendations["recommendation_rank"].tolist() == [1, 2, 3]


def test_cold_user_receives_global_popularity(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    recommendations = model.recommend(
        user_id=999,
        seen_items=set(),
        k=3,
    )

    assert recommendations["movieId"].tolist() == [
        10,
        20,
        30,
    ]


def test_build_seen_item_map(
    train: pd.DataFrame,
) -> None:
    seen = build_seen_item_map(train)

    assert seen[1] == {10, 20}
    assert seen[2] == {10, 30}


def test_batch_recommendations_filter_seen_items(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)
    seen = build_seen_item_map(train)

    recommendations = recommend_for_users(
        model=model,
        user_ids=[1, 999],
        seen_item_map=seen,
        k=2,
    )

    user_one = recommendations.loc[recommendations["userId"] == 1]

    cold_user = recommendations.loc[recommendations["userId"] == 999]

    assert not set(user_one["movieId"]).intersection(seen[1])

    assert cold_user["movieId"].tolist() == [
        10,
        20,
    ]


def test_invalid_k_fails(
    train: pd.DataFrame,
) -> None:
    model = PopularityModel.fit(train)

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        model.recommend(
            user_id=1,
            k=0,
        )


def test_empty_training_data_fails() -> None:
    frame = pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "rating",
        ]
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        PopularityModel.fit(frame)


def test_missing_columns_fail() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        PopularityModel.fit(frame)
