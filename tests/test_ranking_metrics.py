"""Tests for RecomX Top-K ranking metrics."""

from math import log2

import pandas as pd
import pytest

from recomx.evaluation.metrics import (
    average_precision_at_k,
    build_ground_truth_map,
    build_recommendation_map,
    evaluate_recommendations,
    evaluate_user_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 8}

    assert precision_at_k(
        recommended,
        relevant,
        5,
    ) == pytest.approx(2 / 5)


def test_recall_at_k() -> None:
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 8}

    assert recall_at_k(
        recommended,
        relevant,
        5,
    ) == pytest.approx(2 / 3)


def test_hit_rate_at_k() -> None:
    assert (
        hit_rate_at_k(
            [1, 2, 3],
            {3, 8},
            3,
        )
        == 1.0
    )

    assert (
        hit_rate_at_k(
            [1, 2],
            {3, 8},
            2,
        )
        == 0.0
    )


def test_ndcg_at_k() -> None:
    recommended = [1, 2, 3]
    relevant = {2, 3}

    expected_dcg = 1 / log2(3) + 1 / log2(4)

    expected_idcg = 1 / log2(2) + 1 / log2(3)

    assert ndcg_at_k(
        recommended,
        relevant,
        3,
    ) == pytest.approx(expected_dcg / expected_idcg)


def test_average_precision_at_k() -> None:
    recommended = [1, 2, 3, 4]
    relevant = {2, 4}

    expected = ((1 / 2) + (2 / 4)) / 2

    assert average_precision_at_k(
        recommended,
        relevant,
        4,
    ) == pytest.approx(expected)


def test_metrics_with_no_relevant_items() -> None:
    assert (
        recall_at_k(
            [1, 2],
            set(),
            2,
        )
        == 0.0
    )

    assert (
        ndcg_at_k(
            [1, 2],
            set(),
            2,
        )
        == 0.0
    )

    assert (
        average_precision_at_k(
            [1, 2],
            set(),
            2,
        )
        == 0.0
    )


def test_precision_penalizes_short_lists() -> None:
    assert precision_at_k(
        [1, 2],
        {1, 2},
        5,
    ) == pytest.approx(2 / 5)


def test_evaluate_user_at_k() -> None:
    metrics = evaluate_user_at_k(
        user_id=42,
        recommended=[1, 2, 3],
        relevant={2, 3},
        k=3,
    )

    assert metrics.user_id == 42
    assert metrics.hits == 2
    assert metrics.relevant_items == 2
    assert metrics.recommended_items == 3
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.hit_rate == 1.0


def test_build_ground_truth_map() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 20, 30],
        }
    )

    result = build_ground_truth_map(frame)

    assert result == {
        1: {10, 20},
        2: {30},
    }


def test_build_recommendation_map_respects_rank() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [30, 10, 20],
            "recommendation_rank": [
                3,
                1,
                2,
            ],
        }
    )

    result = build_recommendation_map(frame)

    assert result[1] == [
        10,
        20,
        30,
    ]


def test_duplicate_recommendation_rank_fails() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 20],
            "recommendation_rank": [1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate recommendation ranks",
    ):
        build_recommendation_map(frame)


def test_duplicate_recommended_item_fails() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 10],
            "recommendation_rank": [1, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate recommended items",
    ):
        build_recommendation_map(frame)


def test_end_to_end_aggregate_metrics() -> None:
    ground_truth = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 20, 30],
        }
    )

    recommendations = pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
            ],
            "movieId": [
                10,
                99,
                88,
                30,
            ],
            "recommendation_rank": [
                1,
                2,
                1,
                2,
            ],
        }
    )

    user_metrics, aggregate = evaluate_recommendations(
        recommendations,
        ground_truth,
        cutoffs=(2,),
    )

    assert len(user_metrics) == 2

    row = aggregate.iloc[0]

    assert row["users"] == 2
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.75)
    assert row["hit_rate"] == 1.0


def test_missing_ground_truth_user_fails() -> None:
    ground_truth = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
        }
    )

    recommendations = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "recommendation_rank": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Recommendations missing",
    ):
        evaluate_recommendations(
            recommendations,
            ground_truth,
            cutoffs=(1,),
        )


@pytest.mark.parametrize(
    "k",
    [0, -1],
)
def test_invalid_k_fails(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        precision_at_k(
            [1],
            {1},
            k,
        )
