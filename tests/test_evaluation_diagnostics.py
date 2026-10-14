"""Tests for recommendation diagnostics."""

import pandas as pd
import pytest

from recomx.evaluation.diagnostics import (
    average_recommendation_popularity,
    build_long_tail_items,
    catalogue_coverage,
    cold_item_exposure,
    evaluate_by_cohort,
    evaluate_by_strategy,
    long_tail_share,
    recommendation_coverage,
)


@pytest.fixture
def recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
            ],
            "movieId": [
                10,
                30,
                20,
                40,
            ],
            "recommendation_rank": [
                1,
                2,
                1,
                2,
            ],
        }
    )


@pytest.fixture
def item_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [
                10,
                20,
                30,
                40,
                50,
            ],
            "popularity_score": [
                100,
                80,
                20,
                10,
                5,
            ],
        }
    )


def test_catalogue_coverage(
    recommendations: pd.DataFrame,
) -> None:
    result = catalogue_coverage(
        recommendations,
        {10, 20, 30, 40, 50},
    )

    assert result == pytest.approx(4 / 5)


def test_recommendation_coverage(
    recommendations: pd.DataFrame,
) -> None:
    result = recommendation_coverage(
        recommendations,
        {10, 20, 30, 40},
    )

    assert result == 1.0


def test_build_long_tail_items(
    item_scores: pd.DataFrame,
) -> None:
    result = build_long_tail_items(
        item_scores,
        head_fraction=0.40,
    )

    assert result == {
        30,
        40,
        50,
    }


def test_long_tail_share(
    recommendations: pd.DataFrame,
) -> None:
    result = long_tail_share(
        recommendations,
        {30, 40, 50},
    )

    assert result == pytest.approx(2 / 4)


def test_average_recommendation_popularity(
    recommendations: pd.DataFrame,
    item_scores: pd.DataFrame,
) -> None:
    result = average_recommendation_popularity(
        recommendations,
        item_scores,
    )

    assert result == pytest.approx((100 + 20 + 80 + 10) / 4)


def test_cold_item_exposure(
    recommendations: pd.DataFrame,
) -> None:
    count, share = cold_item_exposure(
        recommendations,
        {10, 20, 30},
    )

    assert count == 1
    assert share == pytest.approx(1 / 4)


def test_cohort_evaluation() -> None:
    ground_truth = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
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
                20,
            ],
            "recommendation_rank": [
                1,
                2,
                1,
                2,
            ],
        }
    )

    cohorts = pd.DataFrame(
        {
            "userId": [1, 2],
            "is_warm_user": [
                True,
                False,
            ],
            "is_cold_user": [
                False,
                True,
            ],
            "is_evaluable": [
                True,
                True,
            ],
        }
    )

    result = evaluate_by_cohort(
        recommendations,
        ground_truth,
        cohorts,
        cutoffs=(2,),
    )

    assert set(result["segment"]) == {
        "warm",
        "cold",
    }

    warm = result.loc[result["segment"].eq("warm")].iloc[0]

    cold = result.loc[result["segment"].eq("cold")].iloc[0]

    assert warm["hit_rate"] == 1.0
    assert cold["hit_rate"] == 1.0


def test_strategy_evaluation() -> None:
    ground_truth = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
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
                20,
            ],
            "recommendation_rank": [
                1,
                2,
                1,
                2,
            ],
            "strategy": [
                "genre_personalized",
                "genre_personalized",
                "popularity_fallback",
                "popularity_fallback",
            ],
        }
    )

    result = evaluate_by_strategy(
        recommendations,
        ground_truth,
        cutoffs=(2,),
    )

    assert set(result["strategy"]) == {
        "genre_personalized",
        "popularity_fallback",
    }


def test_multiple_user_strategies_fail() -> None:
    ground_truth = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
        }
    )

    recommendations = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 20],
            "recommendation_rank": [1, 2],
            "strategy": [
                "genre_personalized",
                "popularity_fallback",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="multiple strategies",
    ):
        evaluate_by_strategy(
            recommendations,
            ground_truth,
            cutoffs=(2,),
        )


def test_invalid_head_fraction_fails(
    item_scores: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        build_long_tail_items(
            item_scores,
            head_fraction=1.0,
        )


def test_empty_catalogue_fails(
    recommendations: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        catalogue_coverage(
            recommendations,
            set(),
        )
