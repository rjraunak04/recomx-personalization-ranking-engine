"""Tests for recommendation-specific EDA utilities."""

import pandas as pd
import pytest

from recomx.data.eda import (
    build_summary,
    calculate_sparsity,
    item_popularity,
    popularity_share,
    rating_distribution,
    user_activity,
)


@pytest.fixture
def ratings() -> pd.DataFrame:
    """Small deterministic interaction dataset."""

    return pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
                3,
            ],
            "movieId": [
                10,
                20,
                10,
                30,
                10,
            ],
            "rating": [
                4.0,
                3.5,
                5.0,
                2.0,
                4.5,
            ],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-02-01",
                    "2020-02-02",
                    "2020-03-01",
                ],
                utc=True,
            ),
        }
    )


def test_sparsity(
    ratings: pd.DataFrame,
) -> None:
    possible, observed, density, sparsity = calculate_sparsity(ratings)

    assert possible == 9
    assert observed == 5
    assert density == pytest.approx(5 / 9)
    assert sparsity == pytest.approx(4 / 9)


def test_user_activity(
    ratings: pd.DataFrame,
) -> None:
    activity = user_activity(ratings)

    assert activity.loc[1] == 2
    assert activity.loc[2] == 2
    assert activity.loc[3] == 1


def test_item_popularity(
    ratings: pd.DataFrame,
) -> None:
    popularity = item_popularity(ratings)

    assert popularity.loc[10] == 3
    assert popularity.loc[20] == 1
    assert popularity.loc[30] == 1


def test_popularity_share(
    ratings: pd.DataFrame,
) -> None:
    popularity = item_popularity(ratings)

    share = popularity_share(
        popularity,
        1 / 3,
    )

    assert share == pytest.approx(3 / 5)


def test_invalid_popularity_fraction(
    ratings: pd.DataFrame,
) -> None:
    popularity = item_popularity(ratings)

    with pytest.raises(ValueError):
        popularity_share(
            popularity,
            0.0,
        )


def test_rating_distribution(
    ratings: pd.DataFrame,
) -> None:
    distribution = rating_distribution(ratings)

    assert distribution["interaction_count"].sum() == 5

    assert distribution["share"].sum() == (pytest.approx(1.0))


def test_summary(
    ratings: pd.DataFrame,
) -> None:
    movies = pd.DataFrame(
        {
            "movieId": [
                10,
                20,
                30,
                40,
            ],
            "title": [
                "A",
                "B",
                "C",
                "D",
            ],
            "genres": [
                "Drama",
                "Comedy",
                "Action",
                "Drama",
            ],
        }
    )

    summary = build_summary(
        ratings,
        movies,
    )

    assert summary.interactions == 5
    assert summary.unique_users == 3
    assert summary.unique_items == 3
    assert summary.catalogue_items == 4

    assert summary.catalogue_rating_coverage == pytest.approx(0.75)

    assert summary.sparsity == (pytest.approx(4 / 9))
