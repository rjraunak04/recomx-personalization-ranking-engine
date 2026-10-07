"""Tests for RecomX recommendation data-quality validation."""

import pandas as pd
import pytest

from recomx.data.validate import (
    DataQualityError,
    assert_required_rules,
    validate_movies,
    validate_ratings,
    validate_tags,
)


def valid_ratings() -> pd.DataFrame:
    """Create a valid synthetic ratings table."""

    return pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 20, 10],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ],
                utc=True,
            ),
        }
    )


def test_valid_ratings_pass() -> None:
    results = validate_ratings(valid_ratings())

    assert_required_rules(results)


def test_invalid_rating_domain_fails() -> None:
    frame = valid_ratings()
    frame.loc[0, "rating"] = 6.0

    results = validate_ratings(frame)

    rule = next(result for result in results if result.rule == "ratings_domain")

    assert not rule.passed
    assert rule.value == 1


def test_invalid_half_star_rating_fails() -> None:
    frame = valid_ratings()
    frame.loc[0, "rating"] = 4.2

    results = validate_ratings(frame)

    rule = next(result for result in results if result.rule == "ratings_half_star_scale")

    assert not rule.passed
    assert rule.value == 1


def test_exact_duplicate_rating_fails() -> None:
    frame = valid_ratings()

    frame = pd.concat(
        [
            frame,
            frame.iloc[[0]],
        ],
        ignore_index=True,
    )

    results = validate_ratings(frame)

    rule = next(result for result in results if result.rule == "ratings_exact_duplicates")

    assert not rule.passed
    assert rule.value == 1


def test_repeated_user_movie_is_measured() -> None:
    frame = valid_ratings()

    extra = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [4.5],
            "timestamp": pd.to_datetime(
                ["2020-02-01"],
                utc=True,
            ),
        }
    )

    frame = pd.concat(
        [frame, extra],
        ignore_index=True,
    )

    results = validate_ratings(frame)

    rule = next(result for result in results if result.rule == "ratings_repeated_user_movie_rows")

    assert rule.passed
    assert rule.value == 2


def test_duplicate_movie_ids_fail() -> None:
    movies = pd.DataFrame(
        {
            "movieId": [10, 10],
            "title": ["A", "B"],
            "genres": [
                "Drama",
                "Comedy",
            ],
        }
    )

    results = validate_movies(movies)

    rule = next(result for result in results if result.rule == "movies_unique_ids")

    assert not rule.passed
    assert rule.value == 1


def test_blank_tags_are_nonblocking() -> None:
    tags = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
            "tag": ["", "action"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                ],
                utc=True,
            ),
        }
    )

    results = validate_tags(tags)

    rule = next(result for result in results if result.rule == "tags_nonempty_observation")

    assert rule.passed
    assert rule.value == 1

    assert_required_rules(results)


def test_assert_required_rules_raises() -> None:
    frame = valid_ratings()
    frame.loc[0, "userId"] = -1

    results = validate_ratings(frame)

    with pytest.raises(DataQualityError):
        assert_required_rules(results)
