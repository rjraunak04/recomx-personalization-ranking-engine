"""Tests for explicit-feedback relevance handling."""

from dataclasses import replace

import pandas as pd
import pytest

from recomx.evaluation.relevance import (
    RelevancePolicy,
    build_relevance_profile,
)


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Small explicit-feedback interaction table."""

    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 11, 10, 12, 13],
            "rating": [5.0, 3.5, 4.0, 2.0, 4.5],
        }
    )


def test_default_threshold_is_four() -> None:
    assert RelevancePolicy().minimum_rating == 4.0


def test_rating_four_is_relevant(
    interactions: pd.DataFrame,
) -> None:
    mask = RelevancePolicy().mask(interactions)

    assert mask.tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_relevant_interactions_are_filtered(
    interactions: pd.DataFrame,
) -> None:
    relevant = RelevancePolicy().relevant_interactions(interactions)

    assert len(relevant) == 3
    assert relevant["rating"].min() >= 4.0


def test_input_is_not_mutated(
    interactions: pd.DataFrame,
) -> None:
    original = interactions.copy(deep=True)

    RelevancePolicy().relevant_interactions(interactions)

    pd.testing.assert_frame_equal(
        interactions,
        original,
    )


def test_missing_rating_column_fails() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required rating column",
    ):
        RelevancePolicy().mask(frame)


def test_missing_rating_value_fails() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 11],
            "rating": [4.0, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        RelevancePolicy().mask(frame)


def test_invalid_threshold_fails() -> None:
    policy = replace(
        RelevancePolicy(),
        minimum_rating=5.5,
    )

    with pytest.raises(
        ValueError,
        match="MovieLens rating scale",
    ):
        policy.validate()


def test_profile_counts(
    interactions: pd.DataFrame,
) -> None:
    profile = build_relevance_profile(interactions)

    assert profile.total_interactions == 5
    assert profile.relevant_interactions == 3
    assert profile.non_relevant_interactions == 2
    assert profile.relevance_rate == pytest.approx(0.6)


def test_profile_user_counts(
    interactions: pd.DataFrame,
) -> None:
    profile = build_relevance_profile(interactions)

    assert profile.total_users == 3
    assert profile.relevant_users == 3


def test_profile_item_counts(
    interactions: pd.DataFrame,
) -> None:
    profile = build_relevance_profile(interactions)

    assert profile.total_items == 4

    # Relevant interactions are:
    # movie 10 -> ratings 5.0 and 4.0
    # movie 13 -> rating 4.5
    # Therefore there are 3 relevant interactions
    # but only 2 unique relevant items.
    assert profile.relevant_items == 2


def test_empty_profile_fails() -> None:
    frame = pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "rating",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty interaction",
    ):
        build_relevance_profile(frame)
