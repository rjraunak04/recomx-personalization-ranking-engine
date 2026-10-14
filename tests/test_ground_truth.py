"""Tests for leakage-safe validation ground truth."""

import pandas as pd
import pytest

from recomx.evaluation.ground_truth import (
    build_validation_ground_truth,
)


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 10, 12],
            "rating": [5.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 3, 3, 4],
            "movieId": [10, 13, 14, 15, 16, 17],
            "rating": [5.0, 4.5, 3.0, 4.0, 2.0, 3.5],
        }
    )


def test_seen_relevant_item_is_removed(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    pairs = set(
        result.ground_truth[["userId", "movieId"]].itertuples(
            index=False,
            name=None,
        )
    )

    assert (1, 10) not in pairs
    assert (1, 13) in pairs


def test_non_relevant_validation_items_are_excluded(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    assert (result.ground_truth["rating"] >= 4.0).all()


def test_cold_relevant_user_is_evaluable(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    user_three = result.cohorts.loc[result.cohorts["userId"] == 3].iloc[0]

    assert bool(user_three["is_cold_user"])
    assert bool(user_three["is_evaluable"])


def test_warm_user_is_identified(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    user_one = result.cohorts.loc[result.cohorts["userId"] == 1].iloc[0]

    assert bool(user_one["is_warm_user"])
    assert not bool(user_one["is_cold_user"])


def test_user_without_relevant_target_not_evaluable(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    user_four = result.cohorts.loc[result.cohorts["userId"] == 4].iloc[0]

    assert not bool(user_four["has_relevant_validation_target"])
    assert not bool(user_four["is_evaluable"])


def test_seen_only_relevant_user_becomes_not_evaluable(
    train: pd.DataFrame,
) -> None:
    validation = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [5.0],
        }
    )

    result = build_validation_ground_truth(
        train,
        validation,
    )

    assert result.ground_truth.empty
    assert result.summary.seen_relevant_interactions_removed == 1
    assert result.summary.evaluable_warm_users == 0


def test_summary_counts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    summary = result.summary

    assert summary.train_users == 2
    assert summary.validation_users == 4
    assert summary.warm_validation_users == 2
    assert summary.cold_validation_users == 2
    assert summary.validation_relevant_users == 2
    assert summary.evaluable_warm_users == 1
    assert summary.evaluable_cold_users == 1
    assert summary.users_without_relevant_targets == 2
    assert summary.relevant_validation_interactions == 3
    assert summary.seen_relevant_interactions_removed == 1
    assert summary.ground_truth_interactions == 2
    assert summary.ground_truth_items == 2


def test_ground_truth_is_deterministically_sorted(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    result = build_validation_ground_truth(
        train,
        validation,
    )

    expected = result.ground_truth.sort_values(
        ["userId", "movieId"],
        kind="mergesort",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result.ground_truth,
        expected,
    )


def test_missing_columns_fail(
    validation: pd.DataFrame,
) -> None:
    bad_train = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_validation_ground_truth(
            bad_train,
            validation,
        )


def test_empty_validation_fails(
    train: pd.DataFrame,
) -> None:
    empty_validation = pd.DataFrame(
        columns=[
            "userId",
            "movieId",
            "rating",
        ]
    )

    with pytest.raises(
        ValueError,
        match="validation cannot be empty",
    ):
        build_validation_ground_truth(
            train,
            empty_validation,
        )
