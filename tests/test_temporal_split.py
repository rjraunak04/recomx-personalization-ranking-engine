"""Tests for leakage-safe RecomX temporal splitting."""

import pandas as pd
import pytest

from recomx.data.split import (
    TemporalSplitConfig,
    assert_temporal_integrity,
    temporal_split,
)


def synthetic_ratings() -> pd.DataFrame:
    """Create deterministic temporal interactions."""

    timestamps = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "userId": [(index % 10) + 1 for index in range(100)],
            "movieId": [(index % 20) + 1 for index in range(100)],
            "rating": [4.0 for _ in range(100)],
            "timestamp": timestamps,
        }
    )


def test_config_requires_sum_one() -> None:
    config = TemporalSplitConfig(
        train_fraction=0.7,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_config_requires_positive_fractions() -> None:
    config = TemporalSplitConfig(
        train_fraction=0.9,
        validation_fraction=0.1,
        test_fraction=0.0,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_temporal_split_preserves_rows() -> None:
    ratings = synthetic_ratings()

    result = temporal_split(ratings)

    total = len(result.train) + len(result.validation) + len(result.test)

    assert total == len(ratings)


def test_temporal_order_is_strict() -> None:
    result = temporal_split(synthetic_ratings())

    assert result.train["timestamp"].max() < result.validation["timestamp"].min()

    assert result.validation["timestamp"].max() < result.test["timestamp"].min()


def test_expected_split_sizes() -> None:
    result = temporal_split(synthetic_ratings())

    assert len(result.train) == 80
    assert len(result.validation) == 10
    assert len(result.test) == 10


def test_temporal_integrity_passes() -> None:
    result = temporal_split(synthetic_ratings())

    assert_temporal_integrity(result)


def test_same_timestamp_not_split() -> None:
    ratings = synthetic_ratings()

    ratings.loc[
        79:81,
        "timestamp",
    ] = pd.Timestamp(
        "2020-03-20",
        tz="UTC",
    )

    result = temporal_split(ratings)

    train_times = set(result.train["timestamp"])

    validation_times = set(result.validation["timestamp"])

    test_times = set(result.test["timestamp"])

    assert not (train_times & validation_times)

    assert not (validation_times & test_times)

    assert not (train_times & test_times)


def test_missing_timestamp_fails() -> None:
    ratings = synthetic_ratings()

    ratings.loc[
        0,
        "timestamp",
    ] = pd.NaT

    with pytest.raises(ValueError):
        temporal_split(ratings)


def test_empty_dataset_fails() -> None:
    empty = synthetic_ratings().iloc[0:0]

    with pytest.raises(ValueError):
        temporal_split(empty)


def test_cold_start_is_measured() -> None:
    ratings = synthetic_ratings()

    ratings.loc[
        95:99,
        "userId",
    ] = 999

    ratings.loc[
        95:99,
        "movieId",
    ] = 999

    result = temporal_split(ratings)

    assert result.summary.test_cold_users_vs_history == 1

    assert result.summary.test_cold_items_vs_history == 1
