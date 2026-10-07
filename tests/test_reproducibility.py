"""Tests for RecomX deterministic data reproducibility."""

import pandas as pd
import pytest

from recomx.data.reproducibility import (
    build_reproducibility_manifest,
    dataframe_sha256,
    partition_fingerprint,
)
from recomx.data.split import temporal_split


def sample_ratings() -> pd.DataFrame:
    """Create deterministic interactions."""

    return pd.DataFrame(
        {
            "userId": [(index % 5) + 1 for index in range(50)],
            "movieId": [(index % 9) + 1 for index in range(50)],
            "rating": [float((index % 5) + 1) for index in range(50)],
            "timestamp": pd.date_range(
                "2020-01-01",
                periods=50,
                freq="h",
                tz="UTC",
            ),
        }
    )


def test_hash_is_deterministic() -> None:
    ratings = sample_ratings()

    first = dataframe_sha256(ratings)

    second = dataframe_sha256(ratings.copy())

    assert first == second


def test_hash_is_row_order_invariant() -> None:
    ratings = sample_ratings()

    shuffled = ratings.sample(
        frac=1.0,
        random_state=42,
    )

    assert dataframe_sha256(ratings) == dataframe_sha256(shuffled)


def test_hash_changes_when_data_changes() -> None:
    ratings = sample_ratings()

    changed = ratings.copy()

    changed.loc[
        0,
        "rating",
    ] = 0.5

    assert dataframe_sha256(ratings) != dataframe_sha256(changed)


def test_missing_columns_fail() -> None:
    ratings = sample_ratings().drop(columns=["rating"])

    with pytest.raises(ValueError):
        dataframe_sha256(ratings)


def test_empty_partition_fails() -> None:
    ratings = sample_ratings().iloc[0:0]

    with pytest.raises(ValueError):
        partition_fingerprint(
            "train",
            ratings,
        )


def test_manifest_is_repeatable() -> None:
    ratings = sample_ratings()

    first_result = temporal_split(ratings)

    second_result = temporal_split(ratings.copy())

    first_manifest = build_reproducibility_manifest(first_result)

    second_manifest = build_reproducibility_manifest(second_result)

    assert first_manifest.combined_sha256 == second_manifest.combined_sha256

    assert first_manifest.train.sha256 == second_manifest.train.sha256

    assert first_manifest.validation.sha256 == second_manifest.validation.sha256

    assert first_manifest.locked_test.sha256 == second_manifest.locked_test.sha256


def test_manifest_preserves_source_rows() -> None:
    ratings = sample_ratings()

    result = temporal_split(ratings)

    manifest = build_reproducibility_manifest(result)

    assert manifest.source_rows == 50

    assert manifest.train.rows + manifest.validation.rows + manifest.locked_test.rows == 50
