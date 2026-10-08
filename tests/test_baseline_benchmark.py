"""Tests for the RecomX Day-02 benchmark integrity checks."""

import pandas as pd
import pytest

from scripts.run_baseline_benchmark import (
    _validate_comparison,
    _validate_recommendations,
)


def _recommendations() -> pd.DataFrame:
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
                20,
                30,
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


def test_recommendation_integrity_passes() -> None:
    result = _validate_recommendations(
        frame=_recommendations(),
        expected_users={1, 2},
        max_k=2,
        model_name="example",
    )

    assert result == {
        "users": 2,
        "rows": 4,
        "unique_items": 4,
    }


def test_missing_user_fails() -> None:
    frame = _recommendations().loc[lambda value: value["userId"].eq(1)]

    with pytest.raises(
        ValueError,
        match="user set",
    ):
        _validate_recommendations(
            frame=frame,
            expected_users={1, 2},
            max_k=2,
            model_name="example",
        )


def test_duplicate_item_fails() -> None:
    frame = _recommendations()
    frame.loc[
        frame["userId"].eq(1),
        "movieId",
    ] = 10

    with pytest.raises(
        ValueError,
        match="duplicate user-item",
    ):
        _validate_recommendations(
            frame=frame,
            expected_users={1, 2},
            max_k=2,
            model_name="example",
        )


def test_incomplete_top_k_fails() -> None:
    frame = _recommendations().drop(index=1)

    with pytest.raises(
        ValueError,
        match="exactly Top-2",
    ):
        _validate_recommendations(
            frame=frame,
            expected_users={1, 2},
            max_k=2,
            model_name="example",
        )


def test_comparison_contract_passes() -> None:
    frame = pd.DataFrame(
        {
            "model": [
                "popularity",
                "genre_personalized",
            ],
            "k": [
                10,
                10,
            ],
            "users": [
                2,
                2,
            ],
            "precision": [
                0.2,
                0.1,
            ],
            "recall": [
                0.3,
                0.2,
            ],
            "ndcg": [
                0.4,
                0.3,
            ],
            "map": [
                0.2,
                0.1,
            ],
            "hit_rate": [
                0.5,
                0.4,
            ],
        }
    )

    _validate_comparison(
        comparison=frame,
        cutoffs=(10,),
        evaluation_users=2,
    )


def test_metric_outside_unit_interval_fails() -> None:
    frame = pd.DataFrame(
        {
            "model": [
                "popularity",
                "genre_personalized",
            ],
            "k": [
                10,
                10,
            ],
            "users": [
                2,
                2,
            ],
            "precision": [
                1.2,
                0.1,
            ],
            "recall": [
                0.3,
                0.2,
            ],
            "ndcg": [
                0.4,
                0.3,
            ],
            "map": [
                0.2,
                0.1,
            ],
            "hit_rate": [
                0.5,
                0.4,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"outside \[0, 1\]",
    ):
        _validate_comparison(
            comparison=frame,
            cutoffs=(10,),
            evaluation_users=2,
        )
