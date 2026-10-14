"""Tests for the Day-03 collaborative benchmark integrity contract."""

import pandas as pd
import pytest

from scripts.run_collaborative_benchmark import _validate_recommendations


def _frame():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 30, 40],
            "recommendation_rank": [1, 2, 1, 2],
            "strategy": ["item_knn", "item_knn", "popularity_fallback", "popularity_fallback"],
        }
    )


def test_integrity_passes():
    result = _validate_recommendations(_frame(), {1, 2}, 2, "model")
    assert result == {"users": 2, "rows": 4, "unique_items": 4}


def test_missing_user_fails():
    with pytest.raises(ValueError, match="user set"):
        _validate_recommendations(_frame().loc[lambda x: x.userId.eq(1)], {1, 2}, 2, "model")


def test_duplicate_item_fails():
    frame = _frame()
    frame.loc[1, "movieId"] = 10
    with pytest.raises(ValueError, match="duplicate user-item"):
        _validate_recommendations(frame, {1, 2}, 2, "model")


def test_incomplete_top_k_fails():
    with pytest.raises(ValueError, match="exactly Top-2"):
        _validate_recommendations(_frame().drop(index=1), {1, 2}, 2, "model")
