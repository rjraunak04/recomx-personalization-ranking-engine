"""Tests for Day-03 collaborative diagnostic segmentation."""

import pandas as pd
import pytest

from recomx.evaluation.diagnostics import evaluate_by_cohort, evaluate_by_strategy


def _truth():
    return pd.DataFrame({"userId": [1, 2], "movieId": [10, 20]})


def _recs():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 99, 88, 20],
            "recommendation_rank": [1, 2, 1, 2],
            "strategy": ["item_knn", "item_knn", "popularity_fallback", "popularity_fallback"],
        }
    )


def test_strategy_metrics_keep_routes_separate():
    result = evaluate_by_strategy(_recs(), _truth(), cutoffs=(2,))
    assert set(result["strategy"]) == {"item_knn", "popularity_fallback"}
    assert result["users"].sum() == 2


def test_cohort_metrics_keep_warm_and_cold_separate():
    cohorts = pd.DataFrame(
        {
            "userId": [1, 2],
            "is_warm_user": [True, False],
            "is_cold_user": [False, True],
            "is_evaluable": [True, True],
        }
    )
    result = evaluate_by_cohort(_recs(), _truth(), cohorts, cutoffs=(2,))
    assert set(result["segment"]) == {"warm", "cold"}
    assert result["users"].sum() == 2


def test_strategy_rejects_mixed_route_for_user():
    recs = _recs()
    recs.loc[1, "strategy"] = "popularity_fallback"
    with pytest.raises(ValueError, match="multiple strategies"):
        evaluate_by_strategy(recs, _truth(), cutoffs=(2,))


def test_strategy_metrics_are_bounded():
    result = evaluate_by_strategy(_recs(), _truth(), cutoffs=(1, 2))
    for column in ["precision", "recall", "ndcg", "map", "hit_rate"]:
        assert result[column].between(0.0, 1.0).all()
