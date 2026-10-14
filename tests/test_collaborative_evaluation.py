"""Contract tests for Day-03 collaborative offline evaluation."""

import pandas as pd
import pytest

from recomx.evaluation.metrics import evaluate_recommendations


@pytest.fixture
def truth():
    return pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 30]})


@pytest.fixture
def recs():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 99, 88, 30],
            "recommendation_rank": [1, 2, 1, 2],
        }
    )


def test_same_frozen_metrics_are_used(truth, recs):
    user, aggregate = evaluate_recommendations(recs, truth, cutoffs=(1, 2))
    assert len(user) == 4
    assert aggregate["k"].tolist() == [1, 2]


def test_precision_recall_ndcg_map_hit_rate_present(truth, recs):
    _, aggregate = evaluate_recommendations(recs, truth, cutoffs=(2,))
    required = {"precision", "recall", "ndcg", "map", "hit_rate"}
    assert required.issubset(aggregate.columns)


def test_all_metrics_are_bounded(truth, recs):
    _, aggregate = evaluate_recommendations(recs, truth, cutoffs=(1, 2))
    for column in ["precision", "recall", "ndcg", "map", "hit_rate"]:
        assert aggregate[column].between(0.0, 1.0).all()


def test_missing_user_is_rejected(truth, recs):
    with pytest.raises(ValueError, match="Recommendations missing"):
        evaluate_recommendations(recs.loc[recs["userId"].eq(1)], truth, cutoffs=(2,))


def test_duplicate_recommendation_is_rejected(truth, recs):
    bad = recs.copy()
    bad.loc[1, "movieId"] = 10
    with pytest.raises(ValueError, match="Duplicate recommended items"):
        evaluate_recommendations(bad, truth, cutoffs=(2,))
