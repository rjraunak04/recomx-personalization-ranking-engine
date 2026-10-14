"""Tests for hybrid retrieval primitives."""

import pandas as pd
import pytest

from recomx.models.hybrid_retrieval import (
    HybridRetrievalConfig,
    candidate_recall,
    reciprocal_rank_fusion,
)


def _source(items):
    return pd.DataFrame(
        {
            "userId": [1] * len(items),
            "movieId": items,
            "recommendation_rank": range(1, len(items) + 1),
        }
    )


def test_config_validation():
    with pytest.raises(ValueError):
        HybridRetrievalConfig(candidate_budget=0).validate()


def test_rrf_deduplicates_and_tracks_sources():
    result = reciprocal_rank_fusion(
        {"a": _source([10, 20]), "b": _source([10, 30])}, {}, HybridRetrievalConfig(3, 60)
    )
    assert result.iloc[0]["movieId"] == 10
    assert result.iloc[0]["sources"] == "a|b"
    assert len(result) == 3


def test_rrf_filters_seen_items():
    result = reciprocal_rank_fusion({"a": _source([10, 20])}, {1: {10}})
    assert result["movieId"].tolist() == [20]


def test_rrf_is_deterministic_on_ties():
    result = reciprocal_rank_fusion({"a": _source([20]), "b": _source([10])}, {})
    assert result["movieId"].tolist() == [10, 20]


def test_candidate_recall():
    candidates = pd.DataFrame(
        {"userId": [1, 1], "movieId": [10, 20], "candidate_rank": [1, 2]}
    )
    truth = pd.DataFrame({"userId": [1, 1], "movieId": [10, 30]})
    assert candidate_recall(candidates, truth, 2) == pytest.approx(0.5)
