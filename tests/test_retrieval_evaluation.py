"""Tests for retrieval diagnostics."""

import pandas as pd

from recomx.evaluation.retrieval import overlap_rate, source_contribution


def test_source_contribution():
    frame = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 20],
            "sources": ["item_knn|svd", "content"],
        }
    )
    result = source_contribution(frame)
    assert set(result["source"]) == {"item_knn", "svd", "content"}


def test_overlap_rate():
    left = pd.DataFrame({"userId": [1, 1], "movieId": [10, 20]})
    right = pd.DataFrame({"userId": [1, 1], "movieId": [20, 30]})
    assert overlap_rate(left, right) == 1 / 3
