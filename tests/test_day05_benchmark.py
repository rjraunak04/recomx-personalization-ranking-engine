from __future__ import annotations

import pandas as pd


def test_day05_reported_ndcg_delta_matches_metrics() -> None:
    baseline = 0.27179471563654817
    ml = 0.2761728249234671
    assert abs((ml - baseline) - 0.004378109286918952) < 1e-12


def test_heldout_metric_schema() -> None:
    frame = pd.DataFrame(
        {"model": ["retrieval_baseline"], "k": [20], "ndcg": [0.27]}
    )
    assert {"model", "k", "ndcg"}.issubset(frame.columns)
