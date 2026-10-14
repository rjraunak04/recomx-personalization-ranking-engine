from __future__ import annotations

import pandas as pd
import pytest

from recomx.ranking.dataset import build_ranking_dataset, ranking_dataset_summary


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 20, 21],
            "score": [0.9, 0.8, 0.7, 0.6],
            "candidate_rank": [1, 2, 1, 2],
            "sources": ["a", "b", "a|b", "b"],
        }
    )


def test_labels_are_attached_by_user_item_identity() -> None:
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [11, 20]})
    frame = build_ranking_dataset(_candidates(), truth)
    assert frame["label"].tolist() == [0, 1, 1, 0]


def test_ground_truth_outside_candidate_pool_does_not_create_rows() -> None:
    truth = pd.DataFrame({"userId": [1], "movieId": [999]})
    frame = build_ranking_dataset(_candidates(), truth)
    assert len(frame) == 4
    assert int(frame["label"].sum()) == 0


def test_duplicate_candidates_are_rejected() -> None:
    candidates = pd.concat([_candidates(), _candidates().iloc[[0]]], ignore_index=True)
    truth = pd.DataFrame({"userId": [1], "movieId": [10]})
    with pytest.raises(ValueError, match="duplicate"):
        build_ranking_dataset(candidates, truth)


def test_summary_reports_class_balance() -> None:
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [11, 20]})
    summary = ranking_dataset_summary(build_ranking_dataset(_candidates(), truth))
    assert summary == {
        "rows": 4,
        "users": 2,
        "positive_rows": 2,
        "negative_rows": 2,
        "positive_rate": 0.5,
    }
