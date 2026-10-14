from __future__ import annotations

import pandas as pd
import pytest

from recomx.ranking.baseline import rank_retrieval_baseline


def test_baseline_orders_score_descending_and_movie_id_breaks_ties() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "movieId": [12, 10, 11, 20],
            "retrieval_score": [0.5, 0.5, 0.9, 0.1],
        }
    )
    result = rank_retrieval_baseline(frame, top_k=2)
    user_one = result.loc[result["userId"].eq(1)]
    assert user_one["movieId"].tolist() == [11, 10]
    assert user_one["recommendation_rank"].tolist() == [1, 2]


def test_baseline_respects_top_k() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [1, 2, 3],
            "retrieval_score": [0.1, 0.3, 0.2],
        }
    )
    assert len(rank_retrieval_baseline(frame, top_k=2)) == 2


def test_invalid_top_k_is_rejected() -> None:
    frame = pd.DataFrame(
        {"userId": [1], "movieId": [1], "retrieval_score": [1.0]}
    )
    with pytest.raises(ValueError, match="positive"):
        rank_retrieval_baseline(frame, top_k=0)
