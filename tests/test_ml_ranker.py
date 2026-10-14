from __future__ import annotations

import pandas as pd
import pytest

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.ranking.ml_ranker import LogisticRanker, rank_with_model


def _frame() -> pd.DataFrame:
    rows = []
    for user in (1, 2):
        for rank in range(1, 5):
            row = {name: 0.0 for name in FEATURE_COLUMNS}
            row.update(
                {
                    "userId": user,
                    "movieId": user * 100 + rank,
                    "label": int(rank == 1),
                    "retrieval_score": 1.0 / rank,
                    "retrieval_rank": rank,
                    "inverse_retrieval_rank": 1.0 / rank,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_logistic_ranker_fits_and_scores() -> None:
    frame = _frame()
    model = LogisticRanker(epochs=10).fit(frame)
    scores = model.predict_score(frame)
    assert len(scores) == len(frame)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_ranker_returns_top_k_per_user() -> None:
    frame = _frame()
    model = LogisticRanker(epochs=10).fit(frame)
    ranked = rank_with_model(frame, model, top_k=2)
    assert ranked.groupby("userId").size().eq(2).all()
    assert ranked["recommendation_rank"].between(1, 2).all()


def test_unfitted_ranker_is_rejected() -> None:
    with pytest.raises(ValueError, match="fitted"):
        LogisticRanker().predict_score(_frame())
