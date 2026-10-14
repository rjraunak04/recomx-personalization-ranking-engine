from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.serving.artifacts import RankingArtifact
from recomx.serving.contract import RecommendationRequest
from recomx.serving.service import recommend_known_user


def _artifact() -> RankingArtifact:
    width = len(FEATURE_COLUMNS)
    return RankingArtifact(
        weights=np.ones(width),
        mean=np.zeros(width),
        scale=np.ones(width),
        bias=0.0,
        feature_names=tuple(FEATURE_COLUMNS),
    )


def _candidates() -> pd.DataFrame:
    rows = []
    for movie_id, value in [(30, 0.1), (20, 0.8), (10, 0.8), (40, 0.2)]:
        row = {"userId": 7, "movieId": movie_id}
        row.update({name: value for name in FEATURE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_known_user_is_ranked_deterministically() -> None:
    response = recommend_known_user(
        RecommendationRequest(user_id=7, top_k=3),
        _candidates(),
        _artifact(),
        seen_movie_ids=set(),
    )
    assert response.strategy == "personalized"
    assert [item.movie_id for item in response.recommendations] == [10, 20, 40]
    assert [item.rank for item in response.recommendations] == [1, 2, 3]


def test_seen_items_are_filtered() -> None:
    response = recommend_known_user(
        RecommendationRequest(user_id=7, top_k=3),
        _candidates(),
        _artifact(),
        seen_movie_ids={10, 40},
    )
    assert [item.movie_id for item in response.recommendations] == [20, 30]


def test_wrong_user_candidates_fail_fast() -> None:
    with pytest.raises(ValueError, match="No personalized candidates"):
        recommend_known_user(
            RecommendationRequest(user_id=99),
            _candidates(),
            _artifact(),
            seen_movie_ids=set(),
        )


def test_missing_feature_fails_fast() -> None:
    frame = _candidates().drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing columns"):
        recommend_known_user(
            RecommendationRequest(user_id=7),
            frame,
            _artifact(),
            seen_movie_ids=set(),
        )
