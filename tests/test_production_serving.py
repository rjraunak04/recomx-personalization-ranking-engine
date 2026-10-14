from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.serving.production import ProductionRecommendationBackend
from recomx.serving.runtime import RuntimeSettings


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(
        environment="production",
        host="0.0.0.0",
        port=8000,
        ranking_model_path=tmp_path / "model.npz",
        ranking_features_path=tmp_path / "features.parquet",
        train_ratings_path=tmp_path / "train.parquet",
        movies_path=tmp_path / "movies.parquet",
    )


def _write_artifacts(tmp_path: Path) -> RuntimeSettings:
    settings = _settings(tmp_path)
    np.savez(
        settings.ranking_model_path,
        weights=np.zeros(len(FEATURE_COLUMNS)),
        mean=np.zeros(len(FEATURE_COLUMNS)),
        scale=np.ones(len(FEATURE_COLUMNS)),
        bias=np.array([0.0]),
        feature_names=np.array(FEATURE_COLUMNS),
    )
    rows = []
    for movie_id in (3, 4, 5):
        row = {"userId": 1, "movieId": movie_id}
        row.update({name: 0.0 for name in FEATURE_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(settings.ranking_features_path)
    pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [1, 2, 10, 11, 12],
            "rating": [5.0, 4.0, 5.0, 4.0, 5.0],
        }
    ).to_parquet(settings.train_ratings_path)
    pd.DataFrame(
        {"movieId": [1], "title": ["x"], "genres": ["Drama"]}
    ).to_parquet(settings.movies_path)
    return settings


def test_known_user_gets_personalized_ranking(tmp_path: Path) -> None:
    backend = ProductionRecommendationBackend(_write_artifacts(tmp_path))
    response = backend.recommend(
        __import__("recomx.serving.contract", fromlist=["RecommendationRequest"])
        .RecommendationRequest(user_id=1, top_k=2)
    )
    assert response.strategy == "personalized"
    assert len(response.recommendations) == 2


def test_unknown_user_gets_popularity_fallback(tmp_path: Path) -> None:
    backend = ProductionRecommendationBackend(_write_artifacts(tmp_path))
    request_type = __import__(
        "recomx.serving.contract", fromlist=["RecommendationRequest"]
    ).RecommendationRequest
    response = backend.recommend(request_type(user_id=999, top_k=2))
    assert response.strategy == "cold_start_popularity"
    assert len(response.recommendations) == 2


def test_known_user_without_candidates_falls_back(tmp_path: Path) -> None:
    backend = ProductionRecommendationBackend(_write_artifacts(tmp_path))
    request_type = __import__(
        "recomx.serving.contract", fromlist=["RecommendationRequest"]
    ).RecommendationRequest
    response = backend.recommend(request_type(user_id=2, top_k=2))
    assert response.strategy == "cold_start_popularity"


def test_readiness_callable_reports_real_artifacts(tmp_path: Path) -> None:
    settings = _write_artifacts(tmp_path)
    from recomx.serving.production import create_production_app

    client = TestClient(create_production_app(settings))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["artifacts_ready"] is True
