"""Run the Day-06 final serving acceptance benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from recomx.models.popularity import PopularityModel
from recomx.serving.api import create_app
from recomx.serving.artifacts import load_ranking_artifact
from recomx.serving.contract import RecommendationRequest
from recomx.serving.fallback import recommend_cold_user
from recomx.serving.service import recommend_known_user

CANDIDATES = Path("reports/ranking/validation_ranking_features.parquet")
MODEL = Path("reports/ranking/ml/model.npz")
TRAIN = Path("data/processed/ratings_train.parquet")


class BenchmarkBackend:
    def __init__(self, known_response, cold_response) -> None:
        self.known_response = known_response
        self.cold_response = cold_response

    def recommend(self, request: RecommendationRequest):
        if request.user_id == self.known_response.user_id:
            return self.known_response
        return self.cold_response


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES)
    train = pd.read_parquet(TRAIN, columns=["userId", "movieId", "rating"])
    artifact = load_ranking_artifact(str(MODEL))
    known_id = int(candidates["userId"].iloc[0])
    seen = set(train.loc[train["userId"].eq(known_id), "movieId"].astype(int))
    known = recommend_known_user(
        RecommendationRequest(known_id, 20), candidates, artifact, seen
    )
    popularity_model = PopularityModel.fit(train)
    popularity = popularity_model.item_scores[["movieId", "popularity_score"]].copy()
    cold = recommend_cold_user(RecommendationRequest(2_000_000_001, 20), popularity)

    client = TestClient(create_app(BenchmarkBackend(known, cold)))
    api_known = client.get(f"/v1/recommendations/{known_id}?top_k=20")
    api_cold = client.get("/v1/recommendations/2000000001?top_k=20")
    summary = {
        "known_strategy": known.strategy,
        "known_recommendations": len(known.recommendations),
        "known_seen_violations": sum(item.movie_id in seen for item in known.recommendations),
        "cold_strategy": cold.strategy,
        "cold_recommendations": len(cold.recommendations),
        "api_known_status": api_known.status_code,
        "api_cold_status": api_cold.status_code,
        "artifact_cached": load_ranking_artifact(str(MODEL)) is artifact,
        "locked_test_used": False,
    }
    expected = {
        "known_strategy": "personalized",
        "known_recommendations": 20,
        "known_seen_violations": 0,
        "cold_strategy": "cold_start_popularity",
        "cold_recommendations": 20,
        "api_known_status": 200,
        "api_cold_status": 200,
        "artifact_cached": True,
        "locked_test_used": False,
    }
    if summary != expected:
        raise ValueError(f"Day-06 benchmark failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 SERVING BENCHMARK: PASS")


if __name__ == "__main__":
    main()
