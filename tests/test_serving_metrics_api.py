from fastapi.testclient import TestClient

from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from recomx.serving.metrics import ServingMetrics
from recomx.serving.api import create_app


class _Backend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        return RecommendationResponse(
            user_id=request.user_id,
            strategy="personalized",
            recommendations=(
                RecommendationItem(
                    movie_id=1,
                    rank=1,
                    score=0.9,
                    reason="test",
                ),
            ),
        )


def test_api_updates_and_exposes_metrics() -> None:
    metrics = ServingMetrics()
    client = TestClient(create_app(_Backend(), metrics=metrics))
    response = client.get("/v1/recommendations/1?top_k=1")
    assert response.status_code == 200
    snapshot = client.get("/metrics")
    assert snapshot.status_code == 200
    payload = snapshot.json()
    assert payload["request_count"] == 1
    assert payload["success_count"] == 1
    assert payload["personalized_count"] == 1
    assert payload["recommendation_count"] == 1
