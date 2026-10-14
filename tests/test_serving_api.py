from __future__ import annotations

from fastapi.testclient import TestClient

from recomx.serving.api import create_app
from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


class FakeBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        strategy = "personalized" if request.user_id == 7 else "cold_start_popularity"
        return RecommendationResponse(
            user_id=request.user_id,
            strategy=strategy,
            recommendations=tuple(
                RecommendationItem(
                    movie_id=100 + rank,
                    rank=rank,
                    score=1.0 / rank,
                    reason="test",
                )
                for rank in range(1, request.top_k + 1)
            ),
        )


def _client() -> TestClient:
    return TestClient(create_app(FakeBackend()))


def test_health_and_readiness() -> None:
    client = _client()
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_known_user_route() -> None:
    response = _client().get("/v1/recommendations/7?top_k=3")
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "personalized"
    assert len(body["recommendations"]) == 3


def test_cold_user_route() -> None:
    response = _client().get("/v1/recommendations/999999?top_k=2")
    assert response.status_code == 200
    assert response.json()["strategy"] == "cold_start_popularity"


def test_invalid_top_k_is_rejected() -> None:
    assert _client().get("/v1/recommendations/7?top_k=21").status_code == 422


def test_invalid_user_is_rejected() -> None:
    assert _client().get("/v1/recommendations/0").status_code == 422
