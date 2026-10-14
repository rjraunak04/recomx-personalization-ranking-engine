from __future__ import annotations

from fastapi.testclient import TestClient

from recomx.serving.api import create_app
from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


class ValueErrorBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        raise ValueError("No personalized candidates remain.")


class MissingArtifactBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        raise FileNotFoundError("secret/local/model.npz")


class InvalidResponseBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        return RecommendationResponse(
            user_id=request.user_id,
            strategy="personalized",
            recommendations=(
                RecommendationItem(movie_id=10, rank=2, score=0.5, reason="bad"),
            ),
        )


def test_backend_domain_error_returns_422() -> None:
    response = TestClient(create_app(ValueErrorBackend())).get("/v1/recommendations/7")
    assert response.status_code == 422
    assert response.json()["detail"] == "No personalized candidates remain."


def test_missing_artifact_returns_503_without_path_leak() -> None:
    response = TestClient(create_app(MissingArtifactBackend())).get("/v1/recommendations/7")
    assert response.status_code == 503
    assert response.json()["detail"] == "Serving artifact unavailable."
    assert "model.npz" not in response.text


def test_invalid_backend_response_is_rejected() -> None:
    response = TestClient(create_app(InvalidResponseBackend())).get("/v1/recommendations/7")
    assert response.status_code == 422
    assert "contiguous" in response.json()["detail"]


def test_malformed_path_parameter_is_422() -> None:
    response = TestClient(create_app(ValueErrorBackend())).get("/v1/recommendations/not-an-int")
    assert response.status_code == 422


def test_invalid_query_type_is_422() -> None:
    response = TestClient(create_app(ValueErrorBackend())).get("/v1/recommendations/7?top_k=abc")
    assert response.status_code == 422
