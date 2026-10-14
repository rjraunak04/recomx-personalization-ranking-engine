"""Verify the Day-06 CP-05 HTTP API contract without production artifacts."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from recomx.serving.api import create_app
from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


class VerificationBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        strategy = "personalized" if request.user_id == 7 else "cold_start_popularity"
        return RecommendationResponse(
            user_id=request.user_id,
            strategy=strategy,
            recommendations=tuple(
                RecommendationItem(
                    movie_id=rank,
                    rank=rank,
                    score=1.0 / rank,
                    reason="verification",
                )
                for rank in range(1, request.top_k + 1)
            ),
        )


def main() -> None:
    client = TestClient(create_app(VerificationBackend()))
    health = client.get("/health")
    ready = client.get("/ready")
    known = client.get("/v1/recommendations/7?top_k=3")
    cold = client.get("/v1/recommendations/999999?top_k=3")
    invalid = client.get("/v1/recommendations/7?top_k=21")
    summary = {
        "health_status": health.status_code,
        "ready_status": ready.status_code,
        "known_strategy": known.json()["strategy"],
        "cold_strategy": cold.json()["strategy"],
        "invalid_request_status": invalid.status_code,
        "locked_test_used": False,
    }
    if summary != {
        "health_status": 200,
        "ready_status": 200,
        "known_strategy": "personalized",
        "cold_strategy": "cold_start_popularity",
        "invalid_request_status": 422,
        "locked_test_used": False,
    }:
        raise ValueError(f"Unexpected API verification result: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-05 RECOMMENDATION API: PASS")


if __name__ == "__main__":
    main()
