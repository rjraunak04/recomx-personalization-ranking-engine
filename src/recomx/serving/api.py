"""FastAPI boundary for the RecomX recommendation service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

from fastapi import FastAPI, HTTPException, Query

from recomx.serving.contract import RecommendationRequest, RecommendationResponse


class RecommendationBackend(Protocol):
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse: ...


def create_app(backend: RecommendationBackend) -> FastAPI:
    """Create an API app around an injected recommendation backend."""
    app = FastAPI(title="RecomX Recommendation API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/v1/recommendations/{user_id}")
    def recommendations(
        user_id: int,
        top_k: int = Query(default=20, ge=1, le=20),
    ) -> dict[str, object]:
        request = RecommendationRequest(user_id=user_id, top_k=top_k)
        try:
            request.validate()
            response = backend.recommend(request)
            response.validate()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Serving artifact unavailable.") from exc
        return asdict(response)

    return app
