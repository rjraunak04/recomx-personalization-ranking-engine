"""Concrete production backend and application factory for RecomX."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from recomx.serving.api import create_app
from recomx.serving.artifacts import RankingArtifact, load_ranking_artifact
from recomx.serving.contract import RecommendationRequest, RecommendationResponse
from recomx.serving.fallback import recommend_cold_user
from recomx.serving.runtime import RuntimeSettings
from recomx.serving.service import recommend_known_user


@dataclass
class ProductionRecommendationBackend:
    settings: RuntimeSettings

    @cached_property
    def artifact(self) -> RankingArtifact:
        return load_ranking_artifact(str(self.settings.ranking_model_path))

    @cached_property
    def candidates(self) -> pd.DataFrame:
        return pd.read_parquet(self.settings.ranking_features_path)

    @cached_property
    def train(self) -> pd.DataFrame:
        return pd.read_parquet(
            self.settings.train_ratings_path,
            columns=["userId", "movieId", "rating"],
        )

    @cached_property
    def known_users(self) -> frozenset[int]:
        return frozenset(self.train["userId"].astype(int).unique())

    @cached_property
    def popularity(self) -> pd.DataFrame:
        relevant = self.train.loc[self.train["rating"].ge(4.0), ["movieId"]]
        counts = relevant.groupby("movieId", sort=False).size().rename("popularity_score")
        return counts.reset_index()

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request.validate()
        if request.user_id not in self.known_users:
            return recommend_cold_user(request, self.popularity)

        user_candidates = self.candidates.loc[self.candidates["userId"].eq(request.user_id)]
        if user_candidates.empty:
            return recommend_cold_user(request, self.popularity)

        seen = frozenset(
            self.train.loc[
                self.train["userId"].eq(request.user_id),
                "movieId",
            ].astype(int)
        )
        return recommend_known_user(request, user_candidates, self.artifact, seen)


def create_production_app(settings: RuntimeSettings | None = None):
    runtime = settings or RuntimeSettings.from_env()
    runtime.validate(require_artifacts=True)
    backend = ProductionRecommendationBackend(runtime)
    app = create_app(backend, readiness=runtime.readiness)
    app.state.runtime_settings = runtime
    return app


app = create_production_app()
