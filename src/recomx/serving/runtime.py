"""Production runtime configuration and readiness contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ENVIRONMENTS = {"development", "staging", "production"}


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str
    host: str
    port: int
    ranking_model_path: Path
    ranking_features_path: Path
    train_ratings_path: Path
    movies_path: Path
    locked_test_access: bool = False

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        return cls(
            environment=os.getenv("RECOMX_ENV", "development"),
            host=os.getenv("RECOMX_HOST", "0.0.0.0"),
            port=int(os.getenv("RECOMX_PORT", "8000")),
            ranking_model_path=Path(
                os.getenv("RECOMX_RANKING_MODEL", "reports/ranking/ml/model.npz")
            ),
            ranking_features_path=Path(
                os.getenv(
                    "RECOMX_RANKING_FEATURES",
                    "reports/ranking/validation_ranking_features.parquet",
                )
            ),
            train_ratings_path=Path(
                os.getenv(
                    "RECOMX_TRAIN_RATINGS",
                    "data/processed/ratings_train.parquet",
                )
            ),
            movies_path=Path(
                os.getenv("RECOMX_MOVIES", "data/interim/movies.parquet")
            ),
            locked_test_access=False,
        )

    def validate(self, *, require_artifacts: bool = False) -> None:
        if self.environment not in _ALLOWED_ENVIRONMENTS:
            raise ValueError(f"Unsupported RECOMX_ENV: {self.environment}")
        if not self.host.strip():
            raise ValueError("RECOMX_HOST must not be empty.")
        if not 1 <= self.port <= 65535:
            raise ValueError("RECOMX_PORT must be between 1 and 65535.")
        if self.locked_test_access:
            raise ValueError("Locked test access is forbidden in the serving runtime.")
        paths = self.required_artifacts()
        if any(
            "locked" in str(path).lower() and "test" in str(path).lower()
            for path in paths
        ):
            raise ValueError("Serving runtime must not reference the locked test partition.")
        if require_artifacts:
            missing = [path.name for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing required serving artifacts: {sorted(missing)}"
                )

    def required_artifacts(self) -> tuple[Path, ...]:
        return (
            self.ranking_model_path,
            self.ranking_features_path,
            self.train_ratings_path,
            self.movies_path,
        )

    def readiness(self) -> dict[str, object]:
        self.validate(require_artifacts=False)
        missing = [
            path.name for path in self.required_artifacts() if not path.is_file()
        ]
        return {
            "status": "ready" if not missing else "not_ready",
            "environment": self.environment,
            "artifacts_ready": not missing,
            "missing_artifacts": tuple(sorted(missing)),
            "locked_test_access": False,
        }
