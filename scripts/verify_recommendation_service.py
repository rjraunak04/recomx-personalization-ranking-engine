"""Verify Day-06 CP-03 recommendation orchestration on frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.serving.artifacts import load_ranking_artifact
from recomx.serving.contract import RecommendationRequest
from recomx.serving.service import recommend_known_user

CANDIDATES = Path("reports/ranking/features.parquet")
MODEL = Path("reports/ranking/ml/model.npz")
TRAIN = Path("data/processed/ratings_train.parquet")


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES)
    train = pd.read_parquet(TRAIN, columns=["userId", "movieId"])
    artifact = load_ranking_artifact(str(MODEL))

    user_id = int(candidates["userId"].iloc[0])
    seen = set(
        train.loc[train["userId"].eq(user_id), "movieId"].astype(int).tolist()
    )
    response = recommend_known_user(
        RecommendationRequest(user_id=user_id, top_k=20),
        candidates,
        artifact,
        seen,
    )
    summary = {
        "user_id": user_id,
        "recommendations": len(response.recommendations),
        "strategy": response.strategy,
        "seen_item_violations": sum(
            item.movie_id in seen for item in response.recommendations
        ),
        "deterministic_ranks": [
            item.rank for item in response.recommendations
        ] == list(range(1, len(response.recommendations) + 1)),
        "locked_test_used": False,
    }
    if summary["seen_item_violations"] != 0:
        raise ValueError("Serving response contains seen items.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-03 RECOMMENDATION SERVICE: PASS")


if __name__ == "__main__":
    main()
