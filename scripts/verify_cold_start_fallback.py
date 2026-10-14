"""Verify Day-06 CP-04 cold-start serving using frozen train-only popularity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.models.popularity import PopularityModel
from recomx.serving.contract import RecommendationRequest
from recomx.serving.fallback import recommend_cold_user

TRAIN = Path("data/processed/ratings_train.parquet")
COLD_USER_ID = 2_000_000_001


def main() -> None:
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Missing prerequisite artifact: {TRAIN}")

    train = pd.read_parquet(TRAIN, columns=["userId", "movieId", "rating"])
    if COLD_USER_ID in set(train["userId"].astype(int).unique()):
        raise ValueError("Verification user unexpectedly exists in training data.")

    model = PopularityModel.fit(train)
    popularity = model.item_scores.loc[:, ["movieId", "popularity_score"]].copy()
    response = recommend_cold_user(
        RecommendationRequest(user_id=COLD_USER_ID, top_k=20),
        popularity,
    )
    second = recommend_cold_user(
        RecommendationRequest(user_id=COLD_USER_ID, top_k=20),
        popularity,
    )
    summary = {
        "user_id": COLD_USER_ID,
        "strategy": response.strategy,
        "recommendations": len(response.recommendations),
        "unique_items": len({item.movie_id for item in response.recommendations}),
        "deterministic": response == second,
        "train_only_popularity": True,
        "locked_test_used": False,
    }
    if summary["recommendations"] != 20 or summary["unique_items"] != 20:
        raise ValueError("Cold-start fallback failed the Top-20 contract.")
    if not summary["deterministic"]:
        raise ValueError("Cold-start fallback is not deterministic.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-04 COLD-START FALLBACK: PASS")


if __name__ == "__main__":
    main()
