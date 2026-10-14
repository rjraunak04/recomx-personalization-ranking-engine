"""Train and run Day-05 CP-05 ML ranker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.ranking.ml_ranker import LogisticRanker, rank_with_model

FEATURES = Path("reports/ranking/validation_ranking_features.parquet")
OUTPUT_DIR = Path("reports/ranking/ml")
RECOMMENDATIONS = OUTPUT_DIR / "recommendations_top20.parquet"
MODEL = OUTPUT_DIR / "model.npz"
SUMMARY = OUTPUT_DIR / "summary.json"


def main() -> None:
    if not FEATURES.is_file():
        raise FileNotFoundError(f"Missing prerequisite artifact: {FEATURES}")

    frame = pd.read_parquet(FEATURES)
    # Deterministic user split prevents the model from being trained and assessed
    # on the same user rows while retaining the frozen validation candidate pool.
    user_ids = np.sort(frame["userId"].unique())
    fit_users = set(user_ids[: int(len(user_ids) * 0.8)])
    fit_frame = frame.loc[frame["userId"].isin(fit_users)].copy()
    score_frame = frame.loc[~frame["userId"].isin(fit_users)].copy()

    model = LogisticRanker().fit(fit_frame)
    recommendations = rank_with_model(score_frame, model, top_k=20)
    counts = recommendations.groupby("userId").size()
    if not counts.eq(20).all():
        raise ValueError("ML ranker must return exactly 20 recommendations per scored user.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recommendations.to_parquet(RECOMMENDATIONS, index=False)
    np.savez(
        MODEL,
        weights=model.weights_,
        mean=model.mean_,
        scale=model.scale_,
        bias=np.asarray([model.bias_]),
        feature_names=np.asarray(FEATURE_COLUMNS),
    )
    summary = {
        "fit_users": len(fit_users),
        "fit_rows": len(fit_frame),
        "scored_users": int(score_frame["userId"].nunique()),
        "scored_rows": len(score_frame),
        "recommendation_rows": len(recommendations),
        "feature_count": len(FEATURE_COLUMNS),
        "model": "standardized_logistic_ranker",
        "locked_test_used": False,
    }
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-05 CP-05 ML RANKER: PASS")


if __name__ == "__main__":
    main()
