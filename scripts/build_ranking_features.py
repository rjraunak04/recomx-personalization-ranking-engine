"""Build Day-05 CP-03 leakage-safe ranking features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.ranking.features import FEATURE_COLUMNS, add_ranking_features

RANKING = Path("reports/ranking/validation_ranking_dataset.parquet")
TRAIN = Path("data/processed/ratings_train.parquet")
OUTPUT = Path("reports/ranking/validation_ranking_features.parquet")
SUMMARY = Path("reports/ranking/ranking_feature_summary.json")


def main() -> None:
    for path in (RANKING, TRAIN):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite artifact: {path}")

    ranking = pd.read_parquet(RANKING)
    train = pd.read_parquet(TRAIN, columns=["userId", "movieId", "rating"])
    features = add_ranking_features(ranking, train)

    if features[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Ranking feature matrix contains missing values.")
    if len(features) != len(ranking):
        raise ValueError("Feature engineering changed ranking row count.")
    if int(features["label"].sum()) != int(ranking["label"].sum()):
        raise ValueError("Feature engineering changed relevance labels.")

    features.to_parquet(OUTPUT, index=False)
    summary = {
        "rows": len(features),
        "users": int(features["userId"].nunique()),
        "features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "positive_rows": int(features["label"].sum()),
        "missing_feature_values": 0,
        "locked_test_used": False,
    }
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-05 CP-03 RANKING FEATURES: PASS")


if __name__ == "__main__":
    main()
