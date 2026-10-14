"""Run Day-05 CP-04 deterministic ranking baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.ranking.baseline import rank_retrieval_baseline

FEATURES = Path("reports/ranking/validation_ranking_features.parquet")
OUTPUT_DIR = Path("reports/ranking/baseline")
OUTPUT = OUTPUT_DIR / "recommendations_top20.parquet"
SUMMARY = OUTPUT_DIR / "summary.json"


def main() -> None:
    if not FEATURES.is_file():
        raise FileNotFoundError(f"Missing prerequisite artifact: {FEATURES}")

    features = pd.read_parquet(FEATURES)
    ranked = rank_retrieval_baseline(features, top_k=20)
    counts = ranked.groupby("userId").size()

    if not counts.eq(20).all():
        raise ValueError("Baseline must return exactly 20 recommendations per user.")
    if ranked.duplicated(["userId", "movieId"]).any():
        raise ValueError("Baseline produced duplicate user-item pairs.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(OUTPUT, index=False)
    summary = {
        "rows": len(ranked),
        "users": int(ranked["userId"].nunique()),
        "top_k": 20,
        "minimum_recommendations_per_user": int(counts.min()),
        "maximum_recommendations_per_user": int(counts.max()),
        "ranking_signal": "day04_hybrid_retrieval_score",
        "locked_test_used": False,
    }
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-05 CP-04 RANKING BASELINE: PASS")


if __name__ == "__main__":
    main()
