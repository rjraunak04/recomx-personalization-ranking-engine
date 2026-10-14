"""Build Day-05 CP-02 ranking dataset from frozen Day-04 candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.ranking.dataset import build_ranking_dataset, ranking_dataset_summary

CANDIDATES = Path("reports/hybrid/hybrid_candidates_top100.parquet")
GROUND_TRUTH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
OUTPUT = Path("reports/ranking")
DATASET = OUTPUT / "validation_ranking_dataset.parquet"
SUMMARY = OUTPUT / "ranking_dataset_summary.json"


def main() -> None:
    for path in (CANDIDATES, GROUND_TRUTH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite artifact: {path}")

    candidates = pd.read_parquet(CANDIDATES)
    truth = pd.read_parquet(GROUND_TRUTH, columns=["userId", "movieId"])
    frame = build_ranking_dataset(candidates, truth)
    summary = ranking_dataset_summary(frame)
    summary.update(
        {
            "candidate_budget": 100,
            "expected_users": int(truth["userId"].nunique()),
            "locked_test_used": False,
        }
    )

    counts = frame.groupby("userId").size()
    if int(frame["userId"].nunique()) != summary["expected_users"]:
        raise ValueError("Ranking dataset must cover every evaluable validation user.")
    if not counts.eq(100).all():
        raise ValueError("Ranking dataset requires exactly 100 candidates per user.")
    if frame["candidate_rank"].lt(1).any() or frame["candidate_rank"].gt(100).any():
        raise ValueError("Candidate ranks must remain in 1..100.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATASET, index=False)
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-05 CP-02 RANKING DATASET: PASS")


if __name__ == "__main__":
    main()
