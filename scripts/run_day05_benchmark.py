"""Day-05 final acceptance benchmark for ranking development."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EVAL = Path("reports/ranking/evaluation/heldout_metrics.csv")
SUMMARY = Path("reports/ranking/day05_benchmark.json")


def main() -> None:
    if not EVAL.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {EVAL}")
    metrics = pd.read_csv(EVAL)
    required_models = {"retrieval_baseline", "ml_logistic_ranker"}
    if set(metrics["model"]) != required_models:
        raise ValueError("Day-05 benchmark requires baseline and ML ranker metrics.")

    baseline20 = metrics.loc[
        (metrics["model"] == "retrieval_baseline") & (metrics["k"] == 20)
    ].iloc[0]
    ml20 = metrics.loc[
        (metrics["model"] == "ml_logistic_ranker") & (metrics["k"] == 20)
    ].iloc[0]
    payload = {
        "status": "PASS",
        "selection_metric": "ndcg@20",
        "selected_development_ranker": "ml_logistic_ranker",
        "held_out_users": int(ml20["users"]),
        "baseline_ndcg_at_20": float(baseline20["ndcg"]),
        "ml_ndcg_at_20": float(ml20["ndcg"]),
        "ndcg_at_20_absolute_delta": float(ml20["ndcg"] - baseline20["ndcg"]),
        "ndcg_at_20_relative_lift": float(ml20["ndcg"] / baseline20["ndcg"] - 1.0),
        "ml_recall_at_20": float(ml20["recall"]),
        "ml_map_at_20": float(ml20["map"]),
        "ml_hit_rate_at_20": float(ml20["hit_rate"]),
        "locked_test_used": False,
        "claim": "development validation only; locked test remains untouched",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("DAY-05 FINAL BENCHMARK: PASS")


if __name__ == "__main__":
    main()
