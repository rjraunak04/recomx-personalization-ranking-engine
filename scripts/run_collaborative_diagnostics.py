"""Run Day-03 CP-07 collaborative recommendation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.diagnostics import (
    average_recommendation_popularity,
    build_long_tail_items,
    catalogue_coverage,
    cold_item_exposure,
    evaluate_by_cohort,
    evaluate_by_strategy,
    long_tail_share,
    recommendation_coverage,
)
from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL

GT = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
COHORTS = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")
ITEM_SCORES = Path("reports/baselines/popularity/item_popularity.parquet")
KNN = Path("reports/baselines/collaborative/candidates/item_knn_top20.parquet")
SVD = Path("reports/baselines/collaborative/candidates/latent_svd_top20.parquet")
TRAIN = Path("data/processed/ratings_train.parquet")
MOVIES = Path("data/interim/movies.parquet")
OUTPUT = Path("reports/baselines/collaborative/diagnostics")


def _diagnose(name, recs, truth, cohorts, item_scores, train_items, catalogue, cutoffs):
    eligible = set(item_scores["movieId"].astype(int))
    tail = build_long_tail_items(item_scores)
    cold_rows, cold_share = cold_item_exposure(recs, train_items)
    unique_items = int(recs["movieId"].nunique())
    summary = {
        "model": name,
        "recommendation_rows": int(len(recs)),
        "unique_recommended_items": unique_items,
        "catalogue_coverage": catalogue_coverage(recs, catalogue),
        "eligible_item_coverage": recommendation_coverage(recs, eligible),
        "average_recommendation_popularity": average_recommendation_popularity(recs, item_scores),
        "long_tail_share": long_tail_share(recs, tail),
        "cold_item_exposure_rows": cold_rows,
        "cold_item_exposure_share": cold_share,
        "strategy_users": {
            str(k): int(v)
            for k, v in recs[["userId", "strategy"]]
            .drop_duplicates()
            .groupby("strategy")["userId"].nunique()
            .items()
        },
    }
    cohort = evaluate_by_cohort(recs, truth, cohorts, cutoffs=cutoffs)
    cohort.insert(0, "model", name)
    strategy = evaluate_by_strategy(recs, truth, cutoffs=cutoffs)
    strategy.insert(0, "model", name)
    return summary, cohort, strategy


def main() -> None:
    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()
    required = [GT, COHORTS, ITEM_SCORES, KNN, SVD, TRAIN, MOVIES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifacts: " + ", ".join(missing))

    print("RecomX Day-03 CP-07 Collaborative Diagnostics")
    print("=" * 72)
    print("candidate_models=REUSED (NO RETRAINING)")
    print("locked_test=NOT READ")

    truth = pd.read_parquet(GT, columns=["userId", "movieId"])
    cohorts = pd.read_parquet(COHORTS)
    item_scores = pd.read_parquet(ITEM_SCORES)
    train = pd.read_parquet(TRAIN, columns=["movieId"])
    movies = pd.read_parquet(MOVIES, columns=["movieId"])
    train_items = set(train["movieId"].astype(int))
    catalogue = set(movies["movieId"].astype(int))
    cutoffs = tuple(sorted(protocol.top_k))

    summaries = {}
    cohorts_out = []
    strategies_out = []
    for name, path in [("item_knn", KNN), ("latent_svd", SVD)]:
        recs = pd.read_parquet(path)
        summary, cohort, strategy = _diagnose(
            name, recs, truth, cohorts, item_scores, train_items, catalogue, cutoffs
        )
        summaries[name] = summary
        cohorts_out.append(cohort)
        strategies_out.append(strategy)

    cohort_frame = pd.concat(cohorts_out, ignore_index=True)
    strategy_frame = pd.concat(strategies_out, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort_frame.to_csv(OUTPUT / "cohort_metrics.csv", index=False)
    strategy_frame.to_csv(OUTPUT / "strategy_metrics.csv", index=False)
    with (OUTPUT / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"models": summaries, "locked_test_used": False},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    for name, summary in summaries.items():
        print(f"\n{name}")
        for key, value in summary.items():
            print(f"  {key}={value}")
    print("\nCOHORT METRICS")
    display_columns = [\n        "model", "segment", "k", "users", "precision",\n        "recall", "ndcg", "map", "hit_rate",\n    ]\n    print(cohort_frame[display_columns].to_string(index=False))
    print("\nSTRATEGY METRICS")
    strategy_columns = [\n        "model", "strategy", "k", "users", "precision",\n        "recall", "ndcg", "map", "hit_rate",\n    ]\n    print(strategy_frame[strategy_columns].to_string(index=False))
    print(f"\noutput={OUTPUT}")
    print("DAY-03 CP-07 COLLABORATIVE DIAGNOSTICS: PASS")


if __name__ == "__main__":
    main()
