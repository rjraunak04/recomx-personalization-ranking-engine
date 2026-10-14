"""Verify and summarize the RecomX Day-03 collaborative benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL

GT = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
KNN = Path("reports/baselines/collaborative/candidates/item_knn_top20.parquet")
SVD = Path("reports/baselines/collaborative/candidates/latent_svd_top20.parquet")
COMPARISON = Path("reports/baselines/collaborative/evaluation/cf_model_comparison.csv")
COHORT = Path("reports/baselines/collaborative/diagnostics/cohort_metrics.csv")
STRATEGY = Path("reports/baselines/collaborative/diagnostics/strategy_metrics.csv")
DIAGNOSTICS = Path("reports/baselines/collaborative/diagnostics/diagnostic_summary.json")
OUTPUT = Path("reports/baselines/collaborative/benchmark")
SUMMARY = OUTPUT / "day03_benchmark_summary.json"


def _validate_recommendations(frame: pd.DataFrame, users: set[int], max_k: int, name: str):
    required = {"userId", "movieId", "recommendation_rank", "strategy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if set(frame["userId"].astype(int)) != users:
        raise ValueError(f"{name} user set mismatch.")
    if frame.duplicated(["userId", "movieId"]).any():
        raise ValueError(f"{name} duplicate user-item recommendations.")
    if frame.duplicated(["userId", "recommendation_rank"]).any():
        raise ValueError(f"{name} duplicate ranks.")
    counts = frame.groupby("userId").size()
    if not counts.eq(max_k).all():
        raise ValueError(f"{name} must provide exactly Top-{max_k}.")
    ranks = frame.groupby("userId")["recommendation_rank"].agg(["min", "max"])
    if not (ranks["min"].eq(1).all() and ranks["max"].eq(max_k).all()):
        raise ValueError(f"{name} ranks must be contiguous 1..{max_k}.")
    return {
        "users": len(users),
        "rows": len(frame),
        "unique_items": int(frame["movieId"].nunique()),
    }


def main() -> None:
    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()
    required = [GT, KNN, SVD, COMPARISON, COHORT, STRATEGY, DIAGNOSTICS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Day-03 artifacts: " + ", ".join(missing))

    truth = pd.read_parquet(GT, columns=["userId", "movieId"])
    users = set(truth["userId"].astype(int))
    cutoffs = tuple(sorted(protocol.top_k))
    max_k = max(cutoffs)
    knn = pd.read_parquet(KNN)
    svd = pd.read_parquet(SVD)
    comparison = pd.read_csv(COMPARISON)
    cohort = pd.read_csv(COHORT)
    strategy = pd.read_csv(STRATEGY)
    with DIAGNOSTICS.open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)

    integrity = {
        "item_knn": _validate_recommendations(knn, users, max_k, "item_knn"),
        "latent_svd": _validate_recommendations(svd, users, max_k, "latent_svd"),
    }

    expected_models = {"popularity", "genre_personalized", "item_knn", "latent_svd"}
    if set(comparison["model"]) != expected_models:
        raise ValueError("Final comparison model set mismatch.")
    expected_pairs = {(model, k) for model in expected_models for k in cutoffs}
    actual_pairs = set(zip(comparison["model"], comparison["k"].astype(int), strict=True))
    if actual_pairs != expected_pairs:
        raise ValueError("Missing model/cutoff benchmark combination.")
    if not comparison["users"].eq(len(users)).all():
        raise ValueError("Benchmark evaluation user counts are inconsistent.")

    metrics = ["precision", "recall", "ndcg", "map", "hit_rate"]
    for column in metrics:
        if comparison[column].isna().any() or not comparison[column].between(0, 1).all():
            raise ValueError(f"Invalid benchmark metric: {column}")

    if set(cohort["segment"]) != {"warm", "cold"}:
        raise ValueError("Warm/cold diagnostic segments are incomplete.")
    strategies = set(strategy["strategy"])
    if not {"item_knn", "latent_svd", "popularity_fallback"}.issubset(strategies):
        raise ValueError("Collaborative strategy diagnostics are incomplete.")

    at_20 = comparison.loc[comparison["k"].eq(20)].copy()
    winner = at_20.sort_values(
        ["ndcg", "recall", "precision"], ascending=False, kind="mergesort"
    ).iloc[0]

    summary = {
        "evaluation_users": len(users),
        "ground_truth_interactions": len(truth),
        "cutoffs": list(cutoffs),
        "locked_test_used": False,
        "recommendation_integrity": integrity,
        "best_model_at_20_by_ndcg": str(winner["model"]),
        "metrics_at_20": at_20.set_index("model")[metrics].to_dict(orient="index"),
        "diagnostics": diagnostics["models"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("RecomX Day-03 Collaborative Benchmark")
    print("=" * 72)
    print("evaluation_partition=validation")
    print("locked_test=NOT READ")
    print(f"evaluation_users={len(users)}")
    print(f"ground_truth_interactions={len(truth)}")
    print(f"cutoffs={cutoffs}")
    print("\n=== RECOMMENDATION INTEGRITY ===")
    print(json.dumps(integrity, indent=2, sort_keys=True))
    print("\n=== FINAL MODEL COMPARISON @20 ===")
    print(
        at_20[
            ["model", "precision", "recall", "ndcg", "map", "hit_rate"]
        ].sort_values("ndcg", ascending=False).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nbest_model_at_20_by_ndcg={winner['model']}")
    print(f"summary={SUMMARY}")
    print("DAY-03 COLLABORATIVE BENCHMARK: PASS")


if __name__ == "__main__":
    main()
