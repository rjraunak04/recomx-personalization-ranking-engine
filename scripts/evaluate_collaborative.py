"""Evaluate Day-03 collaborative candidates against the frozen Day-02 protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.metrics import evaluate_recommendations
from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL

GROUND_TRUTH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
POPULARITY = Path("reports/baselines/popularity/recommendations_top20.parquet")
GENRE = Path("reports/baselines/genre/recommendations_top20.parquet")
KNN = Path("reports/baselines/collaborative/candidates/item_knn_top20.parquet")
SVD = Path("reports/baselines/collaborative/candidates/latent_svd_top20.parquet")
OUTPUT = Path("reports/baselines/collaborative/evaluation")


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["userId", "movieId", "recommendation_rank"])


def _evaluate(name: str, recs: pd.DataFrame, truth: pd.DataFrame, cutoffs: tuple[int, ...]):
    user, aggregate = evaluate_recommendations(recs, truth, cutoffs=cutoffs)
    user.insert(0, "model", name)
    aggregate.insert(0, "model", name)
    return user, aggregate


def main() -> None:
    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()
    paths = {
        "ground_truth": GROUND_TRUTH,
        "popularity": POPULARITY,
        "genre_personalized": GENRE,
        "item_knn": KNN,
        "latent_svd": SVD,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifacts: " + ", ".join(missing))

    print("RecomX Day-03 CP-06 Collaborative Offline Evaluation")
    print("=" * 72)
    print("protocol=FROZEN DAY-02 VALIDATION PROTOCOL")
    print("candidate_models=REUSED (NO RETRAINING)")
    print("locked_test=NOT READ")

    truth = pd.read_parquet(GROUND_TRUTH, columns=["userId", "movieId"])
    cutoffs = tuple(sorted(protocol.top_k))
    user_frames = []
    aggregate_frames = []

    for name, path in paths.items():
        if name == "ground_truth":
            continue
        user, aggregate = _evaluate(name, _load(path), truth, cutoffs)
        user_frames.append(user)
        aggregate_frames.append(aggregate)

    user_metrics = pd.concat(user_frames, ignore_index=True)
    comparison = (
        pd.concat(aggregate_frames, ignore_index=True)
        .sort_values(["k", "model"], kind="mergesort")
        .reset_index(drop=True)
    )

    expected_users = int(truth["userId"].nunique())
    if not comparison["users"].eq(expected_users).all():
        raise ValueError("Every model/cutoff must evaluate the same ground-truth users.")

    metric_columns = ["precision", "recall", "ndcg", "map", "hit_rate"]
    if not comparison[metric_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        raise ValueError("Ranking metric outside [0, 1].")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT / "cf_model_comparison.csv", index=False)
    user_metrics.to_parquet(OUTPUT / "cf_user_metrics.parquet", index=False)

    summary = {
        "evaluation_users": expected_users,
        "ground_truth_interactions": int(len(truth)),
        "cutoffs": list(cutoffs),
        "models": sorted(comparison["model"].unique().tolist()),
        "locked_test_used": False,
    }
    with (OUTPUT / "cf_evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        comparison[
            ["model", "k", "users", "precision", "recall", "ndcg", "map", "hit_rate"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"output={OUTPUT}")
    print("DAY-03 CP-06 COLLABORATIVE OFFLINE EVALUATION: PASS")


if __name__ == "__main__":
    main()
