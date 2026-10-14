"""Verify and summarize the RecomX Day-02 baseline benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL

GROUND_TRUTH_PATH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")

COHORT_PATH = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")

POPULARITY_RECOMMENDATIONS_PATH = Path("reports/baselines/popularity/recommendations_top20.parquet")

GENRE_RECOMMENDATIONS_PATH = Path("reports/baselines/genre/recommendations_top20.parquet")

MODEL_COMPARISON_PATH = Path("reports/baselines/evaluation/model_comparison.csv")

COHORT_METRICS_PATH = Path("reports/baselines/diagnostics/cohort_metrics.csv")

STRATEGY_METRICS_PATH = Path("reports/baselines/diagnostics/strategy_metrics.csv")

DIAGNOSTICS_PATH = Path("reports/baselines/diagnostics/diagnostics_summary.json")

OUTPUT_DIR = Path("reports/baselines/benchmark")

SUMMARY_PATH = OUTPUT_DIR / "day02_benchmark_summary.json"


def _require_file(path: Path) -> None:
    """Fail when a required Day-02 artifact is missing."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing required Day-02 artifact: {path}")


def _validate_recommendations(
    frame: pd.DataFrame,
    expected_users: set[int],
    max_k: int,
    model_name: str,
) -> dict[str, int]:
    """Validate recommendation output integrity."""

    required = {
        "userId",
        "movieId",
        "recommendation_rank",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"{model_name} recommendations missing columns: {sorted(missing)}")

    users = set(frame["userId"].astype(int))

    if users != expected_users:
        raise ValueError(f"{model_name} recommendation user set does not match evaluation users.")

    duplicates = frame.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    )

    if duplicates.any():
        raise ValueError(f"{model_name} contains duplicate user-item recommendations.")

    duplicate_ranks = frame.duplicated(
        subset=[
            "userId",
            "recommendation_rank",
        ]
    )

    if duplicate_ranks.any():
        raise ValueError(f"{model_name} contains duplicate ranks.")

    counts = frame.groupby("userId").size()

    if not counts.eq(max_k).all():
        raise ValueError(
            f"{model_name} does not provide exactly Top-{max_k} for every evaluation user."
        )

    rank_min = frame.groupby("userId")["recommendation_rank"].min()

    rank_max = frame.groupby("userId")["recommendation_rank"].max()

    if not (rank_min.eq(1).all() and rank_max.eq(max_k).all()):
        raise ValueError(f"{model_name} recommendation ranks are not contiguous from 1 to {max_k}.")

    return {
        "users": int(len(users)),
        "rows": int(len(frame)),
        "unique_items": int(frame["movieId"].nunique()),
    }


def _validate_comparison(
    comparison: pd.DataFrame,
    cutoffs: tuple[int, ...],
    evaluation_users: int,
) -> None:
    """Validate ranking comparison structure."""

    expected_models = {
        "popularity",
        "genre_personalized",
    }

    actual_models = set(comparison["model"].astype(str))

    if actual_models != expected_models:
        raise ValueError("Unexpected model set in model comparison.")

    expected_pairs = {(model, k) for model in expected_models for k in cutoffs}

    actual_pairs = set(
        zip(
            comparison["model"].astype(str),
            comparison["k"].astype(int),
            strict=True,
        )
    )

    if actual_pairs != expected_pairs:
        raise ValueError("Model comparison does not contain every model-cutoff combination.")

    if not comparison["users"].eq(evaluation_users).all():
        raise ValueError("Model comparison user counts are inconsistent.")

    metric_columns = [
        "precision",
        "recall",
        "ndcg",
        "map",
        "hit_rate",
    ]

    for column in metric_columns:
        if comparison[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

        if not comparison[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains values outside [0, 1].")


def main() -> None:
    """Run the final Day-02 benchmark integrity audit."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    required_paths = (
        GROUND_TRUTH_PATH,
        COHORT_PATH,
        POPULARITY_RECOMMENDATIONS_PATH,
        GENRE_RECOMMENDATIONS_PATH,
        MODEL_COMPARISON_PATH,
        COHORT_METRICS_PATH,
        STRATEGY_METRICS_PATH,
        DIAGNOSTICS_PATH,
    )

    for path in required_paths:
        _require_file(path)

    cutoffs = tuple(sorted(protocol.top_k))

    max_k = max(cutoffs)

    print("RecomX Day-02 Baseline Benchmark")
    print("=" * 72)
    print("locked_test=NOT READ")
    print("evaluation_partition=validation")
    print(f"relevance_threshold={protocol.relevance_threshold:.1f}")
    print(f"cutoffs={cutoffs}")
    print()

    ground_truth = pd.read_parquet(
        GROUND_TRUTH_PATH,
        columns=[
            "userId",
            "movieId",
        ],
    )

    cohorts = pd.read_parquet(
        COHORT_PATH,
    )

    popularity = pd.read_parquet(POPULARITY_RECOMMENDATIONS_PATH)

    genre = pd.read_parquet(GENRE_RECOMMENDATIONS_PATH)

    comparison = pd.read_csv(MODEL_COMPARISON_PATH)

    cohort_metrics = pd.read_csv(COHORT_METRICS_PATH)

    strategy_metrics = pd.read_csv(STRATEGY_METRICS_PATH)

    with DIAGNOSTICS_PATH.open(
        encoding="utf-8",
    ) as handle:
        diagnostics = json.load(handle)

    evaluation_users = set(ground_truth["userId"].astype(int))

    if not evaluation_users:
        raise ValueError("Ground truth contains no evaluation users.")

    evaluable_cohort_users = set(
        cohorts.loc[
            cohorts["is_evaluable"],
            "userId",
        ].astype(int)
    )

    if evaluation_users != evaluable_cohort_users:
        raise ValueError("Ground-truth users do not match evaluable cohorts.")

    popularity_integrity = _validate_recommendations(
        frame=popularity,
        expected_users=evaluation_users,
        max_k=max_k,
        model_name="popularity",
    )

    genre_integrity = _validate_recommendations(
        frame=genre,
        expected_users=evaluation_users,
        max_k=max_k,
        model_name="genre_personalized",
    )

    _validate_comparison(
        comparison=comparison,
        cutoffs=cutoffs,
        evaluation_users=len(evaluation_users),
    )

    expected_segments = {
        "warm",
        "cold",
    }

    if set(cohort_metrics["segment"].astype(str)) != expected_segments:
        raise ValueError("Cohort metrics missing warm/cold segments.")

    expected_strategies = {
        "genre_personalized",
        "popularity_fallback",
    }

    if set(strategy_metrics["strategy"].astype(str)) != expected_strategies:
        raise ValueError("Strategy metrics contain unexpected routing.")

    comparison_at_10 = (
        comparison.loc[
            comparison["k"].eq(10),
            [
                "model",
                "precision",
                "recall",
                "ndcg",
                "map",
                "hit_rate",
            ],
        ]
        .set_index("model")
        .to_dict(orient="index")
    )

    warm_at_20 = (
        cohort_metrics.loc[
            (cohort_metrics["segment"].eq("warm")) & cohort_metrics["k"].eq(20),
            [
                "model",
                "precision",
                "recall",
                "ndcg",
                "map",
                "hit_rate",
            ],
        ]
        .set_index("model")
        .to_dict(orient="index")
    )

    summary = {
        "evaluation": {
            "users": int(len(evaluation_users)),
            "ground_truth_interactions": int(len(ground_truth)),
            "relevance_threshold": float(protocol.relevance_threshold),
            "cutoffs": list(cutoffs),
            "locked_test_used": False,
        },
        "recommendation_integrity": {
            "popularity": (popularity_integrity),
            "genre_personalized": (genre_integrity),
        },
        "ranking_metrics_at_10": (comparison_at_10),
        "warm_user_metrics_at_20": (warm_at_20),
        "beyond_accuracy": diagnostics["models"],
        "routing": {
            "strategies": sorted(expected_strategies),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("=== RECOMMENDATION INTEGRITY ===")
    print(
        json.dumps(
            summary["recommendation_integrity"],
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=== RANKING METRICS @10 ===")

    ranking_frame = comparison.loc[
        comparison["k"].eq(10),
        [
            "model",
            "precision",
            "recall",
            "ndcg",
            "map",
            "hit_rate",
        ],
    ]

    print(
        ranking_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=== WARM USERS @20 ===")

    warm_frame = cohort_metrics.loc[
        (cohort_metrics["segment"].eq("warm")) & cohort_metrics["k"].eq(20),
        [
            "model",
            "users",
            "precision",
            "recall",
            "ndcg",
            "map",
            "hit_rate",
        ],
    ]

    print(
        warm_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=== BEYOND-ACCURACY ===")

    for model_name in (
        "popularity",
        "genre_personalized",
    ):
        values = diagnostics["models"][model_name]

        print(
            f"{model_name}: "
            f"coverage="
            f"{values['catalogue_coverage']:.6f}, "
            f"long_tail="
            f"{values['long_tail_share']:.6f}, "
            f"avg_popularity="
            f"{values['average_recommendation_popularity']:.2f}, "
            f"cold_item_exposure="
            f"{values['cold_item_exposure_share']:.6f}"
        )

    print()
    print(f"summary={SUMMARY_PATH}")
    print()
    print("DAY-02 BASELINE BENCHMARK: PASS")


if __name__ == "__main__":
    main()
