"""Evaluate RecomX Day-02 recommendation baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.metrics import (
    evaluate_recommendations,
)
from recomx.evaluation.protocol import (
    DEFAULT_EVALUATION_PROTOCOL,
)

GROUND_TRUTH_PATH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")

POPULARITY_PATH = Path("reports/baselines/popularity/recommendations_top20.parquet")

GENRE_PATH = Path("reports/baselines/genre/recommendations_top20.parquet")

OUTPUT_DIR = Path("reports/baselines/evaluation")

COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

USER_METRICS_PATH = OUTPUT_DIR / "user_metrics.parquet"

SUMMARY_PATH = OUTPUT_DIR / "ranking_metrics.json"


def _evaluate_model(
    model_name: str,
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cutoffs: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model and attach its name."""

    user_metrics, aggregate = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        cutoffs=cutoffs,
    )

    user_metrics.insert(
        0,
        "model",
        model_name,
    )

    aggregate.insert(
        0,
        "model",
        model_name,
    )

    return user_metrics, aggregate


def main() -> None:
    """Evaluate popularity and genre baselines."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    required_paths = (
        GROUND_TRUTH_PATH,
        POPULARITY_PATH,
        GENRE_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    cutoffs = tuple(sorted(protocol.top_k))

    print("RecomX CP-06 Top-K Ranking Evaluation")
    print("=" * 72)
    print(f"ground_truth={GROUND_TRUTH_PATH}")
    print(f"popularity={POPULARITY_PATH}")
    print(f"genre={GENRE_PATH}")
    print(f"cutoffs={cutoffs}")
    print("locked_test=NOT READ")
    print()

    ground_truth = pd.read_parquet(
        GROUND_TRUTH_PATH,
        columns=[
            "userId",
            "movieId",
        ],
    )

    popularity = pd.read_parquet(
        POPULARITY_PATH,
        columns=[
            "userId",
            "movieId",
            "recommendation_rank",
        ],
    )

    genre = pd.read_parquet(
        GENRE_PATH,
        columns=[
            "userId",
            "movieId",
            "recommendation_rank",
        ],
    )

    popularity_user, popularity_aggregate = _evaluate_model(
        model_name="popularity",
        recommendations=popularity,
        ground_truth=ground_truth,
        cutoffs=cutoffs,
    )

    genre_user, genre_aggregate = _evaluate_model(
        model_name="genre_personalized",
        recommendations=genre,
        ground_truth=ground_truth,
        cutoffs=cutoffs,
    )

    user_metrics = pd.concat(
        [
            popularity_user,
            genre_user,
        ],
        ignore_index=True,
    )

    comparison = (
        pd.concat(
            [
                popularity_aggregate,
                genre_aggregate,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "k",
                "model",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    user_metrics.to_parquet(
        USER_METRICS_PATH,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    summary: dict[str, object] = {
        "evaluation_users": int(ground_truth["userId"].nunique()),
        "ground_truth_interactions": int(len(ground_truth)),
        "cutoffs": list(cutoffs),
        "models": {},
    }

    models_summary: dict[
        str,
        dict[str, dict[str, float | int]],
    ] = {}

    for model_name, model_frame in comparison.groupby(
        "model",
        sort=True,
    ):
        cutoff_summary: dict[
            str,
            dict[str, float | int],
        ] = {}

        for row in model_frame.itertuples(index=False):
            cutoff_summary[str(int(row.k))] = {
                "users": int(row.users),
                "precision": float(row.precision),
                "recall": float(row.recall),
                "ndcg": float(row.ndcg),
                "map": float(row.map),
                "hit_rate": float(row.hit_rate),
                "mean_relevant_items": float(row.mean_relevant_items),
                "mean_hits": float(row.mean_hits),
            }

        models_summary[str(model_name)] = cutoff_summary

    summary["models"] = models_summary

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

    display_columns = [
        "model",
        "k",
        "users",
        "precision",
        "recall",
        "ndcg",
        "map",
        "hit_rate",
        "mean_hits",
    ]

    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"comparison={COMPARISON_PATH}")
    print(f"user_metrics={USER_METRICS_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print()
    print("CP-06 TOP-K RANKING EVALUATION: PASS")


if __name__ == "__main__":
    main()
