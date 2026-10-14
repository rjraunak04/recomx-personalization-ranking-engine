"""Run RecomX Day-02 recommendation diagnostics."""

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
from recomx.evaluation.protocol import (
    DEFAULT_EVALUATION_PROTOCOL,
)

GROUND_TRUTH_PATH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")

COHORT_PATH = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")

POPULARITY_RECOMMENDATIONS_PATH = Path("reports/baselines/popularity/recommendations_top20.parquet")

GENRE_RECOMMENDATIONS_PATH = Path("reports/baselines/genre/recommendations_top20.parquet")

ITEM_POPULARITY_PATH = Path("reports/baselines/popularity/item_popularity.parquet")

TRAIN_PATH = Path("data/processed/ratings_train.parquet")

MOVIES_PATH = Path("data/interim/movies.parquet")

OUTPUT_DIR = Path("reports/baselines/diagnostics")

SUMMARY_PATH = OUTPUT_DIR / "diagnostics_summary.json"

COHORT_METRICS_PATH = OUTPUT_DIR / "cohort_metrics.csv"

STRATEGY_METRICS_PATH = OUTPUT_DIR / "strategy_metrics.csv"


def _model_diagnostics(
    recommendations: pd.DataFrame,
    catalogue_items: set[int],
    eligible_items: set[int],
    long_tail_items: set[int],
    train_items: set[int],
    item_scores: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate beyond-accuracy diagnostics for one model."""

    cold_count, cold_share = cold_item_exposure(
        recommendations,
        train_items,
    )

    return {
        "unique_recommended_items": int(recommendations["movieId"].nunique()),
        "catalogue_coverage": float(
            catalogue_coverage(
                recommendations,
                catalogue_items,
            )
        ),
        "eligible_item_coverage": float(
            recommendation_coverage(
                recommendations,
                eligible_items,
            )
        ),
        "average_recommendation_popularity": float(
            average_recommendation_popularity(
                recommendations,
                item_scores,
            )
        ),
        "long_tail_share": float(
            long_tail_share(
                recommendations,
                long_tail_items,
            )
        ),
        "cold_item_recommendation_rows": int(cold_count),
        "cold_item_exposure_share": float(cold_share),
    }


def main() -> None:
    """Run CP-07 recommendation diagnostics."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    required_paths = (
        GROUND_TRUTH_PATH,
        COHORT_PATH,
        POPULARITY_RECOMMENDATIONS_PATH,
        GENRE_RECOMMENDATIONS_PATH,
        ITEM_POPULARITY_PATH,
        TRAIN_PATH,
        MOVIES_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    cutoffs = tuple(sorted(protocol.top_k))

    print("RecomX CP-07 Recommendation Diagnostics")
    print("=" * 72)
    print("locked_test=NOT READ")
    print("long_tail_definition=outside top 10% popularity head")
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

    popularity_recommendations = pd.read_parquet(POPULARITY_RECOMMENDATIONS_PATH)

    genre_recommendations = pd.read_parquet(GENRE_RECOMMENDATIONS_PATH)

    item_scores = pd.read_parquet(
        ITEM_POPULARITY_PATH,
        columns=[
            "movieId",
            "popularity_score",
        ],
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "movieId",
        ],
    )

    movies = pd.read_parquet(
        MOVIES_PATH,
        columns=[
            "movieId",
        ],
    )

    catalogue_items = set(movies["movieId"].astype(int))

    eligible_items = set(item_scores["movieId"].astype(int))

    train_items = set(train["movieId"].astype(int))

    long_tail_items = build_long_tail_items(
        item_scores,
        head_fraction=0.10,
    )

    popularity_diagnostics = _model_diagnostics(
        recommendations=(popularity_recommendations),
        catalogue_items=catalogue_items,
        eligible_items=eligible_items,
        long_tail_items=long_tail_items,
        train_items=train_items,
        item_scores=item_scores,
    )

    genre_diagnostics = _model_diagnostics(
        recommendations=(genre_recommendations),
        catalogue_items=catalogue_items,
        eligible_items=eligible_items,
        long_tail_items=long_tail_items,
        train_items=train_items,
        item_scores=item_scores,
    )

    cohort_frames: list[pd.DataFrame] = []

    for model_name, recommendations in (
        (
            "popularity",
            popularity_recommendations,
        ),
        (
            "genre_personalized",
            genre_recommendations,
        ),
    ):
        frame = evaluate_by_cohort(
            recommendations=recommendations,
            ground_truth=ground_truth,
            cohorts=cohorts,
            cutoffs=cutoffs,
        )

        frame.insert(
            0,
            "model",
            model_name,
        )

        cohort_frames.append(frame)

    cohort_metrics = (
        pd.concat(
            cohort_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "k",
                "segment",
                "model",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    strategy_metrics = (
        evaluate_by_strategy(
            recommendations=(genre_recommendations),
            ground_truth=ground_truth,
            cutoffs=cutoffs,
        )
        .sort_values(
            [
                "k",
                "strategy",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_metrics.to_csv(
        COHORT_METRICS_PATH,
        index=False,
    )

    strategy_metrics.to_csv(
        STRATEGY_METRICS_PATH,
        index=False,
    )

    summary = {
        "long_tail_head_fraction": 0.10,
        "catalogue_items": len(catalogue_items),
        "eligible_popularity_items": len(eligible_items),
        "train_items": len(train_items),
        "long_tail_items": len(long_tail_items),
        "models": {
            "popularity": (popularity_diagnostics),
            "genre_personalized": (genre_diagnostics),
        },
    }

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

    print("=== BEYOND-ACCURACY ===")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=== COHORT METRICS ===")
    print(
        cohort_metrics[
            [
                "model",
                "segment",
                "k",
                "users",
                "precision",
                "recall",
                "ndcg",
                "map",
                "hit_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=== GENRE ROUTING STRATEGY ===")
    print(
        strategy_metrics[
            [
                "strategy",
                "k",
                "users",
                "precision",
                "recall",
                "ndcg",
                "map",
                "hit_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"summary={SUMMARY_PATH}")
    print(f"cohort_metrics={COHORT_METRICS_PATH}")
    print(f"strategy_metrics={STRATEGY_METRICS_PATH}")
    print()
    print("CP-07 RECOMMENDATION DIAGNOSTICS: PASS")


if __name__ == "__main__":
    main()
