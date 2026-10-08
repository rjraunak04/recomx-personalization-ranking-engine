"""Run the RecomX train-only popularity recommendation baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL
from recomx.evaluation.relevance import RelevancePolicy
from recomx.models.popularity import (
    PopularityModel,
    build_seen_item_map,
    recommend_for_users,
)

TRAIN_PATH = Path("data/processed/ratings_train.parquet")

COHORT_PATH = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")

OUTPUT_DIR = Path("reports/baselines/popularity")

SCORES_PATH = OUTPUT_DIR / "item_popularity.parquet"

RECOMMENDATIONS_PATH = OUTPUT_DIR / "recommendations_top20.parquet"

SUMMARY_PATH = OUTPUT_DIR / "popularity_summary.json"


def main() -> None:
    """Fit and generate the Day-02 popularity baseline."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing train partition: {TRAIN_PATH}")

    if not COHORT_PATH.is_file():
        raise FileNotFoundError(f"Missing CP-03 validation cohort artifact: {COHORT_PATH}")

    max_k = max(protocol.top_k)

    print("RecomX CP-04 Popularity Baseline")
    print("=" * 72)
    print(f"train={TRAIN_PATH}")
    print(f"cohorts={COHORT_PATH}")
    print("validation_ratings=NOT USED FOR MODEL FIT")
    print("locked_test=NOT READ")
    print(f"relevance_threshold={protocol.relevance_threshold:.1f}")
    print(f"max_k={max_k}")
    print()

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "userId",
            "movieId",
            "rating",
        ],
    )

    cohorts = pd.read_parquet(
        COHORT_PATH,
        columns=[
            "userId",
            "is_evaluable",
        ],
    )

    evaluable_users = (
        cohorts.loc[
            cohorts["is_evaluable"],
            "userId",
        ]
        .astype(int)
        .tolist()
    )

    policy = RelevancePolicy(minimum_rating=(protocol.relevance_threshold))

    model = PopularityModel.fit(
        train=train,
        policy=policy,
    )

    seen_item_map = build_seen_item_map(train)

    recommendations = recommend_for_users(
        model=model,
        user_ids=evaluable_users,
        seen_item_map=seen_item_map,
        k=max_k,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.item_scores.to_parquet(
        SCORES_PATH,
        index=False,
    )

    recommendations.to_parquet(
        RECOMMENDATIONS_PATH,
        index=False,
    )

    per_user_counts = recommendations.groupby("userId").size()

    seen_violations = 0

    for user_id, group in recommendations.groupby(
        "userId",
        sort=False,
    ):
        seen = seen_item_map.get(
            int(user_id),
            set(),
        )

        seen_violations += len(set(group["movieId"].astype(int)).intersection(seen))

    summary = {
        "training_interactions": int(len(train)),
        "training_users": int(train["userId"].nunique()),
        "relevant_popularity_items": int(len(model.item_scores)),
        "evaluable_users": int(len(evaluable_users)),
        "recommendation_rows": int(len(recommendations)),
        "max_k": int(max_k),
        "users_with_full_top_k": int(per_user_counts.eq(max_k).sum()),
        "users_without_full_top_k": int(len(evaluable_users) - per_user_counts.eq(max_k).sum()),
        "seen_item_violations": int(seen_violations),
        "top_global_movie_ids": [
            int(value) for value in (model.item_scores["movieId"].head(10).tolist())
        ],
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

    for key, value in summary.items():
        print(f"{key}={value}")

    print()
    print(f"scores={SCORES_PATH}")
    print(f"recommendations={RECOMMENDATIONS_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print()
    print("CP-04 POPULARITY BASELINE: PASS")


if __name__ == "__main__":
    main()
