"""Run the RecomX genre-aware personalized baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL
from recomx.evaluation.relevance import RelevancePolicy
from recomx.models.genre_profile import (
    GenreProfileModel,
    recommend_genre_for_users,
)
from recomx.models.popularity import (
    build_seen_item_map,
)

TRAIN_PATH = Path("data/processed/ratings_train.parquet")

MOVIES_PATH = Path("data/interim/movies.parquet")

COHORT_PATH = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")

OUTPUT_DIR = Path("reports/baselines/genre")

RECOMMENDATIONS_PATH = OUTPUT_DIR / "recommendations_top20.parquet"

PROFILES_PATH = OUTPUT_DIR / "user_genre_profiles.parquet"

SUMMARY_PATH = OUTPUT_DIR / "genre_summary.json"


def main() -> None:
    """Fit and run the genre-aware baseline."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    for path in (
        TRAIN_PATH,
        MOVIES_PATH,
        COHORT_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    max_k = max(protocol.top_k)

    print("RecomX CP-05 Genre Personalized Baseline")
    print("=" * 72)
    print(f"train={TRAIN_PATH}")
    print(f"movies={MOVIES_PATH}")
    print(f"cohorts={COHORT_PATH}")
    print("validation_ratings=NOT USED FOR MODEL FIT")
    print("locked_test=NOT READ")
    print(f"relevance_threshold={protocol.relevance_threshold:.1f}")
    print("popularity_weight=0.05")
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

    movies = pd.read_parquet(
        MOVIES_PATH,
        columns=[
            "movieId",
            "title",
            "genres",
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

    model = GenreProfileModel.fit(
        train=train,
        movies=movies,
        policy=policy,
        popularity_weight=0.05,
    )

    seen_item_map = build_seen_item_map(train)

    recommendations = recommend_genre_for_users(
        model=model,
        user_ids=evaluable_users,
        seen_item_map=seen_item_map,
        k=max_k,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_parquet(
        RECOMMENDATIONS_PATH,
        index=False,
    )

    model.user_genre_profiles.to_parquet(
        PROFILES_PATH,
        index=False,
    )

    per_user_counts = recommendations.groupby("userId").size()

    strategy_counts = {
        str(key): int(value)
        for key, value in (
            recommendations[["userId", "strategy"]]
            .drop_duplicates("userId")["strategy"]
            .value_counts()
            .items()
        )
    }

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
        "catalogue_items": int(movies["movieId"].nunique()),
        "profiled_users": int(model.user_genre_profiles["userId"].nunique()),
        "profiled_genres": int(model.user_genre_profiles["genres"].nunique()),
        "evaluable_users": int(len(evaluable_users)),
        "recommendation_rows": int(len(recommendations)),
        "max_k": int(max_k),
        "users_with_full_top_k": int(per_user_counts.eq(max_k).sum()),
        "users_without_full_top_k": int(len(evaluable_users) - per_user_counts.eq(max_k).sum()),
        "seen_item_violations": int(seen_violations),
        "strategy_user_counts": (strategy_counts),
        "popularity_weight": 0.05,
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
    print(f"recommendations={RECOMMENDATIONS_PATH}")
    print(f"profiles={PROFILES_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print()
    print("CP-05 GENRE PERSONALIZED BASELINE: PASS")


if __name__ == "__main__":
    main()
