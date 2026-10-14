"""Beyond-accuracy recommendation diagnostics for RecomX."""

from __future__ import annotations

import pandas as pd

from recomx.evaluation.metrics import evaluate_recommendations


def validate_recommendations(
    recommendations: pd.DataFrame,
) -> None:
    """Validate diagnostic recommendation input."""

    required = {
        "userId",
        "movieId",
        "recommendation_rank",
    }

    missing = required.difference(recommendations.columns)

    if missing:
        raise ValueError(f"recommendations missing required columns: {sorted(missing)}")

    if recommendations.empty:
        raise ValueError("recommendations cannot be empty.")


def catalogue_coverage(
    recommendations: pd.DataFrame,
    catalogue_items: set[int],
) -> float:
    """Fraction of catalogue items exposed by recommendations."""

    validate_recommendations(recommendations)

    if not catalogue_items:
        raise ValueError("catalogue_items cannot be empty.")

    recommended_items = set(recommendations["movieId"].astype(int))

    return len(recommended_items.intersection(catalogue_items)) / len(catalogue_items)


def recommendation_coverage(
    recommendations: pd.DataFrame,
    eligible_items: set[int],
) -> float:
    """Fraction of recommendation-eligible items exposed."""

    validate_recommendations(recommendations)

    if not eligible_items:
        raise ValueError("eligible_items cannot be empty.")

    recommended_items = set(recommendations["movieId"].astype(int))

    return len(recommended_items.intersection(eligible_items)) / len(eligible_items)


def build_long_tail_items(
    item_scores: pd.DataFrame,
    head_fraction: float = 0.10,
) -> set[int]:
    """Define long-tail items as items outside the popularity head."""

    required = {
        "movieId",
        "popularity_score",
    }

    missing = required.difference(item_scores.columns)

    if missing:
        raise ValueError(f"item_scores missing required columns: {sorted(missing)}")

    if item_scores.empty:
        raise ValueError("item_scores cannot be empty.")

    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be between 0 and 1.")

    ordered = item_scores.sort_values(
        [
            "popularity_score",
            "movieId",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    head_size = max(
        1,
        int(len(ordered) * head_fraction),
    )

    return set(ordered.iloc[head_size:]["movieId"].astype(int))


def long_tail_share(
    recommendations: pd.DataFrame,
    long_tail_items: set[int],
) -> float:
    """Fraction of recommendation rows belonging to the long tail."""

    validate_recommendations(recommendations)

    if not long_tail_items:
        return 0.0

    return float(recommendations["movieId"].astype(int).isin(long_tail_items).mean())


def average_recommendation_popularity(
    recommendations: pd.DataFrame,
    item_scores: pd.DataFrame,
) -> float:
    """Mean train popularity count of recommended items."""

    validate_recommendations(recommendations)

    required = {
        "movieId",
        "popularity_score",
    }

    missing = required.difference(item_scores.columns)

    if missing:
        raise ValueError(f"item_scores missing required columns: {sorted(missing)}")

    popularity = item_scores[
        [
            "movieId",
            "popularity_score",
        ]
    ].drop_duplicates("movieId")

    merged = recommendations[["movieId"]].merge(
        popularity,
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    return float(merged["popularity_score"].fillna(0.0).mean())


def cold_item_exposure(
    recommendations: pd.DataFrame,
    train_items: set[int],
) -> tuple[int, float]:
    """Count and share recommended items unseen anywhere in training."""

    validate_recommendations(recommendations)

    movie_ids = recommendations["movieId"].astype(int)

    cold_mask = ~movie_ids.isin(train_items)

    return (
        int(cold_mask.sum()),
        float(cold_mask.mean()),
    )


def evaluate_by_cohort(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cohorts: pd.DataFrame,
    cutoffs: tuple[int, ...] = (
        5,
        10,
        20,
    ),
) -> pd.DataFrame:
    """Evaluate Top-K accuracy separately for warm and cold users."""

    required = {
        "userId",
        "is_warm_user",
        "is_cold_user",
        "is_evaluable",
    }

    missing = required.difference(cohorts.columns)

    if missing:
        raise ValueError(f"cohorts missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []

    segment_masks = {
        "warm": (cohorts["is_warm_user"] & cohorts["is_evaluable"]),
        "cold": (cohorts["is_cold_user"] & cohorts["is_evaluable"]),
    }

    for segment, mask in segment_masks.items():
        user_ids = set(
            cohorts.loc[
                mask,
                "userId",
            ].astype(int)
        )

        if not user_ids:
            continue

        segment_ground_truth = ground_truth.loc[ground_truth["userId"].isin(user_ids)]

        segment_recommendations = recommendations.loc[recommendations["userId"].isin(user_ids)]

        _, aggregate = evaluate_recommendations(
            recommendations=(segment_recommendations),
            ground_truth=(segment_ground_truth),
            cutoffs=cutoffs,
        )

        aggregate.insert(
            0,
            "segment",
            segment,
        )

        rows.append(aggregate)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def evaluate_by_strategy(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cutoffs: tuple[int, ...] = (
        5,
        10,
        20,
    ),
) -> pd.DataFrame:
    """Evaluate a hybrid model separately by routing strategy."""

    if "strategy" not in recommendations.columns:
        raise ValueError("recommendations missing strategy column.")

    user_strategy = recommendations[["userId", "strategy"]].drop_duplicates()

    strategy_counts = user_strategy.groupby("userId")["strategy"].nunique()

    if strategy_counts.gt(1).any():
        raise ValueError("A user cannot have multiple strategies.")

    rows: list[pd.DataFrame] = []

    for strategy in sorted(user_strategy["strategy"].unique()):
        users = set(
            user_strategy.loc[
                user_strategy["strategy"].eq(strategy),
                "userId",
            ].astype(int)
        )

        strategy_ground_truth = ground_truth.loc[ground_truth["userId"].isin(users)]

        strategy_recommendations = recommendations.loc[recommendations["userId"].isin(users)]

        if strategy_ground_truth.empty:
            continue

        _, aggregate = evaluate_recommendations(
            recommendations=(strategy_recommendations),
            ground_truth=(strategy_ground_truth),
            cutoffs=cutoffs,
        )

        aggregate.insert(
            0,
            "strategy",
            str(strategy),
        )

        rows.append(aggregate)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )
