"""Top-K ranking metrics for RecomX offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import pandas as pd


@dataclass(frozen=True)
class UserRankingMetrics:
    """Ranking metrics for one user at one cutoff."""

    user_id: int
    k: int
    precision: float
    recall: float
    ndcg: float
    average_precision: float
    hit_rate: float
    relevant_items: int
    recommended_items: int
    hits: int


def _validate_k(k: int) -> None:
    """Validate ranking cutoff."""

    if k <= 0:
        raise ValueError("k must be positive.")


def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Precision@K."""

    _validate_k(k)

    top_k = recommended[:k]

    if not top_k:
        return 0.0

    hits = sum(item in relevant for item in top_k)

    # Precision@K uses K as denominator so models that return fewer
    # recommendations do not receive an artificial advantage.
    return hits / k


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Recall@K."""

    _validate_k(k)

    if not relevant:
        return 0.0

    hits = sum(item in relevant for item in recommended[:k])

    return hits / len(relevant)


def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Return one when at least one relevant item occurs in Top-K."""

    _validate_k(k)

    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute binary-relevance NDCG@K."""

    _validate_k(k)

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(
        recommended[:k],
        start=1,
    ):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(
        len(relevant),
        k,
    )

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Average Precision@K for binary relevance."""

    _validate_k(k)

    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(
        recommended[:k],
        start=1,
    ):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    denominator = min(
        len(relevant),
        k,
    )

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def evaluate_user_at_k(
    user_id: int,
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> UserRankingMetrics:
    """Evaluate one user's ranked recommendations."""

    _validate_k(k)

    top_k = recommended[:k]

    hits = sum(item in relevant for item in top_k)

    return UserRankingMetrics(
        user_id=int(user_id),
        k=int(k),
        precision=precision_at_k(
            recommended,
            relevant,
            k,
        ),
        recall=recall_at_k(
            recommended,
            relevant,
            k,
        ),
        ndcg=ndcg_at_k(
            recommended,
            relevant,
            k,
        ),
        average_precision=average_precision_at_k(
            recommended,
            relevant,
            k,
        ),
        hit_rate=hit_rate_at_k(
            recommended,
            relevant,
            k,
        ),
        relevant_items=len(relevant),
        recommended_items=len(top_k),
        hits=hits,
    )


def build_ground_truth_map(
    ground_truth: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build user-to-relevant-item mapping."""

    required = {"userId", "movieId"}
    missing = required.difference(ground_truth.columns)

    if missing:
        raise ValueError(f"ground_truth missing required columns: {sorted(missing)}")

    if ground_truth.empty:
        raise ValueError("ground_truth cannot be empty.")

    pairs = ground_truth[["userId", "movieId"]].drop_duplicates()

    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in pairs.groupby(
            "userId",
            sort=False,
        )
    }


def build_recommendation_map(
    recommendations: pd.DataFrame,
) -> dict[int, list[int]]:
    """Build ordered user-to-recommendation mapping."""

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

    ordered = recommendations.sort_values(
        [
            "userId",
            "recommendation_rank",
            "movieId",
        ],
        kind="mergesort",
    )

    duplicated_ranks = ordered.duplicated(
        subset=[
            "userId",
            "recommendation_rank",
        ]
    )

    if duplicated_ranks.any():
        raise ValueError("Duplicate recommendation ranks detected.")

    duplicated_items = ordered.duplicated(
        subset=[
            "userId",
            "movieId",
        ]
    )

    if duplicated_items.any():
        raise ValueError("Duplicate recommended items detected.")

    return {
        int(user_id): (group["movieId"].astype(int).tolist())
        for user_id, group in ordered.groupby(
            "userId",
            sort=False,
        )
    }


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cutoffs: tuple[int, ...] = (
        5,
        10,
        20,
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ranked recommendations at multiple Top-K cutoffs.

    Returns:
        user_metrics:
            One row per evaluable user and cutoff.

        aggregate_metrics:
            Macro-averaged ranking metrics at each cutoff.
    """

    if not cutoffs:
        raise ValueError("cutoffs cannot be empty.")

    for k in cutoffs:
        _validate_k(k)

    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique.")

    ground_truth_map = build_ground_truth_map(ground_truth)

    recommendation_map = build_recommendation_map(recommendations)

    missing_users = set(ground_truth_map).difference(recommendation_map)

    if missing_users:
        raise ValueError(f"Recommendations missing for {len(missing_users)} ground-truth users.")

    rows: list[dict[str, int | float]] = []

    for user_id in sorted(ground_truth_map):
        relevant = ground_truth_map[user_id]
        recommended = recommendation_map[user_id]

        for k in cutoffs:
            metrics = evaluate_user_at_k(
                user_id=user_id,
                recommended=recommended,
                relevant=relevant,
                k=k,
            )

            rows.append(
                {
                    "userId": metrics.user_id,
                    "k": metrics.k,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "ndcg": metrics.ndcg,
                    "map": (metrics.average_precision),
                    "hit_rate": metrics.hit_rate,
                    "relevant_items": (metrics.relevant_items),
                    "recommended_items": (metrics.recommended_items),
                    "hits": metrics.hits,
                }
            )

    user_metrics = pd.DataFrame(rows)

    aggregate_metrics = (
        user_metrics.groupby(
            "k",
            as_index=False,
        )
        .agg(
            users=("userId", "nunique"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            ndcg=("ndcg", "mean"),
            map=("map", "mean"),
            hit_rate=("hit_rate", "mean"),
            mean_relevant_items=(
                "relevant_items",
                "mean",
            ),
            mean_hits=("hits", "mean"),
        )
        .sort_values(
            "k",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return (
        user_metrics,
        aggregate_metrics,
    )
