"""Candidate generation for Day-03 collaborative recommenders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recomx.models.item_knn import ItemNeighborhoods, recommend_item_knn
from recomx.models.latent_svd import LatentSVDModel
from recomx.models.popularity import PopularityModel
from recomx.models.sparse_interactions import SparseInteractionMatrix

_OUTPUT_COLUMNS = ["userId", "movieId", "score", "recommendation_rank", "strategy"]


@dataclass(frozen=True)
class CandidateGenerationSummary:
    """Audit information for a collaborative candidate run."""

    users: int
    warm_users: int
    cold_users: int
    collaborative_users: int
    fallback_users: int
    recommendation_rows: int
    users_with_full_top_k: int
    seen_item_violations: int


def _popularity_fallback(
    user_id: int,
    popularity: PopularityModel,
    seen_items: set[int],
    top_k: int,
) -> pd.DataFrame:
    frame = popularity.recommend(user_id=user_id, seen_items=seen_items, k=top_k)
    output = frame[["userId", "movieId", "popularity_score", "recommendation_rank"]].copy()
    output = output.rename(columns={"popularity_score": "score"})
    output["score"] = output["score"].astype(float)
    output["strategy"] = "popularity_fallback"
    return output[_OUTPUT_COLUMNS]


def _collaborative_frame(
    user_id: int,
    item_ids: np.ndarray,
    scores: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": int(user_id),
            "movieId": item_ids.astype(np.int64, copy=False),
            "score": scores.astype(float, copy=False),
            "recommendation_rank": np.arange(1, len(item_ids) + 1),
            "strategy": strategy,
        }
    )


def generate_item_knn_candidates(
    user_ids: list[int],
    interactions: SparseInteractionMatrix,
    neighborhoods: ItemNeighborhoods,
    popularity: PopularityModel,
    all_seen_items: dict[int, set[int]],
    top_k: int = 20,
) -> tuple[pd.DataFrame, CandidateGenerationSummary]:
    """Generate Item-KNN candidates with popularity fallback."""

    return _generate_candidates(
        user_ids=user_ids,
        interactions=interactions,
        popularity=popularity,
        all_seen_items=all_seen_items,
        top_k=top_k,
        strategy="item_knn",
        scorer=lambda user_id: recommend_item_knn(
            user_id, interactions, neighborhoods, top_k=top_k
        ),
    )


def generate_svd_candidates(
    user_ids: list[int],
    interactions: SparseInteractionMatrix,
    model: LatentSVDModel,
    popularity: PopularityModel,
    all_seen_items: dict[int, set[int]],
    top_k: int = 20,
    batch_size: int = 4096,
) -> tuple[pd.DataFrame, CandidateGenerationSummary]:
    """Generate latent-SVD candidates with popularity fallback."""

    return _generate_candidates(
        user_ids=user_ids,
        interactions=interactions,
        popularity=popularity,
        all_seen_items=all_seen_items,
        top_k=top_k,
        strategy="latent_svd",
        scorer=lambda user_id: model.recommend(
            user_id,
            interactions,
            top_k=top_k,
            batch_size=batch_size,
        ),
    )


def _generate_candidates(
    user_ids: list[int],
    interactions: SparseInteractionMatrix,
    popularity: PopularityModel,
    all_seen_items: dict[int, set[int]],
    top_k: int,
    strategy: str,
    scorer,
) -> tuple[pd.DataFrame, CandidateGenerationSummary]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    outputs: list[pd.DataFrame] = []
    warm_users = 0
    collaborative_users = 0
    fallback_users = 0

    for raw_user_id in user_ids:
        user_id = int(raw_user_id)
        seen = all_seen_items.get(user_id, set())
        is_warm = interactions.user_index(user_id) is not None
        if is_warm:
            warm_users += 1
            item_ids, scores = scorer(user_id)
            if len(item_ids):
                frame = _collaborative_frame(user_id, item_ids, scores, strategy)
                # Collaborative models encode positive history only. This second
                # guard enforces the stronger contract: remove every train-seen item.
                frame = frame.loc[~frame["movieId"].isin(seen)].head(top_k).copy()
                frame["recommendation_rank"] = np.arange(1, len(frame) + 1)
                if len(frame) == top_k:
                    collaborative_users += 1
                    outputs.append(frame)
                    continue

        fallback_users += 1
        outputs.append(_popularity_fallback(user_id, popularity, seen, top_k))

    recommendations = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(columns=_OUTPUT_COLUMNS)
    )
    counts = recommendations.groupby("userId").size()
    violations = 0
    for user_id, group in recommendations.groupby("userId", sort=False):
        violations += len(
            set(group["movieId"].astype(int)).intersection(all_seen_items.get(int(user_id), set()))
        )

    summary = CandidateGenerationSummary(
        users=len(user_ids),
        warm_users=warm_users,
        cold_users=len(user_ids) - warm_users,
        collaborative_users=collaborative_users,
        fallback_users=fallback_users,
        recommendation_rows=len(recommendations),
        users_with_full_top_k=int(counts.eq(top_k).sum()),
        seen_item_violations=violations,
    )
    return recommendations, summary
