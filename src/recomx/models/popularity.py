"""Train-only popularity recommendation baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recomx.evaluation.relevance import RelevancePolicy


@dataclass(frozen=True)
class PopularityModel:
    """Global item popularity learned from relevant training feedback."""

    item_scores: pd.DataFrame

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        policy: RelevancePolicy | None = None,
    ) -> PopularityModel:
        """Fit popularity scores using relevant training interactions only."""

        required = {"userId", "movieId", "rating"}
        missing = required.difference(train.columns)

        if missing:
            raise ValueError(f"train missing required columns: {sorted(missing)}")

        if train.empty:
            raise ValueError("train cannot be empty.")

        active_policy = policy or RelevancePolicy()
        active_policy.validate()

        relevant = active_policy.relevant_interactions(train)

        if relevant.empty:
            raise ValueError("No relevant training interactions available.")

        scores = (
            relevant.groupby(
                "movieId",
                as_index=False,
            )
            .agg(
                popularity_score=("userId", "size"),
                relevant_users=("userId", "nunique"),
            )
            .sort_values(
                [
                    "popularity_score",
                    "relevant_users",
                    "movieId",
                ],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        scores["popularity_rank"] = range(1, len(scores) + 1)

        return cls(item_scores=scores)

    def recommend(
        self,
        user_id: int,
        seen_items: set[int] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        """Return deterministic Top-K unseen popular items."""

        if k <= 0:
            raise ValueError("k must be positive.")

        seen = seen_items or set()

        candidates = self.item_scores.loc[~self.item_scores["movieId"].isin(seen)].head(k)

        recommendations = candidates[
            [
                "movieId",
                "popularity_score",
                "popularity_rank",
            ]
        ].copy()

        recommendations.insert(
            0,
            "userId",
            user_id,
        )

        recommendations["recommendation_rank"] = range(
            1,
            len(recommendations) + 1,
        )

        return recommendations.reset_index(drop=True)


def build_seen_item_map(
    train: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build each training user's seen-item set."""

    required = {"userId", "movieId"}
    missing = required.difference(train.columns)

    if missing:
        raise ValueError(f"train missing required columns: {sorted(missing)}")

    if train.empty:
        raise ValueError("train cannot be empty.")

    pairs = train[["userId", "movieId"]].drop_duplicates()

    return {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in pairs.groupby(
            "userId",
            sort=False,
        )
    }


def recommend_for_users(
    model: PopularityModel,
    user_ids: list[int],
    seen_item_map: dict[int, set[int]],
    k: int = 20,
) -> pd.DataFrame:
    """Generate Top-K popularity recommendations for many users."""

    if k <= 0:
        raise ValueError("k must be positive.")

    outputs: list[pd.DataFrame] = []

    for user_id in user_ids:
        recommendations = model.recommend(
            user_id=int(user_id),
            seen_items=seen_item_map.get(
                int(user_id),
                set(),
            ),
            k=k,
        )

        outputs.append(recommendations)

    if not outputs:
        return pd.DataFrame(
            columns=[
                "userId",
                "movieId",
                "popularity_score",
                "popularity_rank",
                "recommendation_rank",
            ]
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )
