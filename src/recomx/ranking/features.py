"""Leakage-safe ranking feature engineering for RecomX Day-05."""

from __future__ import annotations

import pandas as pd

SOURCE_NAMES = ("item_knn", "latent_svd", "genre_content", "popularity")


def add_ranking_features(
    ranking: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Create train-only user/item statistics plus retrieval features."""
    required = {"userId", "movieId", "score", "candidate_rank", "sources", "label"}
    missing = required.difference(ranking.columns)
    if missing:
        raise ValueError(f"Ranking dataset missing columns: {sorted(missing)}")
    train_required = {"userId", "movieId", "rating"}
    train_missing = train_required.difference(train.columns)
    if train_missing:
        raise ValueError(f"Train data missing columns: {sorted(train_missing)}")

    frame = ranking.copy()
    user_stats = (
        train.groupby("userId", as_index=False)
        .agg(
            user_history_count=("movieId", "size"),
            user_mean_rating=("rating", "mean"),
        )
    )
    item_stats = (
        train.groupby("movieId", as_index=False)
        .agg(
            item_interaction_count=("userId", "size"),
            item_mean_rating=("rating", "mean"),
        )
    )
    frame = frame.merge(user_stats, on="userId", how="left")
    frame = frame.merge(item_stats, on="movieId", how="left")

    frame["user_history_count"] = frame["user_history_count"].fillna(0).astype("int32")
    frame["user_mean_rating"] = frame["user_mean_rating"].fillna(0.0).astype("float32")
    frame["item_interaction_count"] = frame["item_interaction_count"].fillna(0).astype("int32")
    frame["item_mean_rating"] = frame["item_mean_rating"].fillna(0.0).astype("float32")
    frame["retrieval_score"] = frame["score"].astype("float32")
    frame["retrieval_rank"] = frame["candidate_rank"].astype("int16")
    frame["inverse_retrieval_rank"] = (1.0 / frame["candidate_rank"]).astype("float32")
    frame["source_count"] = frame["sources"].str.split("|").str.len().astype("int8")

    for source in SOURCE_NAMES:
        frame[f"source_{source}"] = (
            frame["sources"]
            .str.split("|")
            .apply(lambda values, source=source: source in values)
            .astype("int8")
        )

    return frame


FEATURE_COLUMNS = (
    "retrieval_score",
    "retrieval_rank",
    "inverse_retrieval_rank",
    "source_count",
    "source_item_knn",
    "source_latent_svd",
    "source_genre_content",
    "source_popularity",
    "user_history_count",
    "user_mean_rating",
    "item_interaction_count",
    "item_mean_rating",
)
