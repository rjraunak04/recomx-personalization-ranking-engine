"""Hybrid multi-source candidate retrieval for RecomX."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HybridRetrievalConfig:
    candidate_budget: int = 100
    rrf_k: int = 60

    def validate(self) -> None:
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive.")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")


def build_genre_profiles(train: pd.DataFrame, movies: pd.DataFrame, threshold: float = 4.0):
    """Build train-only normalized user genre profiles and movie genre sets."""
    positive = train.loc[train["rating"].ge(threshold), ["userId", "movieId"]]
    movie_genres = movies[["movieId", "genres"]].copy()
    movie_genres["genre"] = movie_genres["genres"].fillna("").str.split("|")
    exploded = movie_genres.explode("genre")
    exploded = exploded.loc[
        exploded["genre"].ne("") & exploded["genre"].ne("(no genres listed)")
    ]
    history = positive.merge(exploded[["movieId", "genre"]], on="movieId", how="inner")
    counts = history.groupby(["userId", "genre"]).size().rename("count").reset_index()
    totals = counts.groupby("userId")["count"].transform("sum")
    counts["affinity"] = counts["count"] / totals
    item_genres = exploded.groupby("movieId")["genre"].agg(frozenset).to_dict()
    return counts[["userId", "genre", "affinity"]], item_genres


def reciprocal_rank_fusion(
    source_frames: dict[str, pd.DataFrame],
    seen_items: dict[int, set[int]],
    config: HybridRetrievalConfig | None = None,
) -> pd.DataFrame:
    """Fuse ranked candidate sources deterministically with reciprocal-rank fusion."""
    active = config or HybridRetrievalConfig()
    active.validate()
    rows = []
    for source, frame in source_frames.items():
        required = {"userId", "movieId", "recommendation_rank"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} missing columns: {sorted(missing)}")
        part = frame[["userId", "movieId", "recommendation_rank"]].copy()
        part = part.loc[
            ~part.apply(
                lambda row: int(row.movieId) in seen_items.get(int(row.userId), set()),
                axis=1,
            )
        ]
        part["source"] = source
        part["rrf"] = 1.0 / (active.rrf_k + part["recommendation_rank"].astype(float))
        rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=["userId", "movieId", "score", "candidate_rank", "sources"]
        )
    merged = pd.concat(rows, ignore_index=True)
    grouped = (
        merged.groupby(["userId", "movieId"], as_index=False)
        .agg(score=("rrf", "sum"), sources=("source", lambda x: "|".join(sorted(set(x)))))
    )
    grouped = grouped.sort_values(
        ["userId", "score", "movieId"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    grouped["candidate_rank"] = grouped.groupby("userId").cumcount() + 1
    grouped = grouped.loc[grouped["candidate_rank"].le(active.candidate_budget)]
    return grouped[["userId", "movieId", "score", "candidate_rank", "sources"]].reset_index(
        drop=True
    )


def candidate_recall(
    candidates: pd.DataFrame, ground_truth: pd.DataFrame, cutoff: int = 100
) -> float:
    """Macro-average candidate Recall@K."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    truth = {
        int(user): set(group["movieId"].astype(int))
        for user, group in ground_truth.groupby("userId", sort=False)
    }
    ranked = candidates.loc[candidates["candidate_rank"].le(cutoff)]
    recs = {
        int(user): set(group["movieId"].astype(int))
        for user, group in ranked.groupby("userId", sort=False)
    }
    recalls = [len(recs.get(user, set()) & items) / len(items) for user, items in truth.items()]
    return float(sum(recalls) / len(recalls)) if recalls else 0.0
