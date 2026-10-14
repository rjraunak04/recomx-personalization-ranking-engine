"""Build Day-04 Top-100 hybrid candidates from genuine deep retrieval sources."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.models.hybrid_retrieval import HybridRetrievalConfig, reciprocal_rank_fusion

GT = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
KNN = Path("reports/hybrid/sources/item_knn_top100.parquet")
SVD = Path("reports/hybrid/sources/latent_svd_top100.parquet")
POPULARITY = Path("reports/hybrid/sources/popularity_top100.parquet")
GENRE = Path("reports/baselines/genre/recommendations_top20.parquet")
TRAIN = Path("data/processed/ratings_train.parquet")
OUTPUT = Path("reports/hybrid")
CANDIDATES = OUTPUT / "hybrid_candidates_top100.parquet"
SUMMARY = OUTPUT / "candidate_summary.json"


def _as_source(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame[["userId", "movieId", "recommendation_rank"]].copy()


def main() -> None:
    required = [GT, KNN, SVD, POPULARITY, GENRE, TRAIN]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite artifacts: " + ", ".join(missing)
            + ". Run scripts/generate_day04_deep_candidates.py first."
        )

    users = set(pd.read_parquet(GT, columns=["userId"])["userId"].astype(int))
    train = pd.read_parquet(TRAIN, columns=["userId", "movieId"])
    seen = {
        int(user): set(group["movieId"].astype(int))
        for user, group in train.loc[train["userId"].isin(users)].groupby("userId", sort=False)
    }

    genre = pd.read_parquet(GENRE)
    # Day-02 content output contains popularity fallback rows for users without
    # a genre profile. Exclude those rows so popularity is not counted twice
    # under a misleading content source label.
    if "strategy" in genre.columns:
        genre = genre.loc[genre["strategy"].eq("genre_personalized")]

    sources = {
        "item_knn": _as_source(KNN),
        "latent_svd": _as_source(SVD),
        "genre_content": genre[["userId", "movieId", "recommendation_rank"]].copy(),
        "popularity": _as_source(POPULARITY),
    }
    candidates = reciprocal_rank_fusion(sources, seen, HybridRetrievalConfig(100, 60))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(CANDIDATES, index=False)

    counts = candidates.groupby("userId").size()
    summary = {
        "users": len(users),
        "candidate_rows": len(candidates),
        "users_with_candidates": int(candidates["userId"].nunique()),
        "users_with_100_candidates": int(counts.eq(100).sum()),
        "minimum_candidates_per_user": int(counts.min()),
        "mean_candidates_per_user": float(counts.mean()),
        "unique_items": int(candidates["movieId"].nunique()),
        "seen_item_violations": int(
            sum(
                len(set(group["movieId"].astype(int)) & seen.get(int(user), set()))
                for user, group in candidates.groupby("userId", sort=False)
            )
        ),
        "locked_test_used": False,
    }
    if summary["users_with_candidates"] != len(users):
        raise ValueError("Every evaluable user must receive candidates.")
    if summary["minimum_candidates_per_user"] != 100:
        raise ValueError("Day-04 retrieval contract requires 100 candidates per user.")
    if summary["seen_item_violations"]:
        raise ValueError("Train-seen item leakage detected.")

    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-04 HYBRID CANDIDATE GENERATION: PASS")


if __name__ == "__main__":
    main()
