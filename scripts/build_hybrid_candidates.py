"""Build Day-04 Top-100 hybrid candidates from reusable retrieval artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.models.hybrid_retrieval import HybridRetrievalConfig, reciprocal_rank_fusion

GT = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
KNN = Path("reports/baselines/collaborative/candidates/item_knn_top20.parquet")
SVD = Path("reports/baselines/collaborative/candidates/latent_svd_top20.parquet")
POPULARITY = Path("reports/baselines/popularity/recommendations_top20.parquet")
GENRE = Path("reports/baselines/genre/recommendations_top20.parquet")
TRAIN = Path("data/processed/ratings_train.parquet")
OUTPUT = Path("reports/hybrid")
CANDIDATES = OUTPUT / "hybrid_candidates_top100.parquet"
SUMMARY = OUTPUT / "candidate_summary.json"


def _as_source(path: Path, rank_name: str = "recommendation_rank") -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame[["userId", "movieId", rank_name]].rename(
        columns={rank_name: "recommendation_rank"}
    )


def main() -> None:
    required = [GT, KNN, SVD, POPULARITY, GENRE, TRAIN]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifacts: " + ", ".join(missing))

    users = set(pd.read_parquet(GT, columns=["userId"])["userId"].astype(int))
    train = pd.read_parquet(TRAIN, columns=["userId", "movieId"])
    seen = {
        int(user): set(group["movieId"].astype(int))
        for user, group in train.loc[train["userId"].isin(users)].groupby("userId", sort=False)
    }
    sources = {
        "item_knn": _as_source(KNN),
        "latent_svd": _as_source(SVD),
        "genre_content": _as_source(GENRE),
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
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-04 HYBRID CANDIDATE GENERATION: PASS")


if __name__ == "__main__":
    main()
