"""Generate Day-03 CP-05 Item-KNN and latent-SVD validation candidates."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse

from recomx.models.cf_candidates import (
    generate_item_knn_candidates,
    generate_svd_candidates,
)
from recomx.models.item_knn import ItemNeighborhoods
from recomx.models.latent_svd import LatentSVDModel
from recomx.models.popularity import PopularityModel, build_seen_item_map
from recomx.models.sparse_interactions import SparseInteractionMatrix

TRAIN_PATH = Path("data/processed/ratings_train.parquet")
COHORT_PATH = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")
SPARSE_DIR = Path("reports/baselines/collaborative/sparse")
KNN_DIR = Path("reports/baselines/collaborative/item_knn")
SVD_DIR = Path("reports/baselines/collaborative/latent_svd")
OUTPUT_DIR = Path("reports/baselines/collaborative/candidates")


def _load_interactions() -> SparseInteractionMatrix:
    matrix = sparse.load_npz(SPARSE_DIR / "positive_user_item_csr.npz").tocsr()
    user_ids = np.load(SPARSE_DIR / "user_ids.npy", allow_pickle=False)
    item_ids = np.load(SPARSE_DIR / "item_ids.npy", allow_pickle=False)
    return SparseInteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index={int(v): i for i, v in enumerate(user_ids)},
        item_to_index={int(v): i for i, v in enumerate(item_ids)},
    )


def main() -> None:
    required = [
        TRAIN_PATH,
        COHORT_PATH,
        SPARSE_DIR / "positive_user_item_csr.npz",
        KNN_DIR / "item_knn_cosine_csr.npz",
        SVD_DIR / "user_factors.npy",
        SVD_DIR / "item_factors.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifacts: " + ", ".join(missing))

    print("RecomX Day-03 CP-05 Collaborative Candidate Generation")
    print("=" * 72)
    print("validation_ratings=NOT READ")
    print("locked_test=NOT READ")
    print("existing CP-03 Item-KNN artifact=REUSED (NO REBUILD)")

    train = pd.read_parquet(TRAIN_PATH, columns=["userId", "movieId", "rating"])
    cohorts = pd.read_parquet(COHORT_PATH)
    users = cohorts.loc[cohorts["is_evaluable"], "userId"].astype(int).tolist()
    interactions = _load_interactions()
    seen = build_seen_item_map(train)
    popularity = PopularityModel.fit(train)

    knn = ItemNeighborhoods(
        matrix=sparse.load_npz(KNN_DIR / "item_knn_cosine_csr.npz").tocsr(),
        item_ids=interactions.item_ids.copy(),
    )
    svd = LatentSVDModel(
        user_factors=np.load(SVD_DIR / "user_factors.npy", allow_pickle=False),
        item_factors=np.load(SVD_DIR / "item_factors.npy", allow_pickle=False),
        user_ids=interactions.user_ids.copy(),
        item_ids=interactions.item_ids.copy(),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"evaluable_users": len(users), "top_k": 20, "locked_test_used": False}

    started = perf_counter()
    knn_recs, knn_summary = generate_item_knn_candidates(
        users, interactions, knn, popularity, seen, top_k=20
    )
    summary["item_knn_seconds"] = perf_counter() - started
    summary["item_knn"] = knn_summary.__dict__
    knn_recs.to_parquet(OUTPUT_DIR / "item_knn_top20.parquet", index=False)

    started = perf_counter()
    svd_recs, svd_summary = generate_svd_candidates(
        users, interactions, svd, popularity, seen, top_k=20, batch_size=4096
    )
    summary["latent_svd_seconds"] = perf_counter() - started
    summary["latent_svd"] = svd_summary.__dict__
    svd_recs.to_parquet(OUTPUT_DIR / "latent_svd_top20.parquet", index=False)

    with (OUTPUT_DIR / "candidate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={OUTPUT_DIR}")
    print("DAY-03 CP-05 CANDIDATE GENERATION: PASS")


if __name__ == "__main__":
    main()
