"""Generate genuine deep Day-04 retrieval sources without rebuilding fitted models."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse

from recomx.models.item_knn import ItemNeighborhoods, recommend_item_knn
from recomx.models.latent_svd import LatentSVDModel
from recomx.models.popularity import PopularityModel, build_seen_item_map
from recomx.models.sparse_interactions import SparseInteractionMatrix

TRAIN = Path("data/processed/ratings_train.parquet")
COHORTS = Path("reports/baselines/ground_truth/validation_user_cohorts.parquet")
SPARSE = Path("reports/baselines/collaborative/sparse")
KNN = Path("reports/baselines/collaborative/item_knn/item_knn_cosine_csr.npz")
SVD = Path("reports/baselines/collaborative/latent_svd")
OUTPUT = Path("reports/hybrid/sources")
TOP_K = 100


def _interactions() -> SparseInteractionMatrix:
    matrix = sparse.load_npz(SPARSE / "positive_user_item_csr.npz").tocsr()
    user_ids = np.load(SPARSE / "user_ids.npy", allow_pickle=False)
    item_ids = np.load(SPARSE / "item_ids.npy", allow_pickle=False)
    return SparseInteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index={int(v): i for i, v in enumerate(user_ids)},
        item_to_index={int(v): i for i, v in enumerate(item_ids)},
    )


def _frame(user_id: int, item_ids: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": int(user_id),
            "movieId": item_ids.astype(np.int64, copy=False),
            "score": scores.astype(float, copy=False),
            "recommendation_rank": np.arange(1, len(item_ids) + 1),
        }
    )


def main() -> None:
    required = [
        TRAIN,
        COHORTS,
        SPARSE / "positive_user_item_csr.npz",
        SPARSE / "user_ids.npy",
        SPARSE / "item_ids.npy",
        KNN,
        SVD / "user_factors.npy",
        SVD / "item_factors.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifacts: " + ", ".join(missing))

    train = pd.read_parquet(TRAIN, columns=["userId", "movieId", "rating"])
    cohorts = pd.read_parquet(COHORTS)
    users = cohorts.loc[cohorts["is_evaluable"], "userId"].astype(int).tolist()
    interactions = _interactions()
    seen = build_seen_item_map(train)
    popularity = PopularityModel.fit(train)
    neighborhoods = ItemNeighborhoods(
        matrix=sparse.load_npz(KNN).tocsr(),
        item_ids=interactions.item_ids.copy(),
    )
    svd = LatentSVDModel(
        user_factors=np.load(SVD / "user_factors.npy", allow_pickle=False),
        item_factors=np.load(SVD / "item_factors.npy", allow_pickle=False),
        user_ids=interactions.user_ids.copy(),
        item_ids=interactions.item_ids.copy(),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    knn_rows: list[pd.DataFrame] = []
    svd_rows: list[pd.DataFrame] = []
    pop_rows: list[pd.DataFrame] = []
    warm = 0
    started = perf_counter()

    for user_id in users:
        user_seen = seen.get(user_id, set())
        pop = popularity.recommend(user_id=user_id, seen_items=user_seen, k=TOP_K)
        pop_rows.append(
            pop[["userId", "movieId", "popularity_score", "recommendation_rank"]].rename(
                columns={"popularity_score": "score"}
            )
        )

        if interactions.user_index(user_id) is None:
            continue
        warm += 1

        item_ids, scores = recommend_item_knn(
            user_id, interactions, neighborhoods, top_k=TOP_K + len(user_seen)
        )
        knn_frame = _frame(user_id, item_ids, scores)
        knn_frame = knn_frame.loc[~knn_frame["movieId"].isin(user_seen)].head(TOP_K).copy()
        knn_frame["recommendation_rank"] = np.arange(1, len(knn_frame) + 1)
        if not knn_frame.empty:
            knn_rows.append(knn_frame)

        item_ids, scores = svd.recommend(
            user_id, interactions, top_k=TOP_K + len(user_seen), batch_size=4096
        )
        svd_frame = _frame(user_id, item_ids, scores)
        svd_frame = svd_frame.loc[~svd_frame["movieId"].isin(user_seen)].head(TOP_K).copy()
        svd_frame["recommendation_rank"] = np.arange(1, len(svd_frame) + 1)
        if not svd_frame.empty:
            svd_rows.append(svd_frame)

    outputs = {
        "item_knn": pd.concat(knn_rows, ignore_index=True),
        "latent_svd": pd.concat(svd_rows, ignore_index=True),
        "popularity": pd.concat(pop_rows, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_parquet(OUTPUT / f"{name}_top100.parquet", index=False)

    summary = {
        "users": len(users),
        "positive_matrix_warm_users": warm,
        "top_k": TOP_K,
        "rows": {name: len(frame) for name, frame in outputs.items()},
        "users_by_source": {
            name: int(frame["userId"].nunique()) for name, frame in outputs.items()
        },
        "seconds": perf_counter() - started,
        "item_knn_graph_rebuilt": False,
        "svd_refit": False,
        "locked_test_used": False,
    }
    with (OUTPUT / "deep_source_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-04 DEEP SOURCE GENERATION: PASS")


if __name__ == "__main__":
    main()
