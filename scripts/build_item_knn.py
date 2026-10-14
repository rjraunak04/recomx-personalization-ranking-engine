"""Build and audit the Day-03 CP-03 sparse Item-KNN graph."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from recomx.models.collaborative_protocol import DEFAULT_COLLABORATIVE_PROTOCOL
from recomx.models.item_knn import ItemKNNConfig, build_item_neighborhoods
from recomx.models.sparse_interactions import SparseInteractionMatrix

INPUT_DIR = Path("reports/baselines/collaborative/sparse")
OUTPUT_DIR = Path("reports/baselines/collaborative/item_knn")
MATRIX_PATH = INPUT_DIR / "positive_user_item_csr.npz"
USERS_PATH = INPUT_DIR / "user_ids.npy"
ITEMS_PATH = INPUT_DIR / "item_ids.npy"
GRAPH_PATH = OUTPUT_DIR / "item_knn_cosine_csr.npz"
SUMMARY_PATH = OUTPUT_DIR / "item_knn_summary.json"


def main() -> None:
    protocol = DEFAULT_COLLABORATIVE_PROTOCOL
    protocol.validate()

    required = (MATRIX_PATH, USERS_PATH, ITEMS_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run CP-02 first; missing sparse artifacts: " + ", ".join(missing)
        )

    matrix = sparse.load_npz(MATRIX_PATH).tocsr().astype(np.float32)
    user_ids = np.load(USERS_PATH, allow_pickle=False)
    item_ids = np.load(ITEMS_PATH, allow_pickle=False)
    interactions = SparseInteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index={int(value): index for index, value in enumerate(user_ids)},
        item_to_index={int(value): index for index, value in enumerate(item_ids)},
    )

    config = ItemKNNConfig(
        neighbors=protocol.item_knn_neighbors,
        min_similarity=0.0,
    )

    print("RecomX Day-03 CP-03 Item-KNN")
    print("=" * 72)
    print(f"items={interactions.n_items}")
    print(f"positive_pairs={interactions.n_interactions}")
    print(f"neighbors_per_item<={config.neighbors}")
    print("similarity=cosine")
    print("dense_item_similarity=FORBIDDEN")
    print("validation=NOT READ")
    print("locked_test=NOT READ")

    started = perf_counter()
    model = build_item_neighborhoods(interactions, config)
    elapsed = perf_counter() - started

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(GRAPH_PATH, model.matrix, compressed=True)

    row_counts = np.diff(model.matrix.indptr)
    summary = {
        "items": model.n_items,
        "edges": model.n_edges,
        "max_neighbors": int(row_counts.max(initial=0)),
        "mean_neighbors": float(row_counts.mean()),
        "neighbors_limit": config.neighbors,
        "similarity": "cosine",
        "graph_format": "csr",
        "dtype": str(model.matrix.dtype),
        "self_edges": int(np.count_nonzero(model.matrix.diagonal())),
        "build_seconds": elapsed,
        "validation_used": False,
        "locked_test_used": False,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"graph={GRAPH_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print("DAY-03 CP-03 ITEM-KNN: PASS")


if __name__ == "__main__":
    main()
