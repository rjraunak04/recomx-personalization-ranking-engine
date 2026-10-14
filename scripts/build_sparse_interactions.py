"""Build and audit the Day-03 train-only sparse interaction representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recomx.evaluation.relevance import RelevancePolicy
from recomx.models.collaborative_protocol import DEFAULT_COLLABORATIVE_PROTOCOL
from recomx.models.sparse_interactions import build_sparse_interaction_matrix

TRAIN_PATH = Path("data/processed/ratings_train.parquet")
OUTPUT_DIR = Path("reports/baselines/collaborative/sparse")
MATRIX_PATH = OUTPUT_DIR / "positive_user_item_csr.npz"
USERS_PATH = OUTPUT_DIR / "user_ids.npy"
ITEMS_PATH = OUTPUT_DIR / "item_ids.npy"
SUMMARY_PATH = OUTPUT_DIR / "sparse_matrix_summary.json"


def main() -> None:
    """Create the train-only positive-feedback CSR representation."""

    protocol = DEFAULT_COLLABORATIVE_PROTOCOL
    protocol.validate()

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing train partition: {TRAIN_PATH}")

    print("RecomX Day-03 CP-02 Sparse Interaction Matrix")
    print("=" * 72)
    print(f"train={TRAIN_PATH}")
    print("validation=NOT READ")
    print("locked_test=NOT READ")
    print(f"relevance_threshold={protocol.relevance_threshold:.1f}")
    print(f"matrix_format={protocol.matrix_format}")
    print(f"dtype={protocol.matrix_dtype}")
    print()

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=["userId", "movieId", "rating"],
    )

    representation = build_sparse_interaction_matrix(
        train,
        policy=RelevancePolicy(minimum_rating=protocol.relevance_threshold),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(MATRIX_PATH, representation.matrix, compressed=True)
    np.save(USERS_PATH, representation.user_ids, allow_pickle=False)
    np.save(ITEMS_PATH, representation.item_ids, allow_pickle=False)

    possible_pairs = representation.n_users * representation.n_items
    sparse_bytes = (
        representation.matrix.data.nbytes
        + representation.matrix.indices.nbytes
        + representation.matrix.indptr.nbytes
    )
    dense_float32_bytes = possible_pairs * np.dtype(np.float32).itemsize

    summary = {
        "training_rows": int(len(train)),
        "positive_users": representation.n_users,
        "positive_items": representation.n_items,
        "positive_user_item_pairs": representation.n_interactions,
        "possible_positive_matrix_cells": int(possible_pairs),
        "density": representation.density,
        "sparsity": 1.0 - representation.density,
        "matrix_format": "csr",
        "dtype": str(representation.matrix.dtype),
        "sparse_storage_bytes": int(sparse_bytes),
        "dense_float32_bytes_avoided": int(dense_float32_bytes),
        "locked_test_used": False,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for key, value in summary.items():
        print(f"{key}={value}")

    print()
    print(f"matrix={MATRIX_PATH}")
    print(f"users={USERS_PATH}")
    print(f"items={ITEMS_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print()
    print("DAY-03 CP-02 SPARSE INTERACTION MATRIX: PASS")


if __name__ == "__main__":
    main()
