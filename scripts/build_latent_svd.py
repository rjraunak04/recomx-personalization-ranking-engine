"""Fit and audit the Day-03 CP-04 train-only latent SVD model."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from recomx.models.collaborative_protocol import DEFAULT_COLLABORATIVE_PROTOCOL
from recomx.models.latent_svd import LatentSVDConfig, fit_latent_svd
from recomx.models.sparse_interactions import SparseInteractionMatrix

INPUT_DIR = Path("reports/baselines/collaborative/sparse")
OUTPUT_DIR = Path("reports/baselines/collaborative/latent_svd")
MATRIX_PATH = INPUT_DIR / "positive_user_item_csr.npz"
USERS_PATH = INPUT_DIR / "user_ids.npy"
ITEMS_PATH = INPUT_DIR / "item_ids.npy"
USER_FACTORS_PATH = OUTPUT_DIR / "user_factors.npy"
ITEM_FACTORS_PATH = OUTPUT_DIR / "item_factors.npy"
SUMMARY_PATH = OUTPUT_DIR / "latent_svd_summary.json"


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

    config = LatentSVDConfig(
        factors=protocol.latent_factors,
        random_state=protocol.random_state,
    )

    print("RecomX Day-03 CP-04 Latent SVD")
    print("=" * 72)
    print(f"users={interactions.n_users}")
    print(f"items={interactions.n_items}")
    print(f"positive_pairs={interactions.n_interactions}")
    print(f"latent_factors={config.factors}")
    print(f"random_state={config.random_state}")
    print("algorithm=scipy.sparse.linalg.svds")
    print("dense_user_item_reconstruction=FORBIDDEN")
    print("validation=NOT READ")
    print("locked_test=NOT READ")

    started = perf_counter()
    model = fit_latent_svd(interactions, config)
    elapsed = perf_counter() - started

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(USER_FACTORS_PATH, model.user_factors, allow_pickle=False)
    np.save(ITEM_FACTORS_PATH, model.item_factors, allow_pickle=False)

    factor_bytes = model.user_factors.nbytes + model.item_factors.nbytes
    summary = {
        "users": interactions.n_users,
        "items": interactions.n_items,
        "positive_pairs": interactions.n_interactions,
        "latent_factors": model.n_factors,
        "user_factor_shape": list(model.user_factors.shape),
        "item_factor_shape": list(model.item_factors.shape),
        "dtype": str(model.user_factors.dtype),
        "factor_storage_bytes": int(factor_bytes),
        "fit_seconds": elapsed,
        "random_state": config.random_state,
        "validation_used": False,
        "locked_test_used": False,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"user_factors={USER_FACTORS_PATH}")
    print(f"item_factors={ITEM_FACTORS_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print("DAY-03 CP-04 LATENT SVD: PASS")


if __name__ == "__main__":
    main()
