"""Verify deterministic RecomX Day-01 temporal data preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recomx.data.reproducibility import (
    build_reproducibility_manifest,
    save_manifest,
)
from recomx.data.split import (
    TemporalSplitConfig,
    assert_temporal_integrity,
    temporal_split,
)

INPUT_PATH = Path("data/interim/ratings.parquet")

REPORT_PATH = Path("reports/reproducibility/manifest.json")


def main() -> None:
    """Run deterministic split twice and compare fingerprints."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing ratings dataset: {INPUT_PATH}")

    print("Loading MovieLens ratings...")

    ratings = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
    )

    print(f"Loaded {len(ratings):,} interactions.")

    config = TemporalSplitConfig(
        train_fraction=0.80,
        validation_fraction=0.10,
        test_fraction=0.10,
    )

    print("Running deterministic split A...")

    result_a = temporal_split(
        ratings,
        config,
    )

    assert_temporal_integrity(result_a)

    manifest_a = build_reproducibility_manifest(result_a)

    print("Running deterministic split B...")

    result_b = temporal_split(
        ratings,
        config,
    )

    assert_temporal_integrity(result_b)

    manifest_b = build_reproducibility_manifest(result_b)

    if manifest_a.combined_sha256 != manifest_b.combined_sha256:
        raise RuntimeError(
            "Reproducibility failure: identical input produced different partition fingerprints."
        )

    save_manifest(
        manifest_a,
        REPORT_PATH,
    )

    print()
    print("RecomX CP-07 Reproducibility Audit")
    print("=" * 72)

    for partition in (
        manifest_a.train,
        manifest_a.validation,
        manifest_a.locked_test,
    ):
        print(
            f"{partition.name:<14} "
            f"rows={partition.rows:,} "
            f"users={partition.unique_users:,} "
            f"items={partition.unique_items:,}"
        )

        print(f"{'':14} sha256={partition.sha256}")

    print()
    print(f"combined_sha256={manifest_a.combined_sha256}")

    print("repeat_run_match=True")

    print(f"manifest={REPORT_PATH}")

    print()
    print("CP-07 REPRODUCIBILITY PASSED")


if __name__ == "__main__":
    main()
