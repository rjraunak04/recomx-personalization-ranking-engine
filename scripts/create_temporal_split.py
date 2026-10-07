"""Create leakage-safe temporal MovieLens partitions for RecomX."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.data.split import (
    TemporalSplitConfig,
    assert_temporal_integrity,
    save_temporal_splits,
    temporal_split,
)

INPUT_PATH = Path("data/interim/ratings.parquet")

OUTPUT_DIR = Path("data/processed")

REPORT_DIR = Path("reports/splits")


def main() -> None:
    """Create and verify CP-06 temporal partitions."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing CP-03 ratings file: {INPUT_PATH}")

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

    print("Creating chronological 80/10/10 split...")

    result = temporal_split(
        ratings,
        config,
    )

    assert_temporal_integrity(result)

    paths = save_temporal_splits(
        result,
        OUTPUT_DIR,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = REPORT_DIR / "temporal_split_summary.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result.summary.to_dict(),
            handle,
            indent=2,
        )

    print()
    print("RecomX CP-06 Temporal Split")
    print("=" * 70)

    for key, value in result.summary.to_dict().items():
        if isinstance(
            value,
            float,
        ):
            print(f"{key:<40} {value:.6f}")
        else:
            print(f"{key:<40} {value}")

    print()
    print("Generated partitions:")

    for name, path in paths.items():
        size_mb = path.stat().st_size / 1024**2

        print(f"  {name:<12} {path} ({size_mb:.2f} MB)")

    print()
    print("Temporal integrity: PASS")

    print("Locked test policy: DO NOT use test data for model selection or tuning.")

    print()
    print("CP-06 PASSED")


if __name__ == "__main__":
    main()
