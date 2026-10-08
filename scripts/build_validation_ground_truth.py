"""Build Day-02 validation ground truth for RecomX."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.ground_truth import (
    build_validation_ground_truth,
)
from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL
from recomx.evaluation.relevance import RelevancePolicy

TRAIN_PATH = Path("data/processed/ratings_train.parquet")

VALIDATION_PATH = Path("data/processed/ratings_validation.parquet")

OUTPUT_DIR = Path("reports/baselines/ground_truth")

GROUND_TRUTH_PATH = OUTPUT_DIR / "validation_ground_truth.parquet"

COHORT_PATH = OUTPUT_DIR / "validation_user_cohorts.parquet"

SUMMARY_PATH = OUTPUT_DIR / "validation_ground_truth_summary.json"


def main() -> None:
    """Create leakage-safe validation targets."""

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing train partition: {TRAIN_PATH}")

    if not VALIDATION_PATH.is_file():
        raise FileNotFoundError(f"Missing validation partition: {VALIDATION_PATH}")

    print("RecomX CP-03 Validation Ground Truth")
    print("=" * 72)
    print(f"train={TRAIN_PATH}")
    print(f"validation={VALIDATION_PATH}")
    print("locked_test=NOT READ")
    print(f"relevance_threshold={protocol.relevance_threshold:.1f}")
    print()

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "userId",
            "movieId",
            "rating",
        ],
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "userId",
            "movieId",
            "rating",
        ],
    )

    result = build_validation_ground_truth(
        train=train,
        validation=validation,
        policy=RelevancePolicy(minimum_rating=(protocol.relevance_threshold)),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.ground_truth.to_parquet(
        GROUND_TRUTH_PATH,
        index=False,
    )

    result.cohorts.to_parquet(
        COHORT_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result.summary.as_dict(),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    for key, value in result.summary.as_dict().items():
        print(f"{key}={value:,}")

    print()
    print(f"ground_truth={GROUND_TRUTH_PATH}")
    print(f"cohorts={COHORT_PATH}")
    print(f"summary={SUMMARY_PATH}")
    print()
    print("CP-03 VALIDATION GROUND TRUTH: PASS")


if __name__ == "__main__":
    main()
