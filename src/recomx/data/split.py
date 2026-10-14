"""Leakage-safe temporal splitting for RecomX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Configuration for chronological recommendation splits."""

    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10

    def validate(self) -> None:
        """Validate configured split fractions."""

        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )

        if any(fraction <= 0.0 for fraction in fractions):
            raise ValueError("All temporal split fractions must be positive.")

        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Temporal split fractions must sum to 1.0.")


@dataclass(frozen=True)
class TemporalSplitSummary:
    """Audit information for a temporal recommendation split."""

    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_fraction_actual: float
    validation_fraction_actual: float
    test_fraction_actual: float
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_validation_cutoff: str
    validation_test_cutoff: str
    validation_cold_users: int
    validation_cold_items: int
    test_cold_users_vs_train: int
    test_cold_items_vs_train: int
    test_cold_users_vs_history: int
    test_cold_items_vs_history: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary."""

        return asdict(self)


@dataclass(frozen=True)
class TemporalSplitResult:
    """Dataframes and audit summary produced by temporal splitting."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: TemporalSplitSummary


def _timestamp_at_fraction(
    sorted_timestamps: pd.Series,
    fraction: float,
) -> pd.Timestamp:
    """Select timestamp boundary near a requested cumulative fraction."""

    if sorted_timestamps.empty:
        raise ValueError("Cannot determine temporal cutoff from empty data.")

    index = int(len(sorted_timestamps) * fraction)

    index = min(
        max(index, 1),
        len(sorted_timestamps) - 1,
    )

    return sorted_timestamps.iloc[index]


def _cold_entity_count(
    current: pd.DataFrame,
    history: pd.DataFrame,
    column: str,
) -> int:
    """Count unique current entities absent from historical data."""

    historical = set(history[column].unique())

    current_entities = set(current[column].unique())

    return len(current_entities - historical)


def temporal_split(
    ratings: pd.DataFrame,
    config: TemporalSplitConfig | None = None,
) -> TemporalSplitResult:
    """Split interactions chronologically without timestamp leakage.

    All rows sharing the same timestamp remain in the same partition.

    Partition semantics:

    train:
        timestamp < train/validation cutoff

    validation:
        train/validation cutoff <= timestamp
        < validation/test cutoff

    test:
        timestamp >= validation/test cutoff
    """

    if config is None:
        config = TemporalSplitConfig()

    config.validate()

    required = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    missing = sorted(required - set(ratings.columns))

    if missing:
        raise ValueError("Missing required split columns: " + ", ".join(missing))

    if ratings.empty:
        raise ValueError("Cannot split an empty ratings dataset.")

    if ratings["timestamp"].isna().any():
        raise ValueError("Temporal splitting requires non-null timestamps.")

    ordered = ratings.sort_values(
        [
            "timestamp",
            "userId",
            "movieId",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    timestamps = ordered["timestamp"]

    train_cutoff = _timestamp_at_fraction(
        timestamps,
        config.train_fraction,
    )

    validation_cutoff = _timestamp_at_fraction(
        timestamps,
        (config.train_fraction + config.validation_fraction),
    )

    if train_cutoff >= validation_cutoff:
        raise ValueError(
            "Temporal cutoffs are not strictly ordered. "
            "Dataset does not provide enough timestamp variation."
        )

    train = ordered.loc[ordered["timestamp"] < train_cutoff].copy()

    validation = ordered.loc[
        (ordered["timestamp"] >= train_cutoff) & (ordered["timestamp"] < validation_cutoff)
    ].copy()

    test = ordered.loc[ordered["timestamp"] >= validation_cutoff].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")

    history_before_test = pd.concat(
        [
            train[["userId", "movieId"]],
            validation[["userId", "movieId"]],
        ],
        ignore_index=True,
    )

    total_rows = len(ordered)

    summary = TemporalSplitSummary(
        total_rows=total_rows,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_fraction_actual=(len(train) / total_rows),
        validation_fraction_actual=(len(validation) / total_rows),
        test_fraction_actual=(len(test) / total_rows),
        train_start=str(train["timestamp"].min()),
        train_end=str(train["timestamp"].max()),
        validation_start=str(validation["timestamp"].min()),
        validation_end=str(validation["timestamp"].max()),
        test_start=str(test["timestamp"].min()),
        test_end=str(test["timestamp"].max()),
        train_validation_cutoff=str(train_cutoff),
        validation_test_cutoff=str(validation_cutoff),
        validation_cold_users=(
            _cold_entity_count(
                validation,
                train,
                "userId",
            )
        ),
        validation_cold_items=(
            _cold_entity_count(
                validation,
                train,
                "movieId",
            )
        ),
        test_cold_users_vs_train=(
            _cold_entity_count(
                test,
                train,
                "userId",
            )
        ),
        test_cold_items_vs_train=(
            _cold_entity_count(
                test,
                train,
                "movieId",
            )
        ),
        test_cold_users_vs_history=(
            _cold_entity_count(
                test,
                history_before_test,
                "userId",
            )
        ),
        test_cold_items_vs_history=(
            _cold_entity_count(
                test,
                history_before_test,
                "movieId",
            )
        ),
    )

    return TemporalSplitResult(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
    )


def assert_temporal_integrity(
    result: TemporalSplitResult,
) -> None:
    """Verify row preservation and strict chronological separation."""

    partitions = (
        result.train,
        result.validation,
        result.test,
    )

    if any(partition.empty for partition in partitions):
        raise ValueError("Temporal partitions must not be empty.")

    observed_rows = sum(len(partition) for partition in partitions)

    if observed_rows != result.summary.total_rows:
        raise ValueError("Temporal split does not preserve all input rows.")

    train_end = result.train["timestamp"].max()

    validation_start = result.validation["timestamp"].min()

    validation_end = result.validation["timestamp"].max()

    test_start = result.test["timestamp"].min()

    if not train_end < validation_start:
        raise ValueError("Train/validation temporal leakage detected.")

    if not validation_end < test_start:
        raise ValueError("Validation/test temporal leakage detected.")

    train_timestamps = set(result.train["timestamp"].unique())

    validation_timestamps = set(result.validation["timestamp"].unique())

    test_timestamps = set(result.test["timestamp"].unique())

    if train_timestamps & validation_timestamps:
        raise ValueError("Train and validation share timestamps.")

    if validation_timestamps & test_timestamps:
        raise ValueError("Validation and test share timestamps.")

    if train_timestamps & test_timestamps:
        raise ValueError("Train and test share timestamps.")


def save_temporal_splits(
    result: TemporalSplitResult,
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Persist temporal partitions as compressed Parquet."""

    root = Path(output_dir)

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train": root / "ratings_train.parquet",
        "validation": (root / "ratings_validation.parquet"),
        "test": (root / "ratings_locked_test.parquet"),
    }

    result.train.to_parquet(
        paths["train"],
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    result.validation.to_parquet(
        paths["validation"],
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    result.test.to_parquet(
        paths["test"],
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    return paths
