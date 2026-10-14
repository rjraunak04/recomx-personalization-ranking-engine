"""Reproducibility utilities for the RecomX data foundation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recomx.data.split import TemporalSplitResult


@dataclass(frozen=True)
class PartitionFingerprint:
    """Deterministic fingerprint for one temporal partition."""

    name: str
    rows: int
    unique_users: int
    unique_items: int
    start_timestamp: str
    end_timestamp: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable fingerprint."""

        return asdict(self)


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Audit manifest for deterministic temporal partitions."""

    algorithm: str
    source_rows: int
    train: PartitionFingerprint
    validation: PartitionFingerprint
    locked_test: PartitionFingerprint
    combined_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable manifest."""

        return asdict(self)


def dataframe_sha256(
    frame: pd.DataFrame,
) -> str:
    """Create deterministic SHA-256 from recommendation interactions.

    The hash is based on canonical row order and stable CSV
    serialization rather than Parquet file bytes. This avoids treating
    storage-level metadata as part of the logical dataset identity.
    """

    required = [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]

    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError("Missing fingerprint columns: " + ", ".join(missing))

    canonical = (
        frame[required]
        .sort_values(
            [
                "timestamp",
                "userId",
                "movieId",
                "rating",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    hasher = hashlib.sha256()

    chunk_size = 250_000

    for start in range(
        0,
        len(canonical),
        chunk_size,
    ):
        chunk = canonical.iloc[start : start + chunk_size].copy()

        chunk["timestamp"] = chunk["timestamp"].astype("int64")

        payload = chunk.to_csv(
            index=False,
            header=(start == 0),
            lineterminator="\n",
            float_format="%.6f",
        ).encode("utf-8")

        hasher.update(payload)

    return hasher.hexdigest()


def partition_fingerprint(
    name: str,
    frame: pd.DataFrame,
) -> PartitionFingerprint:
    """Build deterministic audit metadata for one partition."""

    if frame.empty:
        raise ValueError(f"Cannot fingerprint empty partition: {name}")

    return PartitionFingerprint(
        name=name,
        rows=len(frame),
        unique_users=int(frame["userId"].nunique()),
        unique_items=int(frame["movieId"].nunique()),
        start_timestamp=str(frame["timestamp"].min()),
        end_timestamp=str(frame["timestamp"].max()),
        sha256=dataframe_sha256(frame),
    )


def build_reproducibility_manifest(
    result: TemporalSplitResult,
) -> ReproducibilityManifest:
    """Build a deterministic manifest for a temporal split."""

    train = partition_fingerprint(
        "train",
        result.train,
    )

    validation = partition_fingerprint(
        "validation",
        result.validation,
    )

    locked_test = partition_fingerprint(
        "locked_test",
        result.test,
    )

    combined_payload = json.dumps(
        {
            "train": train.sha256,
            "validation": validation.sha256,
            "locked_test": locked_test.sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    combined_sha256 = hashlib.sha256(combined_payload).hexdigest()

    return ReproducibilityManifest(
        algorithm="sha256-canonical-interactions-v1",
        source_rows=result.summary.total_rows,
        train=train,
        validation=validation,
        locked_test=locked_test,
        combined_sha256=combined_sha256,
    )


def save_manifest(
    manifest: ReproducibilityManifest,
    path: str | Path,
) -> Path:
    """Persist a reproducibility manifest."""

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            manifest.to_dict(),
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return destination
