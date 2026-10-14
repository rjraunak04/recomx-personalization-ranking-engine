"""Fast verification of the RecomX Day-01 data foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_SOURCE_ROWS = 25_000_095

EXPECTED_SPLIT_ROWS = {
    "train": 20_000_076,
    "validation": 2_500_009,
    "locked_test": 2_500_010,
}

EXPECTED_COMBINED_SHA256 = "5a57a4afe7b5ac54221d529ed4a8db433451aab45323dba1b84879d50cfe091c"

INTERIM_FILES = {
    "ratings": Path("data/interim/ratings.parquet"),
    "movies": Path("data/interim/movies.parquet"),
    "tags": Path("data/interim/tags.parquet"),
    "genome_scores": Path("data/interim/genome_scores.parquet"),
    "genome_tags": Path("data/interim/genome_tags.parquet"),
    "links": Path("data/interim/links.parquet"),
}

PROCESSED_FILES = {
    "train": Path("data/processed/ratings_train.parquet"),
    "validation": Path("data/processed/ratings_validation.parquet"),
    "locked_test": Path("data/processed/ratings_locked_test.parquet"),
}

MANIFEST_PATH = Path("reports/reproducibility/manifest.json")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required Day-01 artifact missing: {path}")


def parquet_rows(path: Path) -> int:
    require_file(path)
    return pq.ParquetFile(path).metadata.num_rows


def verify_interim() -> None:
    print("Checking interim datasets...")

    for name, path in INTERIM_FILES.items():
        rows = parquet_rows(path)
        if rows <= 0:
            raise RuntimeError(f"Interim table is empty: {name}")
        print(f"  PASS {name:<15} rows={rows:,}")

    ratings_rows = parquet_rows(INTERIM_FILES["ratings"])
    if ratings_rows != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"Unexpected ratings row count: {ratings_rows:,}")


def verify_processed() -> None:
    print("\nChecking temporal partitions...")

    total = 0

    for name, path in PROCESSED_FILES.items():
        rows = parquet_rows(path)
        expected = EXPECTED_SPLIT_ROWS[name]

        if rows != expected:
            raise RuntimeError(f"{name} row mismatch: expected={expected:,}, observed={rows:,}")

        total += rows
        print(f"  PASS {name:<15} rows={rows:,}")

    if total != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("Temporal partitions do not preserve source rows.")


def verify_manifest() -> None:
    print("\nChecking reproducibility manifest...")

    require_file(MANIFEST_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("source_rows") != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("Manifest source row count mismatch.")

    if manifest.get("combined_sha256") != EXPECTED_COMBINED_SHA256:
        raise RuntimeError("Reproducibility fingerprint mismatch.")

    for name, expected_rows in EXPECTED_SPLIT_ROWS.items():
        partition = manifest.get(name)

        if not isinstance(partition, dict):
            raise RuntimeError(f"Manifest partition missing: {name}")

        if partition.get("rows") != expected_rows:
            raise RuntimeError(f"Manifest row mismatch: {name}")

    print("  PASS source rows")
    print("  PASS partition rows")
    print("  PASS combined SHA-256")


def main() -> None:
    print("RecomX Day-01 Verification")
    print("=" * 72)

    verify_interim()
    verify_processed()
    verify_manifest()

    print("\n" + "=" * 72)
    print("DAY-01 DATA FOUNDATION: PASS")


if __name__ == "__main__":
    main()
