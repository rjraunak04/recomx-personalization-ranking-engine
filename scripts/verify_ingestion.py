"""Verify RecomX CP-03 MovieLens ingestion."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

FILES = (
    "ratings.parquet",
    "movies.parquet",
    "tags.parquet",
    "genome_scores.parquet",
    "genome_tags.parquet",
    "links.parquet",
)


def main() -> None:
    root = Path("data/interim")

    print("RecomX CP-03 verification")
    print("=" * 45)

    for filename in FILES:
        path = root / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows

        if rows <= 0:
            raise ValueError(f"{filename} is empty")

        print(f"{filename:<24} rows={rows:,}")

    ratings = pd.read_parquet(
        root / "ratings.parquet",
        columns=["userId", "movieId", "rating", "timestamp"],
    )

    if not pd.api.types.is_datetime64_any_dtype(
        ratings["timestamp"]
    ):
        raise TypeError("ratings timestamp is not datetime")

    print()
    print("Ratings summary")
    print("rows:", f"{len(ratings):,}")
    print("users:", f"{ratings['userId'].nunique():,}")
    print("movies:", f"{ratings['movieId'].nunique():,}")
    print(
        "rating range:",
        ratings["rating"].min(),
        "to",
        ratings["rating"].max(),
    )
    print(
        "date range:",
        ratings["timestamp"].min(),
        "to",
        ratings["timestamp"].max(),
    )

    print()
    print("CP-03 PASSED")


if __name__ == "__main__":
    main()
