"""Raw-to-interim MovieLens ingestion."""

import gc
from pathlib import Path

from recomx.data.load import (
    load_genome_scores,
    load_genome_tags,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
)

JOBS = (
    ("ratings.csv", "ratings.parquet", load_ratings),
    ("movies.csv", "movies.parquet", load_movies),
    ("tags.csv", "tags.parquet", load_tags),
    ("genome-scores.csv", "genome_scores.parquet", load_genome_scores),
    ("genome-tags.csv", "genome_tags.parquet", load_genome_tags),
    ("links.csv", "links.parquet", load_links),
)


def ingest_movielens(
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
) -> None:
    """Convert MovieLens CSV tables into typed Parquet tables."""

    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    for source_name, output_name, loader in JOBS:
        print(f"Loading {source_name}...")

        frame = loader(raw_dir / source_name)

        print(
            f"  rows={len(frame):,}, "
            f"columns={len(frame.columns)}"
        )

        destination = interim_dir / output_name
        print(f"Writing {destination}...")

        frame.to_parquet(
            destination,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        del frame
        gc.collect()
