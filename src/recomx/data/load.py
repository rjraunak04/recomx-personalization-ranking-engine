"""Schema-aware MovieLens 25M loaders."""

from pathlib import Path

import pandas as pd


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"MovieLens file not found: {path}")


def load_ratings(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    frame = pd.read_csv(
        path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], unit="s", utc=True
    )
    return frame


def load_movies(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    return pd.read_csv(
        path,
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string",
        },
    )


def load_tags(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    frame = pd.read_csv(
        path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "tag": "string",
            "timestamp": "int64",
        },
    )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], unit="s", utc=True
    )
    return frame


def load_genome_scores(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    return pd.read_csv(
        path,
        dtype={
            "movieId": "int32",
            "tagId": "int16",
            "relevance": "float32",
        },
    )


def load_genome_tags(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    return pd.read_csv(
        path,
        dtype={"tagId": "int16", "tag": "string"},
    )


def load_links(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _require(path)
    return pd.read_csv(
        path,
        dtype={
            "movieId": "int32",
            "imdbId": "string",
            "tmdbId": "string",
        },
    )
