"""Candidate-retrieval evaluation utilities."""

from __future__ import annotations

import pandas as pd


def source_contribution(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique candidate rows carrying each retrieval source."""
    if "sources" not in candidates.columns:
        raise ValueError("candidates missing sources column.")
    exploded = candidates[["userId", "movieId", "sources"]].copy()
    exploded["source"] = exploded["sources"].str.split("|")
    exploded = exploded.explode("source")
    return (
        exploded.groupby("source", as_index=False)
        .agg(candidate_rows=("movieId", "size"), users=("userId", "nunique"))
        .sort_values(["candidate_rows", "source"], ascending=[False, True])
        .reset_index(drop=True)
    )


def overlap_rate(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Jaccard overlap between two user-item candidate sets."""
    a = set(zip(left["userId"].astype(int), left["movieId"].astype(int), strict=True))
    b = set(zip(right["userId"].astype(int), right["movieId"].astype(int), strict=True))
    union = a | b
    return len(a & b) / len(union) if union else 0.0
