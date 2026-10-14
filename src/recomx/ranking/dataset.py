"""Leakage-safe ranking dataset construction for RecomX Day-05."""

from __future__ import annotations

import pandas as pd


def build_ranking_dataset(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Attach binary validation relevance labels to a frozen candidate pool."""
    required_candidates = {"userId", "movieId", "score", "candidate_rank", "sources"}
    required_truth = {"userId", "movieId"}
    missing_candidates = required_candidates.difference(candidates.columns)
    missing_truth = required_truth.difference(ground_truth.columns)
    if missing_candidates:
        raise ValueError(f"Candidates missing columns: {sorted(missing_candidates)}")
    if missing_truth:
        raise ValueError(f"Ground truth missing columns: {sorted(missing_truth)}")
    if candidates.duplicated(["userId", "movieId"]).any():
        raise ValueError("Candidate pool contains duplicate user-item pairs.")

    truth_pairs = ground_truth[["userId", "movieId"]].drop_duplicates().assign(label=1)
    frame = candidates.merge(truth_pairs, on=["userId", "movieId"], how="left")
    frame["label"] = frame["label"].fillna(0).astype("int8")
    return frame.sort_values(
        ["userId", "candidate_rank", "movieId"], kind="mergesort"
    ).reset_index(drop=True)


def ranking_dataset_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    """Return compact class-balance and coverage diagnostics."""
    if frame.empty:
        return {
            "rows": 0,
            "users": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "positive_rate": 0.0,
        }
    positives = int(frame["label"].sum())
    return {
        "rows": len(frame),
        "users": int(frame["userId"].nunique()),
        "positive_rows": positives,
        "negative_rows": int(len(frame) - positives),
        "positive_rate": float(positives / len(frame)),
    }
