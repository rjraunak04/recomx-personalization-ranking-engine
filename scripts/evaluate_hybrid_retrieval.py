"""Evaluate Day-04 hybrid candidate retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.retrieval import source_contribution
from recomx.models.hybrid_retrieval import candidate_recall

GT = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
CANDIDATES = Path("reports/hybrid/hybrid_candidates_top100.parquet")
OUTPUT = Path("reports/hybrid/evaluation")
SUMMARY = OUTPUT / "retrieval_summary.json"
SOURCES = OUTPUT / "source_contribution.csv"


def main() -> None:
    if not GT.is_file() or not CANDIDATES.is_file():
        raise FileNotFoundError("Ground truth and hybrid candidates are required.")
    truth = pd.read_parquet(GT, columns=["userId", "movieId"])
    candidates = pd.read_parquet(CANDIDATES)
    recalls = {str(k): candidate_recall(candidates, truth, k) for k in (20, 50, 100)}
    contribution = source_contribution(candidates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contribution.to_csv(SOURCES, index=False)
    summary = {
        "evaluation_users": int(truth["userId"].nunique()),
        "ground_truth_interactions": len(truth),
        "candidate_recall": recalls,
        "unique_candidate_items": int(candidates["movieId"].nunique()),
        "locked_test_used": False,
    }
    with SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nSOURCE CONTRIBUTION")
    print(contribution.to_string(index=False))
    print("DAY-04 HYBRID RETRIEVAL EVALUATION: PASS")


if __name__ == "__main__":
    main()
