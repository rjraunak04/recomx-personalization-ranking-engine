"""Final Day-04 hybrid retrieval integrity benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CANDIDATES = Path("reports/hybrid/hybrid_candidates_top100.parquet")
BUILD_SUMMARY = Path("reports/hybrid/candidate_summary.json")
EVAL_SUMMARY = Path("reports/hybrid/evaluation/retrieval_summary.json")
OUTPUT = Path("reports/hybrid/benchmark/day04_benchmark_summary.json")


def main() -> None:
    for path in (CANDIDATES, BUILD_SUMMARY, EVAL_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Day-04 artifact: {path}")
    frame = pd.read_parquet(CANDIDATES)
    required = {"userId", "movieId", "score", "candidate_rank", "sources"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Candidates missing columns: {sorted(missing)}")
    if frame.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user-item candidates.")
    if frame.duplicated(["userId", "candidate_rank"]).any():
        raise ValueError("Duplicate candidate ranks.")
    if frame["candidate_rank"].gt(100).any() or frame["candidate_rank"].lt(1).any():
        raise ValueError("Candidate ranks outside 1..100.")
    with BUILD_SUMMARY.open(encoding="utf-8") as handle:
        build = json.load(handle)
    with EVAL_SUMMARY.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    if build["locked_test_used"] or evaluation["locked_test_used"]:
        raise ValueError("Locked test must remain unused.")
    summary = {
        "build": build,
        "evaluation": evaluation,
        "candidate_budget": 100,
        "locked_test_used": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-04 HYBRID RETRIEVAL BENCHMARK: PASS")


if __name__ == "__main__":
    main()
