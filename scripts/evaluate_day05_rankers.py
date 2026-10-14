"""Evaluate Day-05 baseline and ML ranker on the identical held-out user cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.metrics import evaluate_recommendations

BASELINE = Path("reports/ranking/baseline/recommendations_top20.parquet")
ML = Path("reports/ranking/ml/recommendations_top20.parquet")
GROUND_TRUTH = Path("reports/baselines/ground_truth/validation_ground_truth.parquet")
OUTPUT = Path("reports/ranking/evaluation")


def _evaluate(name: str, recommendations: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    _, aggregate = evaluate_recommendations(recommendations, truth, cutoffs=(5, 10, 20))
    aggregate.insert(0, "model", name)
    return aggregate


def main() -> None:
    for path in (BASELINE, ML, GROUND_TRUTH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite artifact: {path}")

    ml = pd.read_parquet(ML)
    baseline = pd.read_parquet(BASELINE)
    truth = pd.read_parquet(GROUND_TRUTH, columns=["userId", "movieId"])

    held_out_users = set(ml["userId"].unique())
    baseline = baseline.loc[baseline["userId"].isin(held_out_users)].copy()
    truth = truth.loc[truth["userId"].isin(held_out_users)].copy()

    if set(baseline["userId"].unique()) != held_out_users:
        raise ValueError("Baseline does not cover the complete ML held-out cohort.")
    if set(truth["userId"].unique()) != held_out_users:
        raise ValueError("Ground truth does not cover the complete ML held-out cohort.")

    comparison = pd.concat(
        [
            _evaluate("retrieval_baseline", baseline, truth),
            _evaluate("ml_logistic_ranker", ml, truth),
        ],
        ignore_index=True,
    )
    pivot = comparison.pivot(index="k", columns="model", values="ndcg")
    delta = (
        pivot["ml_logistic_ranker"] - pivot["retrieval_baseline"]
    ).rename("ndcg_delta").reset_index()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT / "heldout_metrics.csv", index=False)
    delta.to_csv(OUTPUT / "ndcg_delta.csv", index=False)

    payload = {
        "held_out_users": len(held_out_users),
        "locked_test_used": False,
        "metrics": comparison.to_dict(orient="records"),
        "ndcg_delta": delta.to_dict(orient="records"),
    }
    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))
    print("DAY-05 CP-06/07 HELD-OUT RANKING EVALUATION: PASS")


if __name__ == "__main__":
    main()
