"""Profile the Day-02 relevance policy on RecomX training data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recomx.evaluation.protocol import DEFAULT_EVALUATION_PROTOCOL
from recomx.evaluation.relevance import (
    RelevancePolicy,
    build_relevance_profile,
)

TRAIN_PATH = Path("data/processed/ratings_train.parquet")
REPORT_PATH = Path("reports/baselines/relevance_profile.json")


def main() -> None:
    """Run relevance profiling on the leakage-safe training partition."""

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training partition not found: {TRAIN_PATH}")

    protocol = DEFAULT_EVALUATION_PROTOCOL
    protocol.validate()

    policy = RelevancePolicy(minimum_rating=protocol.relevance_threshold)

    print("RecomX CP-02 Relevance Profile")
    print("=" * 72)
    print(f"source={TRAIN_PATH}")
    print(f"minimum_rating={policy.minimum_rating:.1f}")
    print("Loading training interactions...")

    frame = pd.read_parquet(
        TRAIN_PATH,
        columns=["userId", "movieId", "rating"],
    )

    profile = build_relevance_profile(
        frame,
        policy,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            profile.as_dict(),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print()
    print(f"total_interactions={profile.total_interactions:,}")
    print(f"relevant_interactions={profile.relevant_interactions:,}")
    print(f"non_relevant_interactions={profile.non_relevant_interactions:,}")
    print(f"relevance_rate={profile.relevance_rate:.6f}")
    print(f"total_users={profile.total_users:,}")
    print(f"relevant_users={profile.relevant_users:,}")
    print(f"total_items={profile.total_items:,}")
    print(f"relevant_items={profile.relevant_items:,}")
    print(f"report={REPORT_PATH}")
    print()
    print("CP-02 RELEVANCE POLICY PASSED")


if __name__ == "__main__":
    main()
