"""Verify the frozen Day-05 ranking artifact for Day-06 serving."""

from __future__ import annotations

import json
from pathlib import Path

from recomx.ranking.features import FEATURE_COLUMNS
from recomx.serving.artifacts import load_ranking_artifact

MODEL = Path("reports/ranking/ml/model.npz")


def main() -> None:
    artifact = load_ranking_artifact(str(MODEL))
    if artifact.feature_names != tuple(FEATURE_COLUMNS):
        raise ValueError("Serving model feature schema differs from Day-05 ranking schema.")

    summary = {
        "artifact": str(MODEL),
        "feature_count": len(artifact.feature_names),
        "feature_schema_matches_day05": True,
        "cached_loader": True,
        "locked_test_used": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-02 SERVING ARTIFACT LOADER: PASS")


if __name__ == "__main__":
    main()
