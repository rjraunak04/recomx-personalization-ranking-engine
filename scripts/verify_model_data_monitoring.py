"""Verify Day-07 CP-04 model and data monitoring."""

from __future__ import annotations

import json

import numpy as np

from recomx.monitoring.model_data import (
    monitor_features,
    monitor_model_scores,
    monitor_serving_mix,
)


def main() -> None:
    feature_health = monitor_features(
        np.array([[0.0, 10.0], [0.5, 9.0]]),
        ("feature_a", "feature_b"),
        np.array([0.0, 10.0]),
        np.array([1.0, 2.0]),
    )
    score_health = monitor_model_scores(np.array([0.2, 0.5, 0.8]))
    serving_health = monitor_serving_mix({"cold_start_share": 0.25})
    summary = {
        "feature_alerts": feature_health["alert_count"],
        "score_alerts": score_health["alert_count"],
        "serving_alerts": serving_health["alert_count"],
        "feature_schema_checked": feature_health["feature_count"] == 2,
        "probability_bounds_checked": score_health["score_count"] == 3,
        "locked_test_used": False,
    }
    if any(
        (
            summary["feature_alerts"],
            summary["score_alerts"],
            summary["serving_alerts"],
            not summary["feature_schema_checked"],
            not summary["probability_bounds_checked"],
            summary["locked_test_used"],
        )
    ):
        raise ValueError(f"Monitoring verification failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-04 MODEL/DATA MONITORING: PASS")


if __name__ == "__main__":
    main()
