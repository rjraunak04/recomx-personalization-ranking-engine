"""Verify Day-07 CP-03 recommendation metrics monitoring."""

from __future__ import annotations

import json

from recomx.serving.metrics import ServingMetrics
from recomx.serving.observability import ServingEvent


def main() -> None:
    metrics = ServingMetrics()
    events = (
        ServingEvent(
            event="recommendation_request",
            status="success",
            latency_ms=10.0,
            strategy="personalized",
            recommendation_count=20,
        ),
        ServingEvent(
            event="recommendation_request",
            status="success",
            latency_ms=30.0,
            strategy="cold_start_popularity",
            recommendation_count=10,
        ),
        ServingEvent(
            event="recommendation_request",
            status="error",
            latency_ms=5.0,
            error_type="ValueError",
        ),
    )
    for event in events:
        metrics.observe(event)
    summary = metrics.snapshot()
    expected = {
        "request_count": 3,
        "success_count": 2,
        "error_count": 1,
        "recommendation_count": 30,
        "mean_latency_ms": 15.0,
        "max_latency_ms": 30.0,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"Unexpected metric {key}: {summary[key]} != {value}")
    if summary["personalized_share"] != 0.5:
        raise ValueError("Unexpected personalized share.")
    if summary["cold_start_share"] != 0.5:
        raise ValueError("Unexpected cold-start share.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-03 RECOMMENDATION METRICS MONITORING: PASS")


if __name__ == "__main__":
    main()
