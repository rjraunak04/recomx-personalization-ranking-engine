from recomx.serving.metrics import ServingMetrics
from recomx.serving.observability import ServingEvent


def _event(
    *,
    status: str = "success",
    latency_ms: float = 10.0,
    strategy: str | None = "personalized",
    count: int | None = 20,
) -> ServingEvent:
    return ServingEvent(
        event="recommendation_request",
        status=status,
        latency_ms=latency_ms,
        strategy=strategy,
        recommendation_count=count,
    )


def test_empty_snapshot_is_zero_safe() -> None:
    snapshot = ServingMetrics().snapshot()
    assert snapshot["request_count"] == 0
    assert snapshot["success_rate"] == 0.0
    assert snapshot["mean_latency_ms"] == 0.0


def test_metrics_aggregate_success_and_error_rates() -> None:
    metrics = ServingMetrics()
    metrics.observe(_event())
    metrics.observe(_event(status="error", strategy=None, count=None))
    snapshot = metrics.snapshot()
    assert snapshot["request_count"] == 2
    assert snapshot["success_count"] == 1
    assert snapshot["error_count"] == 1
    assert snapshot["success_rate"] == 0.5
    assert snapshot["error_rate"] == 0.5


def test_metrics_track_strategy_mix_and_recommendations() -> None:
    metrics = ServingMetrics()
    metrics.observe(_event(strategy="personalized", count=20))
    metrics.observe(_event(strategy="cold_start_popularity", count=10))
    snapshot = metrics.snapshot()
    assert snapshot["personalized_share"] == 0.5
    assert snapshot["cold_start_share"] == 0.5
    assert snapshot["recommendation_count"] == 30
    assert snapshot["mean_recommendations"] == 15.0


def test_metrics_track_mean_and_max_latency() -> None:
    metrics = ServingMetrics()
    metrics.observe(_event(latency_ms=10.0))
    metrics.observe(_event(latency_ms=30.0))
    snapshot = metrics.snapshot()
    assert snapshot["mean_latency_ms"] == 20.0
    assert snapshot["max_latency_ms"] == 30.0


def test_metrics_ignore_non_recommendation_events() -> None:
    metrics = ServingMetrics()
    metrics.observe(
        ServingEvent(event="health_check", status="success", latency_ms=1.0)
    )
    assert metrics.snapshot()["request_count"] == 0
