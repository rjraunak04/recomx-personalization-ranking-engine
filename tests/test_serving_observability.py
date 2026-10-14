import json
import logging

from recomx.serving.observability import ServingEvent, emit_serving_event


def test_event_is_machine_readable_and_contains_core_fields() -> None:
    payload = json.loads(
        ServingEvent(
            event="recommendation_request",
            status="success",
            latency_ms=12.345,
            strategy="personalized",
            recommendation_count=20,
        ).to_json()
    )
    assert payload["event"] == "recommendation_request"
    assert payload["strategy"] == "personalized"
    assert payload["recommendation_count"] == 20


def test_event_omits_optional_null_fields() -> None:
    payload = json.loads(
        ServingEvent(event="recommendation_request", status="error", latency_ms=1.0).to_json()
    )
    assert "strategy" not in payload
    assert "error_type" not in payload


def test_logger_emits_json(caplog) -> None:
    logger = logging.getLogger("recomx.test")
    with caplog.at_level(logging.INFO, logger="recomx.test"):
        emit_serving_event(
            logger,
            ServingEvent(
                event="recommendation_request",
                status="success",
                latency_ms=2.0,
                strategy="cold_start_popularity",
                recommendation_count=5,
            ),
        )
    payload = json.loads(caplog.records[-1].message)
    assert payload["status"] == "success"


def test_event_schema_has_no_user_or_path_fields() -> None:
    payload = ServingEvent(
        event="recommendation_request",
        status="error",
        latency_ms=3.0,
        error_type="FileNotFoundError",
    ).to_json()
    assert "user_id" not in payload
    assert "path" not in payload
