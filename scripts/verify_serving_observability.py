"""Verify Day-07 CP-02 structured serving observability."""

from __future__ import annotations

import json

from recomx.serving.observability import ServingEvent


def main() -> None:
    success = json.loads(
        ServingEvent(
            event="recommendation_request",
            status="success",
            latency_ms=8.25,
            strategy="personalized",
            recommendation_count=20,
        ).to_json()
    )
    failure = json.loads(
        ServingEvent(
            event="recommendation_request",
            status="error",
            latency_ms=1.5,
            error_type="FileNotFoundError",
        ).to_json()
    )
    summary = {
        "success_machine_readable": success["status"] == "success",
        "strategy_recorded": success["strategy"] == "personalized",
        "recommendation_count_recorded": success["recommendation_count"] == 20,
        "error_type_recorded": failure["error_type"] == "FileNotFoundError",
        "user_id_logged": "user_id" in success or "user_id" in failure,
        "local_path_logged": "path" in success or "path" in failure,
        "locked_test_used": False,
    }
    if not all(
        (
            summary["success_machine_readable"],
            summary["strategy_recorded"],
            summary["recommendation_count_recorded"],
            summary["error_type_recorded"],
            not summary["user_id_logged"],
            not summary["local_path_logged"],
            not summary["locked_test_used"],
        )
    ):
        raise ValueError(f"Observability verification failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-02 STRUCTURED OBSERVABILITY: PASS")


if __name__ == "__main__":
    main()
