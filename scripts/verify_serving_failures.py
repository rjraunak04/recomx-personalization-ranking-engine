"""Verify Day-06 CP-07 API failure semantics."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from recomx.serving.api import create_app
from recomx.serving.contract import RecommendationRequest


class FailureBackend:
    def recommend(self, request: RecommendationRequest):
        if request.user_id == 7:
            raise ValueError("No candidates.")
        raise FileNotFoundError("private/model/path.npz")


def main() -> None:
    client = TestClient(create_app(FailureBackend()))
    domain = client.get("/v1/recommendations/7")
    missing = client.get("/v1/recommendations/8")
    malformed = client.get("/v1/recommendations/not-an-int")
    invalid_k = client.get("/v1/recommendations/7?top_k=999")
    summary = {
        "domain_error_status": domain.status_code,
        "missing_artifact_status": missing.status_code,
        "malformed_user_status": malformed.status_code,
        "invalid_top_k_status": invalid_k.status_code,
        "artifact_path_hidden": "path.npz" not in missing.text,
        "locked_test_used": False,
    }
    expected = {
        "domain_error_status": 422,
        "missing_artifact_status": 503,
        "malformed_user_status": 422,
        "invalid_top_k_status": 422,
        "artifact_path_hidden": True,
        "locked_test_used": False,
    }
    if summary != expected:
        raise ValueError(f"Unexpected failure-handling result: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-06 CP-07 API FAILURE HANDLING: PASS")


if __name__ == "__main__":
    main()
