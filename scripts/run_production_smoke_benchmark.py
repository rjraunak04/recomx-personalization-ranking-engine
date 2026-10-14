"""Run Day-07 CP-07 production smoke and repeated-request benchmark."""

from __future__ import annotations

import json

from recomx.serving.benchmark import run_backend_benchmark
from recomx.serving.contract import RecommendationRequest
from recomx.serving.production import ProductionRecommendationBackend
from recomx.serving.runtime import RuntimeSettings


def main() -> None:
    settings = RuntimeSettings.from_env()
    settings.validate(require_artifacts=True)
    backend = ProductionRecommendationBackend(settings)

    known_candidates = backend.candidates["userId"].dropna().astype(int).unique()
    if len(known_candidates) == 0:
        raise ValueError("No personalized serving users are available.")
    known_user = int(known_candidates[0])
    cold_user = max(max(backend.known_users), known_user) + 1

    smoke_known = backend.recommend(RecommendationRequest(known_user, 20))
    smoke_cold = backend.recommend(RecommendationRequest(cold_user, 20))

    workload = tuple(
        RecommendationRequest(known_user if index % 2 == 0 else cold_user, 20)
        for index in range(40)
    )
    result = run_backend_benchmark(backend, workload)

    summary = {
        "requests": result.requests,
        "successes": result.successes,
        "errors": result.errors,
        "success_rate": result.success_rate,
        "mean_latency_ms": result.mean_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "max_latency_ms": result.max_latency_ms,
        "throughput_rps": result.throughput_rps,
        "known_strategy": smoke_known.strategy,
        "cold_strategy": smoke_cold.strategy,
        "known_recommendations": len(smoke_known.recommendations),
        "cold_recommendations": len(smoke_cold.recommendations),
        "artifacts_ready": settings.readiness()["artifacts_ready"],
        "locked_test_used": False,
    }
    required = (
        summary["success_rate"] == 1.0
        and summary["known_strategy"] == "personalized"
        and summary["cold_strategy"] == "cold_start_popularity"
        and summary["known_recommendations"] == 20
        and summary["cold_recommendations"] == 20
        and summary["artifacts_ready"]
        and not summary["locked_test_used"]
    )
    if not required:
        raise ValueError(f"Production smoke benchmark failed: {summary}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-07 PRODUCTION SMOKE/LOAD BENCHMARK: PASS")


if __name__ == "__main__":
    main()
