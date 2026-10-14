"""Deterministic smoke/load benchmark primitives for RecomX serving."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from recomx.serving.contract import RecommendationRequest, RecommendationResponse


class BenchmarkBackend(Protocol):
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse: ...


@dataclass(frozen=True)
class BenchmarkResult:
    requests: int
    successes: int
    errors: int
    mean_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    throughput_rps: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.requests if self.requests else 0.0


def run_backend_benchmark(
    backend: BenchmarkBackend,
    requests: tuple[RecommendationRequest, ...],
) -> BenchmarkResult:
    if not requests:
        raise ValueError("Benchmark requires at least one request.")

    latencies: list[float] = []
    successes = 0
    errors = 0
    started = perf_counter()
    for request in requests:
        request_started = perf_counter()
        try:
            response = backend.recommend(request)
            response.validate()
            successes += 1
        except (ValueError, FileNotFoundError):
            errors += 1
        latencies.append((perf_counter() - request_started) * 1000)
    elapsed = perf_counter() - started
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return BenchmarkResult(
        requests=len(requests),
        successes=successes,
        errors=errors,
        mean_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=ordered[p95_index],
        max_latency_ms=max(latencies),
        throughput_rps=len(requests) / elapsed if elapsed > 0 else 0.0,
    )
