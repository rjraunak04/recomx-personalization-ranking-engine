"""In-process serving metrics for RecomX recommendation traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from recomx.serving.observability import ServingEvent


@dataclass
class ServingMetrics:
    """Thread-safe aggregate metrics derived from privacy-safe serving events."""

    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    personalized_count: int = 0
    cold_start_count: int = 0
    recommendation_count: int = 0
    latency_total_ms: float = 0.0
    latency_max_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def observe(self, event: ServingEvent) -> None:
        if event.event != "recommendation_request":
            return
        with self._lock:
            self.request_count += 1
            self.latency_total_ms += event.latency_ms
            self.latency_max_ms = max(self.latency_max_ms, event.latency_ms)
            if event.status == "success":
                self.success_count += 1
                self.recommendation_count += event.recommendation_count or 0
                if event.strategy == "personalized":
                    self.personalized_count += 1
                elif event.strategy == "cold_start_popularity":
                    self.cold_start_count += 1
            else:
                self.error_count += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            requests = self.request_count
            successes = self.success_count
            return {
                "request_count": requests,
                "success_count": successes,
                "error_count": self.error_count,
                "success_rate": successes / requests if requests else 0.0,
                "error_rate": self.error_count / requests if requests else 0.0,
                "personalized_count": self.personalized_count,
                "cold_start_count": self.cold_start_count,
                "personalized_share": (
                    self.personalized_count / successes if successes else 0.0
                ),
                "cold_start_share": (
                    self.cold_start_count / successes if successes else 0.0
                ),
                "recommendation_count": self.recommendation_count,
                "mean_recommendations": (
                    self.recommendation_count / successes if successes else 0.0
                ),
                "mean_latency_ms": (
                    self.latency_total_ms / requests if requests else 0.0
                ),
                "max_latency_ms": self.latency_max_ms,
            }
