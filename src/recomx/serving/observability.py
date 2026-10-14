"""Structured, privacy-conscious observability for recommendation serving."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ServingEvent:
    event: str
    status: str
    latency_ms: float
    strategy: str | None = None
    recommendation_count: int | None = None
    error_type: str | None = None

    def to_json(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def emit_serving_event(logger: logging.Logger, event: ServingEvent) -> None:
    """Emit one machine-readable serving event without user IDs or local paths."""
    logger.info(event.to_json())
