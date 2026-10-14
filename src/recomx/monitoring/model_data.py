"""Model and data health checks for production serving artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonitoringThresholds:
    missing_rate_warn: float = 0.01
    mean_shift_warn: float = 3.0
    score_mean_min: float = 0.05
    score_mean_max: float = 0.95
    cold_start_share_warn: float = 0.80


_DEFAULT_THRESHOLDS = MonitoringThresholds()


def monitor_features(
    values: np.ndarray,
    feature_names: Sequence[str],
    reference_mean: np.ndarray,
    reference_scale: np.ndarray,
    thresholds: MonitoringThresholds = _DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    matrix = np.asarray(values, dtype=float)
    mean = np.asarray(reference_mean, dtype=float)
    scale = np.asarray(reference_scale, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("Feature matrix does not match the feature schema.")
    if mean.shape != (matrix.shape[1],) or scale.shape != (matrix.shape[1],):
        raise ValueError("Reference statistics do not match the feature schema.")
    if np.any(scale <= 0):
        raise ValueError("Reference feature scales must be positive.")

    missing_rate = np.mean(~np.isfinite(matrix), axis=0)
    safe = np.where(np.isfinite(matrix), matrix, mean)
    current_mean = np.mean(safe, axis=0)
    standardized_shift = np.abs(current_mean - mean) / scale

    alerts: list[str] = []
    for idx, name in enumerate(feature_names):
        if missing_rate[idx] > thresholds.missing_rate_warn:
            alerts.append(f"missing_rate:{name}")
        if standardized_shift[idx] > thresholds.mean_shift_warn:
            alerts.append(f"mean_shift:{name}")

    return {
        "row_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "max_missing_rate": float(np.max(missing_rate)) if matrix.size else 0.0,
        "max_standardized_mean_shift": (
            float(np.max(standardized_shift)) if matrix.size else 0.0
        ),
        "alert_count": len(alerts),
        "alerts": tuple(alerts),
    }


def monitor_model_scores(
    scores: np.ndarray,
    thresholds: MonitoringThresholds = _DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Model score monitoring requires at least one score.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Model scores must be finite.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Model scores must be probabilities in [0, 1].")

    mean_score = float(np.mean(values))
    alerts = []
    if not thresholds.score_mean_min <= mean_score <= thresholds.score_mean_max:
        alerts.append("score_mean_out_of_range")
    return {
        "score_count": int(values.size),
        "score_mean": mean_score,
        "score_min": float(np.min(values)),
        "score_max": float(np.max(values)),
        "alert_count": len(alerts),
        "alerts": tuple(alerts),
    }


def monitor_serving_mix(
    metrics: Mapping[str, int | float],
    thresholds: MonitoringThresholds = _DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    cold_share = float(metrics.get("cold_start_share", 0.0))
    alerts = []
    if cold_share > thresholds.cold_start_share_warn:
        alerts.append("cold_start_share_high")
    return {
        "cold_start_share": cold_share,
        "alert_count": len(alerts),
        "alerts": tuple(alerts),
    }
