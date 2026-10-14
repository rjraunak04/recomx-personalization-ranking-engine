import numpy as np
import pytest

from recomx.monitoring.model_data import (
    MonitoringThresholds,
    monitor_features,
    monitor_model_scores,
    monitor_serving_mix,
)


def test_feature_monitor_is_healthy_near_reference() -> None:
    result = monitor_features(
        np.array([[0.0, 10.0], [0.5, 9.0]]),
        ("a", "b"),
        np.array([0.0, 10.0]),
        np.array([1.0, 2.0]),
    )
    assert result["row_count"] == 2
    assert result["alert_count"] == 0


def test_feature_monitor_detects_missingness_and_shift() -> None:
    result = monitor_features(
        np.array([[np.nan, 20.0], [5.0, 20.0]]),
        ("a", "b"),
        np.array([0.0, 10.0]),
        np.array([1.0, 2.0]),
        MonitoringThresholds(missing_rate_warn=0.1, mean_shift_warn=2.0),
    )
    assert "missing_rate:a" in result["alerts"]
    assert "mean_shift:a" in result["alerts"]
    assert "mean_shift:b" in result["alerts"]


def test_feature_monitor_rejects_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="schema"):
        monitor_features(
            np.ones((2, 2)),
            ("only_one",),
            np.zeros(2),
            np.ones(2),
        )


def test_model_score_monitor_is_healthy() -> None:
    result = monitor_model_scores(np.array([0.2, 0.5, 0.8]))
    assert result["score_mean"] == pytest.approx(0.5)
    assert result["alert_count"] == 0


def test_model_score_monitor_detects_bad_mean() -> None:
    result = monitor_model_scores(np.array([0.99, 0.99]))
    assert "score_mean_out_of_range" in result["alerts"]


def test_model_score_monitor_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        monitor_model_scores(np.array([0.5, 1.2]))


def test_serving_mix_detects_high_cold_start_share() -> None:
    result = monitor_serving_mix({"cold_start_share": 0.9})
    assert result["alerts"] == ("cold_start_share_high",)
