from pathlib import Path

import pytest

from recomx.serving.runtime import RuntimeSettings


def _settings(tmp_path: Path, **overrides) -> RuntimeSettings:
    paths = [tmp_path / name for name in ("model.npz", "features.parquet", "train.parquet", "movies.parquet")]
    values = {
        "environment": "production",
        "host": "0.0.0.0",
        "port": 8000,
        "ranking_model_path": paths[0],
        "ranking_features_path": paths[1],
        "train_ratings_path": paths[2],
        "movies_path": paths[3],
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def test_runtime_contract_accepts_valid_configuration(tmp_path: Path) -> None:
    _settings(tmp_path).validate()


def test_runtime_rejects_invalid_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _settings(tmp_path, environment="local-ish").validate()


def test_runtime_rejects_invalid_port(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="port"):
        _settings(tmp_path, port=70000).validate()


def test_runtime_forbids_locked_test_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked test"):
        _settings(tmp_path, ranking_features_path=tmp_path / "locked_test.parquet").validate()


def test_readiness_reports_missing_artifacts_without_paths(tmp_path: Path) -> None:
    result = _settings(tmp_path).readiness()
    assert result["status"] == "not_ready"
    assert result["artifacts_ready"] is False
    assert "model.npz" in result["missing_artifacts"]
    assert str(tmp_path) not in str(result)


def test_readiness_passes_when_artifacts_exist(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for path in settings.required_artifacts():
        path.touch()
    result = settings.readiness()
    assert result["status"] == "ready"
    assert result["artifacts_ready"] is True
    assert result["locked_test_access"] is False
