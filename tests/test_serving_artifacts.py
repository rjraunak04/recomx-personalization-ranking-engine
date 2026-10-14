from __future__ import annotations

import numpy as np
import pytest

from recomx.serving.artifacts import load_ranking_artifact


def _write_artifact(path, *, scale=None) -> None:
    np.savez(
        path,
        weights=np.array([0.1, 0.2]),
        mean=np.array([1.0, 2.0]),
        scale=np.array([1.0, 1.0]) if scale is None else scale,
        bias=np.array([0.3]),
        feature_names=np.array(["a", "b"]),
    )


def test_load_ranking_artifact(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_artifact(path)
    artifact = load_ranking_artifact(str(path))
    assert artifact.feature_names == ("a", "b")
    assert artifact.bias == pytest.approx(0.3)


def test_loader_is_cached(tmp_path) -> None:
    load_ranking_artifact.cache_clear()
    path = tmp_path / "model.npz"
    _write_artifact(path)
    first = load_ranking_artifact(str(path))
    second = load_ranking_artifact(str(path))
    assert first is second


def test_missing_artifact_fails_fast(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing ranking artifact"):
        load_ranking_artifact(str(tmp_path / "missing.npz"))


def test_invalid_scale_is_rejected(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_artifact(path, scale=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="scale"):
        load_ranking_artifact(str(path))
