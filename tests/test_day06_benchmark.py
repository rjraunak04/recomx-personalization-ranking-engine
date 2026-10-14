from __future__ import annotations

from pathlib import Path


def test_day06_acceptance_assets_exist() -> None:
    required = [
        Path("configs/serving.yaml"),
        Path("src/recomx/serving/api.py"),
        Path("src/recomx/serving/artifacts.py"),
        Path("src/recomx/serving/contract.py"),
        Path("src/recomx/serving/explanations.py"),
        Path("src/recomx/serving/fallback.py"),
        Path("src/recomx/serving/service.py"),
        Path("scripts/run_day06_benchmark.py"),
        Path("docs/day06_serving.md"),
    ]
    assert all(path.is_file() for path in required)
