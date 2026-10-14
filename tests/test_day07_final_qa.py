from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_day07_production_components_exist() -> None:
    required = (
        "Dockerfile",
        ".dockerignore",
        ".github/workflows/ci.yml",
        "src/recomx/serving/production.py",
        "src/recomx/serving/observability.py",
        "src/recomx/serving/metrics.py",
        "src/recomx/monitoring/model_data.py",
        "scripts/run_production_smoke_benchmark.py",
        "scripts/run_quality_gates.py",
    )
    assert all((ROOT / path).exists() for path in required)


def test_readme_documents_full_pipeline_and_limitations() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "Hybrid Top-100 Candidate Retrieval",
        "ML Learning-to-Rank Model",
        "FastAPI Serving",
        "Docker + CI/CD",
        "Scope & Honest Limitations",
        "locked test",
    ):
        assert required in readme
