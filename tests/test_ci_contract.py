from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_has_required_quality_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for required in (
        "actions/checkout@v4",
        "actions/setup-python@v5",
        'python-version: ["3.11", "3.12"]',
        "python -m ruff check .",
        "python -m compileall -q src scripts",
        "git diff --check",
        "python -m pytest -q",
    ):
        assert required in workflow


def test_ci_uses_least_privilege_and_concurrency() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "contents: read" in workflow
    assert "cancel-in-progress: true" in workflow


def test_container_contract_checks_production_entrypoint() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "Container contract" in workflow
    assert "USER recomx" in workflow
    assert "recomx.serving.production:create_production_app" in workflow
    assert '"--factory"' in workflow
    assert '! grep -q "COPY data/processed ./" Dockerfile' in workflow
    assert '! grep -q "ratings_test_locked" Dockerfile' in workflow


def test_local_quality_runner_covers_same_core_gates() -> None:
    runner = (ROOT / "scripts" / "run_quality_gates.py").read_text(encoding="utf-8")
    assert '"ruff", "check", "."' in runner
    assert '"compileall", "-q", "src", "scripts"' in runner
    assert '"pytest", "-q"' in runner
    assert '"git", "diff", "--check"' in runner
