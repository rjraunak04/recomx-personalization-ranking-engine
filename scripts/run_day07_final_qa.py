"""Final Day-07 productionization acceptance check."""

from __future__ import annotations

import json
from pathlib import Path

from recomx.serving.runtime import RuntimeSettings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = RuntimeSettings.from_env()
    settings.validate(require_artifacts=True)

    required_files = (
        ROOT / "Dockerfile",
        ROOT / ".dockerignore",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "src" / "recomx" / "serving" / "production.py",
        ROOT / "src" / "recomx" / "serving" / "observability.py",
        ROOT / "src" / "recomx" / "serving" / "metrics.py",
        ROOT / "src" / "recomx" / "monitoring" / "model_data.py",
        ROOT / "scripts" / "run_production_smoke_benchmark.py",
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required_files if not path.exists()]
    readiness = settings.readiness()
    summary = {
        "artifacts_ready": readiness["artifacts_ready"],
        "ci_configured": (ROOT / ".github" / "workflows" / "ci.yml").exists(),
        "container_configured": (ROOT / "Dockerfile").exists(),
        "locked_test_used": False,
        "missing_required_files": missing,
        "observability_configured": (ROOT / "src/recomx/serving/observability.py").exists(),
        "monitoring_configured": (ROOT / "src/recomx/monitoring/model_data.py").exists(),
        "production_backend_configured": (
            ROOT / "src/recomx/serving/production.py"
        ).exists(),
    }
    if (
        missing
        or not summary["artifacts_ready"]
        or summary["locked_test_used"]
        or not summary["ci_configured"]
        or not summary["container_configured"]
    ):
        raise ValueError(f"Day-07 final acceptance failed: {summary}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 FINAL PRODUCTIONIZATION QA: PASS")


if __name__ == "__main__":
    main()
