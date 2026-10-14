"""Verify Day-07 CP-05 deployment contract without starting a server."""

from __future__ import annotations

import json

from recomx.serving.runtime import RuntimeSettings


def main() -> None:
    settings = RuntimeSettings.from_env()
    settings.validate(require_artifacts=True)
    readiness = settings.readiness()
    summary = {
        "artifacts_ready": readiness["artifacts_ready"],
        "environment": settings.environment,
        "host_configured": bool(settings.host),
        "port": settings.port,
        "locked_test_used": settings.locked_test_access,
        "production_entrypoint": "recomx.serving.production:app",
    }
    if not summary["artifacts_ready"] or summary["locked_test_used"]:
        raise ValueError(f"Deployment readiness failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-05 CONTAINER/DEPLOYMENT READINESS: PASS")


if __name__ == "__main__":
    main()
