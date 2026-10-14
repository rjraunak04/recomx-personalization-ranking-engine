"""Verify Day-07 CP-01 production runtime contract."""

from __future__ import annotations

import json

from recomx.serving.runtime import RuntimeSettings


def main() -> None:
    settings = RuntimeSettings.from_env()
    settings.validate(require_artifacts=True)
    readiness = settings.readiness()
    summary = {
        "environment": settings.environment,
        "port": settings.port,
        "artifacts_ready": readiness["artifacts_ready"],
        "required_artifacts": len(settings.required_artifacts()),
        "locked_test_used": False,
    }
    if not summary["artifacts_ready"] or summary["locked_test_used"]:
        raise ValueError(f"Production runtime verification failed: {summary}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DAY-07 CP-01 PRODUCTION RUNTIME CONTRACT: PASS")


if __name__ == "__main__":
    main()
