"""Run local CI-equivalent RecomX production quality gates."""

from __future__ import annotations

import subprocess
import sys


def _run(label: str, command: list[str]) -> None:
    print(f"==> {label}")
    subprocess.run(command, check=True)


def main() -> None:
    python = sys.executable
    gates = (
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("compile", [python, "-m", "compileall", "-q", "src", "scripts"]),
        ("tests", [python, "-m", "pytest", "-q"]),
    )
    for label, command in gates:
        _run(label, command)
    _run("git diff hygiene", ["git", "diff", "--check"])
    print("DAY-07 CP-06 CI/CD QUALITY GATES: PASS")


if __name__ == "__main__":
    main()
