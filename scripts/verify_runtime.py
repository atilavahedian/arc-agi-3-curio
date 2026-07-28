"""Verify the local ARC runtime matches the current Kaggle bundle.

The competition rerun installs fixed wheels and copies a fixed checkout of the
official agent framework.  Local measurements are comparable only when they
use those same versions.  ``agents/__init__.py`` is the one permitted working-
tree modification because ``scripts/slim_framework.py`` rewrites that registry
to avoid importing optional agent dependencies.
"""
from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"
EXPECTED_PACKAGES = {
    "arc-agi": "0.9.8",
    "arcengine": "0.9.3",
}
EXPECTED_FRAMEWORK_COMMIT = "135f20aaf44f13341ebc425666bf03b5cac58d3c"
ALLOWED_FRAMEWORK_CHANGES = {" M agents/__init__.py"}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(FRAMEWORK), *args], text=True
    ).rstrip("\n")


def main() -> None:
    problems: list[str] = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        if actual != expected:
            problems.append(f"{package}={actual}, expected {expected}")

    if not (FRAMEWORK / ".git").exists():
        problems.append(f"missing framework checkout: {FRAMEWORK}")
    else:
        actual_commit = git("rev-parse", "HEAD")
        if actual_commit != EXPECTED_FRAMEWORK_COMMIT:
            problems.append(
                f"framework={actual_commit}, expected {EXPECTED_FRAMEWORK_COMMIT}"
            )
        changes = set(filter(None, git("status", "--porcelain").splitlines()))
        unexpected = changes - ALLOWED_FRAMEWORK_CHANGES
        if unexpected:
            problems.append(
                "unexpected framework changes: " + ", ".join(sorted(unexpected))
            )

    if problems:
        raise SystemExit("runtime parity failed:\n  - " + "\n  - ".join(problems))
    print("runtime parity verified:")
    for package, expected in EXPECTED_PACKAGES.items():
        print(f"  {package}={expected}")
    print(f"  framework={EXPECTED_FRAMEWORK_COMMIT}")


if __name__ == "__main__":
    main()
