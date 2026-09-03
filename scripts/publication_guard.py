"""Enforce the repository's reviewed coursework-publication boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {".csv", ".exe", ".exp", ".ilk", ".lib", ".log", ".obj", ".pdb"}
FORBIDDEN_ROOTS = {"build", "out", ".vs"}


def tracked_paths() -> list[PurePosixPath]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], text=True, encoding="utf-8"
    )
    return [PurePosixPath(item) for item in output.split("\0") if item]


def main() -> None:
    violations: list[str] = []
    for path in tracked_paths():
        if path.parts and path.parts[0] in FORBIDDEN_ROOTS:
            violations.append(f"generated directory is tracked: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"generated binary/result is tracked: {path}")

    readme = " ".join(Path("README.md").read_text(encoding="utf-8").split())
    required = (
        "trusted, homogeneous lab network",
        "implemented but unverified",
        "does not claim a universal speedup",
        "--serial",
        "--listen 127.0.0.1",
    )
    for statement in required:
        if statement not in readme:
            violations.append(f"README is missing required boundary: {statement}")
    if "--bind" in readme:
        violations.append("README still advertises the unsupported --bind option")
    if "/openmp:experimental" in readme.lower():
        violations.append("README still requires the experimental OpenMP switch")
    if Path("lu_driver1.cpp").exists():
        violations.append("historical duplicate driver remains on the maintained root surface")

    if violations:
        raise SystemExit("Publication guard failed:\n- " + "\n- ".join(violations))
    print("Publication guard passed.")


if __name__ == "__main__":
    main()
