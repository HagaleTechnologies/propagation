"""Locate and build the vendored ITURHFProp binary.

`build-p533` console script; run from anywhere inside the repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default cwd) to the directory containing baselines/p533."""
    p = (start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "baselines" / "p533").is_dir():
            return candidate
    raise FileNotFoundError("baselines/p533 not found above " + str(p))


def binary_path() -> Path:
    return repo_root() / "baselines" / "p533" / "bin" / "iturhfprop"


def main() -> None:
    script = repo_root() / "baselines" / "p533" / "build.sh"
    result = subprocess.run(["bash", str(script)], check=False)
    sys.exit(result.returncode)
