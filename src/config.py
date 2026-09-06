"""Config loading + run directory creation.

Contract: load_config() resolves a YAML config, stamps it with the current git
SHA, and writes the resolved copy to runs/<id>/config.yaml. Nothing else in the
pipeline may read configs/ directly -- stages read the resolved copy, so a run
is always reproducible from its own directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    """Current commit SHA, with a -dirty suffix if the tree has uncommitted changes."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.call(
        ["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT
    )
    return f"{sha}-dirty" if dirty else sha


def load_config(path: str, run_id: str) -> dict:
    """Resolve a config, stamp it with run_id + git SHA, write runs/<id>/config.yaml."""
    raise NotImplementedError("Stage 0")


def write_metrics(run_id: str, stage: str, metrics: dict) -> None:
    """Merge a stage's metrics into runs/<id>/metrics.json.

    metrics.json is committed. Keep it small and diffable -- scalars and short
    lists only, no histograms or per-feature arrays.
    """
    raise NotImplementedError("Stage 0")
