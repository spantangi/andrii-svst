"""Config loading + run directory creation.

Contract: load_config() resolves a YAML config, stamps it with the current git
SHA, and writes the resolved copy to runs/<id>/config.yaml. Nothing else in the
pipeline reads configs/ directly -- stages read the resolved copy, so a run is
always reproducible from its own directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    """Current commit SHA, with a -dirty suffix if the tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"{sha}-dirty" if dirty else sha


def load_config(path: str, run_id: str) -> dict:
    """Resolve a config, stamp it with run_id + git SHA, write runs/<id>/config.yaml."""
    cfg = yaml.safe_load((REPO_ROOT / path).read_text())
    cfg.setdefault("run", {})
    cfg["run"]["id"] = run_id
    cfg["run"]["git_sha"] = git_sha()
    cfg["run"]["source_config"] = path
    cfg["run"].setdefault("seed", 0)

    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


def write_metrics(run_id: str, stage: str, metrics: dict) -> None:
    """Merge a stage's metrics into runs/<id>/metrics.json.

    metrics.json is committed. Keep it small and diffable -- scalars and short
    lists only, no histograms or per-feature arrays.
    """
    path = REPO_ROOT / "runs" / run_id / "metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[stage] = metrics
    existing.setdefault("_git_sha", git_sha())
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
