# src/run_log.py
from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.utils import get_logger

logger = get_logger(__name__)


_TRACKED_PACKAGES = (
    "scikit-learn", "pandas", "numpy", "scipy", "joblib", "beautifulsoup4", "lxml",
    "langdetect", "PyYAML", "matplotlib", "seaborn", "streamlit", "pytest",
)
_MODULE_NAMES = {"scikit-learn": "sklearn", "beautifulsoup4": "bs4", "PyYAML": "yaml"}


def _package_versions() -> Dict[str, str]:
    """Installed versions via importlib.metadata, falling back to module.__version__."""
    from importlib import import_module
    from importlib.metadata import PackageNotFoundError, version

    versions: Dict[str, str] = {"python": platform.python_version()}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
            continue
        except PackageNotFoundError:
            pass
        try:
            versions[package] = str(import_module(_MODULE_NAMES.get(package, package.lower())).__version__)
        except Exception:
            versions[package] = "not installed"
    return versions


def write_environment_freeze(results_dir: Path) -> Path:
    """Write ``environment_freeze.txt`` (pip freeze) once per results directory.

    An existing file is never overwritten. Failures are recorded inside the file
    instead of stopping the run.
    """
    import subprocess
    import sys

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "environment_freeze.txt"
    if target.exists():
        return target
    header = f"# python {platform.python_version()} on {platform.platform()}\n"
    try:
        output = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120, check=True).stdout
    except Exception as exc:  # pip missing, offline, timeout ...
        output = f"# pip freeze failed: {exc}\n"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(header + output)
    return target


def _hardware_info() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": str(os.cpu_count()),
    }

def config_digest(cfg_path: Path) -> str:
    return hashlib.sha256(cfg_path.read_bytes()).hexdigest()


def init_run_log(cfg_path: Path, seed: int) -> Dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "config_digest": config_digest(cfg_path),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "hardware": _hardware_info(),
        "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "stages": [],
    }


def start_stage(log: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    log["stages"].append({
        "name": stage_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "outputs": [],
        "status": "running",
    })
    return log


def end_stage(
    log: Dict[str, Any],
    outputs: List[str],
    status: str = "ok",
) -> Dict[str, Any]:
    stage = log["stages"][-1]
    completed = datetime.now(timezone.utc)
    stage["completed_at"] = completed.isoformat()
    started = datetime.fromisoformat(stage["started_at"])
    stage["duration_seconds"] = round((completed - started).total_seconds(), 3)
    stage["outputs"] = outputs
    stage["status"] = status
    return log


def finalise_run_log(log: Dict[str, Any]) -> Dict[str, Any]:
    log["completed_at"] = datetime.now(timezone.utc).isoformat()
    return log


def save_run_log(log: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    logger.info(f"Saved run log to {path}")