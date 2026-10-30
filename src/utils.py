# src/utils.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the global seed for random, numpy, and os.environ."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256_file(path: Union[str, Path]) -> str:
    """Compute the SHA-256 hash of a file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    """Compute the SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(p: Union[str, Path]) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: Union[str, Path]) -> None:
    """Write an object to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON file into a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_digest(cfg: Dict[str, Any]) -> str:
    """Compute a short SHA-256 digest of the configuration."""
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def new_run_id() -> str:
    """Generate a new unique run ID."""
    return uuid.uuid4().hex[:12]