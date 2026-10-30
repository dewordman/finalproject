# src/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def require(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Retrieve a value from the config using a dotted key.
    
    Raises ValueError if the key is missing or its value is None.
    """
    keys = dotted_key.split(".")
    current = cfg
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            raise ValueError(f"Missing required configuration key: '{dotted_key}'")
        current = current[k]
    if current is None:
        raise ValueError(f"Configuration key '{dotted_key}' is null and must be set.")
    return current