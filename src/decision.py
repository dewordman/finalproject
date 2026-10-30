# src/decision.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["Legitimate", "Spam", "Phishing"]
BANDS = ["low", "moderate", "high"]


def load_decision_table(path: Path) -> Dict[str, Any]:
    """Load and validate the decision table from YAML."""
    with path.open("r", encoding="utf-8") as f:
        table = yaml.safe_load(f)

    for required in ("rule_version", "t_low", "t_high", "actions"):
        if required not in table:
            raise ValueError(f"decision_table.yaml is missing required field: {required}")

    for cls in CLASS_NAMES:
        if cls not in table["actions"]:
            raise ValueError(f"decision_table.yaml is missing actions for class: {cls}")
        for band in BANDS:
            if band not in table["actions"][cls]:
                raise ValueError(f"decision_table.yaml is missing actions[{cls}][{band}]")

    return table


def decide(confidence: float, table: Dict[str, Any]) -> str:
    """Return the confidence band: 'low', 'moderate', or 'high'.

    - low       if confidence < t_low
    - high      if confidence >= t_high
    - moderate  otherwise
    """
    t_low = table["t_low"]
    t_high = table["t_high"]

    if t_low is None or t_high is None:
        raise ValueError("t_low and t_high must be set in the decision table before calling decide().")
    if not (t_low < t_high):
        raise ValueError(f"t_low ({t_low}) must be strictly less than t_high ({t_high}).")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1]; got {confidence}")

    if confidence < t_low:
        return "low"
    elif confidence >= t_high:
        return "high"
    else:
        return "moderate"


def lookup_action(class_name: str, band: str, table: Dict[str, Any]) -> str:
    """Return the advisory action string for the given class and band."""
    if class_name not in table["actions"]:
        raise ValueError(f"Unknown class: {class_name}")
    if band not in BANDS:
        raise ValueError(f"Unknown band: {band}")
    return table["actions"][class_name][band]