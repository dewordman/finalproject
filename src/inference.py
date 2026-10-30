# src/inference.py
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.config import require
from src.decision import decide, lookup_action, load_decision_table
from src.explain import explain_prediction
from src.text_build import build_model_text
from src.utils import get_logger

logger = get_logger(__name__)


def predict_email(
    subject: str,
    body: str,
    bundle: Dict[str, Any],
    table: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict email threat classification with full explanation and decision support.
    
    Args:
        subject: Email subject line
        body: Email body text
        bundle: Loaded model bundle containing pipeline, calibrator, metadata
        table: Decision table with thresholds and actions
    
    Returns:
        Dictionary with prediction results, explanations, and decision
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Input validation
    subject_clean = subject.strip() if subject else ""
    body_clean = body.strip() if body else ""
    
    # T03: Both empty
    if not subject_clean and not body_clean:
        return {
            "status": "rejected",
            "message": "Both subject and body are empty. Please provide at least one.",
            "timestamp": timestamp,
        }
    
    # Build model text
    model_text = build_model_text(
        subject_clean, body_clean, bundle["mask_terms"],
        mask_dates=bool(bundle.get("mask_dates", False)),
        options=bundle.get("text_options"),
    )
    
    # T09: Oversized input
    max_chars = bundle.get("max_input_chars", 20000)
    if len(model_text) > max_chars:
        return {
            "status": "rejected",
            "message": f"Input exceeds maximum length of {max_chars} characters.",
            "timestamp": timestamp,
        }
    
    # Get prediction
    pipeline = bundle["pipeline"]
    calibrator = bundle.get("calibrator")
    cal_method = bundle.get("calibration_method", "raw")
    
    # Predict class
    pred_class_idx = int(pipeline.predict([model_text])[0])
    class_name = bundle["class_map"][pred_class_idx]
    
    # Get confidence
    if calibrator is not None:
        # Use calibrated probabilities
        proba = calibrator.predict_proba([model_text])[0]
        confidence = float(proba[pred_class_idx])
    else:
        # Use raw probabilities
        proba = pipeline.predict_proba([model_text])[0]
        confidence = float(proba[pred_class_idx])
    
    # Get explanations
    explanation = explain_prediction(pipeline, model_text, top_n=5)
    
    supporting_terms = [
        {"term": t["term"], "contribution": t["contribution"]}
        for t in explanation["supporting"]
    ]
    
    opposing_terms = [
        {"term": t["term"], "contribution": t["contribution"]}
        for t in explanation["opposing"]
    ]
    
    # Apply decision rules
    band = decide(confidence, table)
    action = lookup_action(class_name, band, table)
    
    return {
        "status": "success",
        "predicted_class": class_name,
        "confidence": round(confidence, 4),
        "confidence_method": cal_method,
        "probabilities": {bundle["class_map"][i]: round(float(value), 4) for i, value in enumerate(proba)},
        "band": band,
        "action": action,
        "supporting_terms": supporting_terms,
        "opposing_terms": opposing_terms,
        "model_version": bundle["model_version"],
        "rule_version": table["rule_version"],
        "timestamp": timestamp,
    }


def load_bundle(bundle_path: Path) -> Dict[str, Any]:
    """Load a model bundle from disk."""
    bundle = joblib.load(bundle_path)
    logger.info(f"Loaded bundle from {bundle_path}")
    logger.info(f"Model version: {bundle['model_version']}")
    logger.info(f"Calibration method: {bundle.get('calibration_method', 'raw')}")
    return bundle


def compute_text_hash(subject: str, body: str) -> str:
    """Compute SHA-256 hash of subject+body for logging."""
    combined = f"{subject}{body}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def log_prediction(
    result: Dict[str, Any],
    subject: str,
    body: str,
    log_path: Path,
) -> None:
    """Log prediction to CSV file (without storing actual text)."""
    import uuid
    
    if result["status"] == "rejected":
        return
    
    record = {
        "record_uuid": str(uuid.uuid4()),
        "timestamp": result["timestamp"],
        "model_version": result["model_version"],
        "rule_version": result["rule_version"],
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "band": result["band"],
        "text_hash": compute_text_hash(subject, body),
    }
    
    # Create logs directory if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to CSV
    df = pd.DataFrame([record])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
    
    logger.info(f"Logged prediction to {log_path}")