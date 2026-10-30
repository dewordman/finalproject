# src/error_review.py
"""Structured manual error-review sheet built from VALIDATION predictions only.

The sealed test partition is never read. Message bodies are not written to the
sheet; reviewers look them up by ``record_id`` in the protected data folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import require
from src.evaluate import _predict_with_model
from src.explain import explain_prediction
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

CATEGORIES = [
    ("legitimate_to_spam_or_phishing", lambda t, p: (t == 0) & (p != 0)),
    ("spam_to_legitimate", lambda t, p: (t == 1) & (p == 0)),
    ("phishing_to_legitimate", lambda t, p: (t == 2) & (p == 0)),
    ("phishing_to_spam", lambda t, p: (t == 2) & (p == 1)),
]
CODING_COLUMNS = ["ambiguous_intent", "promotional_wording", "impersonation_or_urgency",
                  "formatting_artefact", "plausible_source_dependence", "coder_note"]
CLASS_NAMES = {0: "Legitimate", 1: "Spam", 2: "Phishing"}


def length_band(text: str) -> str:
    length = len(text) if isinstance(text, str) else 0
    return "short" if length < 500 else "medium" if length < 2000 else "long"


def run_error_review(cfg: Dict[str, Any]) -> pd.DataFrame:
    processed = Path(require(cfg, "paths.processed_dir"))
    results = Path(require(cfg, "paths.results_dir"))
    models = Path(require(cfg, "paths.models_dir"))
    per_category = int(require(cfg, "error_review.per_category"))
    seed = int(require(cfg, "seed"))
    out_path = results / "error_review_sheet.csv"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; choose a fresh results directory")
    ensure_dir(results)

    development = pd.read_pickle(processed / "records_development.pkl")
    validation = development.loc[development["partition_id"].eq("validation")].copy()
    if validation.empty or not validation["partition_id"].eq("validation").all():
        raise AssertionError("Error review must use validation rows only")
    group_sizes = development.groupby("duplicate_group_id")["record_id"].transform("size")
    validation["duplicate_group_size"] = group_sizes.loc[validation.index].to_numpy()

    predicted, assessed, _ = _predict_with_model("logreg", models, validation["model_text"])
    predicted = np.asarray(predicted, dtype=int)
    validation["predicted_class"] = predicted
    validation["confidence"] = np.asarray(assessed)[np.arange(len(predicted)), predicted]
    pipeline = joblib.load(models / "logreg.joblib")

    true = validation["class_id"].to_numpy()
    rows = []
    for name, rule in CATEGORIES:
        selected = validation.loc[rule(true, predicted)]
        selected = selected.drop_duplicates("duplicate_group_id")  # one row per group
        if len(selected) < per_category:
            logger.warning("Only %d groups available for %s (requested %d)",
                           len(selected), name, per_category)
        take = selected.sample(n=min(per_category, len(selected)), random_state=seed)
        for _, row in take.iterrows():
            terms = explain_prediction(pipeline, row["model_text"], top_n=5)["supporting"]
            rows.append({
                "category": name, "record_id": row["record_id"],
                "source": row["source_dataset"], "length_band": length_band(row["model_text"]),
                "duplicate_group_size": int(row["duplicate_group_size"]),
                "true_class": CLASS_NAMES[int(row["class_id"])],
                "predicted_class": CLASS_NAMES[int(row["predicted_class"])],
                "confidence": round(float(row["confidence"]), 4),
                "top_supporting_terms": " | ".join(item["term"] for item in terms),
                **{column: "" for column in CODING_COLUMNS},
            })
    columns = ["category", "record_id", "source", "length_band", "duplicate_group_size",
               "true_class", "predicted_class", "confidence", "top_supporting_terms",
               *CODING_COLUMNS]
    sheet = pd.DataFrame(rows, columns=columns)  # keeps the header even with zero errors
    sheet.to_csv(out_path, index=False, encoding="utf-8", mode="x")
    logger.info("Saved %s (%d rows); all coding columns are empty for manual completion",
                out_path, len(sheet))
    return sheet
