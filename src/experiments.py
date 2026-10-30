# src/experiments.py
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from src.config import require
from src.evaluate import _safe_div, LABELS, LABEL_NAMES, source_only_baseline, _predict_with_model
from src.explain import explain_prediction
from src.features import build_pipeline
import joblib
from src.sealed import (complete_experiment_lock, final_test_session,
                        reserve_experiment_lock)
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

ABSENT_NOTE = "class-conditioned result; phishing absent"


# ---------------------------------------------------------------------------
# Helper to prevent MemoryError on large training sets
# ---------------------------------------------------------------------------

def _cap_max_features(params: Dict[str, Any], max_features_cap: int = 100000) -> Dict[str, Any]:
    """Cap max_features to prevent MemoryError on very large training sets.

    If the selected config has max_features=None (unlimited), replace it with
    the cap. If it already has a numeric value, keep the smaller of the two.
    """
    capped = copy.deepcopy(params)
    current = capped.get("tfidf", {}).get("max_features", None)
    if current is None:
        capped["tfidf"]["max_features"] = max_features_cap
    else:
        try:
            capped["tfidf"]["max_features"] = min(int(current), max_features_cap)
        except (TypeError, ValueError):
            capped["tfidf"]["max_features"] = max_features_cap
    return capped


# ---------------------------------------------------------------------------
# Metric computation for held-out tests
# ---------------------------------------------------------------------------

def compute_heldout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    present_class_ids: List[int],
) -> Dict[str, Any]:
    """Compute per-class metrics only for classes actually present in the held-out set.

    The confusion matrix always has 3 columns (Legitimate, Spam, Phishing) so that
    wrongly predicted Phishing is visible.  Macro F1 is **refused** (set to None)
    when any of the three classes has zero support.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class: Dict[str, Dict[str, Any]] = {}
    for k in present_class_ids:
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[LABEL_NAMES[k]] = {
            "class_id": k,
            "support": int(cm[k, :].sum()),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    all_three_present = all(cm[k, :].sum() > 0 for k in range(3))
    if all_three_present:
        macro_f1 = float(np.mean([per_class[LABEL_NAMES[k]]["f1"] for k in range(3)]))
        note = ""
    else:
        macro_f1 = None
        note = ABSENT_NOTE

    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "note": note,
        "all_classes_present": all_three_present,
    }


# ---------------------------------------------------------------------------
# Source-only baseline for held-out tests
# ---------------------------------------------------------------------------

def heldout_source_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict the majority TRAIN class for each source_dataset.

    For a held-out source that does not appear in training, fall back to the
    overall majority class in training.
    """
    majority_per_source = (
        train_df.groupby("source_dataset")["class_id"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    overall_majority = int(train_df["class_id"].value_counts().index[0])
    y_pred = test_df["source_dataset"].map(majority_per_source).fillna(overall_majority).astype(int).to_numpy()
    return y_pred


# ---------------------------------------------------------------------------
# Single source hold-out experiment
# ---------------------------------------------------------------------------

def holdout_source_test(
    heldout_source: str,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_params: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Train a fresh pipeline on non-held-out sources, test on the held-out source."""

    # Training: development records from all sources EXCEPT the held-out one
    exp_train = train_val_df[train_val_df["source_dataset"] != heldout_source].copy()
    # Evaluation: development records of the held-out source (test partition untouched)
    exp_test = test_df[test_df["source_dataset"] == heldout_source].copy()

    if exp_test.empty:
        logger.warning(f"No test records for held-out source '{heldout_source}'. Skipping.")
        return {"heldout_source": heldout_source, "skipped": True}

    # Determine which classes are actually present in the test set
    present_ids = sorted(exp_test["class_id"].unique().tolist())
    present_names = [LABEL_NAMES[k] for k in present_ids]
    logger.info(
        f"Hold-out '{heldout_source}': train={len(exp_train)}, test={len(exp_test)}, "
        f"present classes={present_names}"
    )

    # Cap max_features to prevent MemoryError on large training sets
    capped_params = _cap_max_features(selected_params, max_features_cap=100000)

    # Fit fresh pipeline
    pipe = build_pipeline("logreg", capped_params, seed)
    pipe.fit(exp_train["model_text"], exp_train["class_id"])

    y_true = exp_test["class_id"].to_numpy()
    y_pred = pipe.predict(exp_test["model_text"])

    metrics = compute_heldout_metrics(y_true, y_pred, present_ids)

    # Source-only baseline
    y_pred_src = heldout_source_baseline(exp_train, exp_test)
    cm_src = confusion_matrix(y_true, y_pred_src, labels=LABELS)
    baseline_acc = float(np.mean(y_pred_src == y_true))

    if metrics["macro_f1"] is None:
        logger.info(f"  {metrics['note']}")

    return {
        "heldout_source": heldout_source,
        "skipped": False,
        "train_size": len(exp_train),
        "test_size": len(exp_test),
        "present_classes": present_names,
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "macro_f1": metrics["macro_f1"],
        "note": metrics["note"],
        "source_baseline_accuracy": round(baseline_acc, 6),
        "source_baseline_cm": cm_src,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_source_tests(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Source-conditioned hold-out experiments (Chapter Three, Experiment 2a/2b).

    DEVELOPMENT DATA ONLY: both the training sources and the held-out source are
    drawn from the training + validation partitions (``records_development.pkl``).
    The sealed test partition is never loaded here. Fresh pipelines are fitted for
    every experiment; the selected models are not touched.
    """
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    models_dir = Path(require(cfg, "paths.models_dir"))
    seed = require(cfg, "seed")
    ensure_dir(results_dir)
    summary_path = results_dir / "source_tests_summary.csv"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists; choose a fresh results directory")

    sel_path = models_dir / "selected_config.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"Run --stage select first. Missing {sel_path}")
    with sel_path.open("r", encoding="utf-8") as f:
        logreg_params = json.load(f)["logreg"]

    development = pd.read_pickle(processed_dir / "records_development.pkl")
    if not development["partition_id"].isin(["train", "validation"]).all():
        raise AssertionError("Source tests may use training and validation rows only")

    summary_rows: List[Dict[str, Any]] = []
    for label, source, name in (("2a", "Enron-Spam", "holdout_enron"),
                                ("2b", "SpamAssassin", "holdout_spamassassin")):
        logger.info("Experiment %s: hold-out %s (development data only)", label, source)
        result = holdout_source_test(source, development, development, logreg_params, seed)
        if not result.get("skipped"):
            _save_holdout_results(result, name, results_dir)
            summary_rows.append(_summary_row(result))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8", mode="x")
    logger.info("Saved %s", summary_path)
    return summary_df


def _save_holdout_results(res: Dict[str, Any], name: str, results_dir: Path) -> None:
    """Save per-class metrics and confusion matrix for a hold-out experiment."""
    rows = []
    for cname, m in res["per_class"].items():
        rows.append({"class": cname, **m})
    pd.DataFrame(rows).to_csv(results_dir / f"source_test_{name}.csv", index=False, encoding="utf-8")

    cm = res["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.index.name = "true\\predicted"
    cm_df.to_csv(results_dir / f"confusion_matrix_{name}.csv", encoding="utf-8")

    cm_src = res["source_baseline_cm"]
    cm_src_df = pd.DataFrame(cm_src, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_src_df.index.name = "true\\predicted"
    cm_src_df.to_csv(results_dir / f"confusion_matrix_{name}_baseline.csv", encoding="utf-8")

    logger.info(f"Saved results/source_test_{name}.csv")


def _summary_row(res: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment": f"holdout_{res['heldout_source']}",
        "heldout_source": res["heldout_source"],
        "train_size": res["train_size"],
        "test_size": res["test_size"],
        "present_classes": ", ".join(res["present_classes"]),
        "macro_f1": res["macro_f1"],
        "note": res["note"],
        "source_baseline_accuracy": res["source_baseline_accuracy"],
    }
    for cname, m in res["per_class"].items():
        row[f"recall_{cname}"] = m["recall"]
        row[f"precision_{cname}"] = m["precision"]
        row[f"f1_{cname}"] = m["f1"]
    return row


# ---------------------------------------------------------------------------
# Temporal generalisation on the external Nazario files (once-only, locked)
# ---------------------------------------------------------------------------

TEMPORAL_NOTE = ("temporal generalisation within one phishing source (one person's inbox); "
                 "legitimate and spam absent")


def _temporal_tables(external: pd.DataFrame, predicted: np.ndarray,
                     confidence: np.ndarray) -> pd.DataFrame:
    """Overall and per-year phishing recall, with where the misses went."""
    frame = pd.DataFrame({"era_year": external["era_year"].to_numpy(),
                          "predicted": predicted, "confidence": confidence})

    def row(label: str, part: pd.DataFrame) -> Dict[str, Any]:
        return {"era_year": label, "n": len(part),
                "phishing_recall": round(float((part["predicted"] == 2).mean()), 6),
                "mean_confidence": round(float(part["confidence"].mean()), 6),
                "predicted_spam": int((part["predicted"] == 1).sum()),
                "predicted_legitimate": int((part["predicted"] == 0).sum()),
                "note": TEMPORAL_NOTE}

    rows = [row("overall", frame)]
    for year, part in frame.groupby("era_year", sort=True):
        rows.append(row(str(int(year)), part))
    return pd.DataFrame(rows)


def _missed_terms(pipeline: Any, texts: List[str], top_n: int = 10) -> pd.DataFrame:
    """Terms that pushed missed messages toward the (wrong) predicted class."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for text in texts:
        for item in explain_prediction(pipeline, text, top_n=10)["supporting"]:
            totals[item["term"]] = totals.get(item["term"], 0.0) + float(item["contribution"])
            counts[item["term"]] = counts.get(item["term"], 0) + 1
    ranked = sorted(totals, key=lambda term: totals[term], reverse=True)[:top_n]
    return pd.DataFrame({"term": ranked,
                         "total_contribution": [round(totals[t], 6) for t in ranked],
                         "n_missed_messages": [counts[t] for t in ranked]})


def run_nazario_temporal(cfg: Dict[str, Any], cfg_path: Path, force: bool = False,
                         reason: Optional[str] = None) -> pd.DataFrame:
    """Score the already fitted models on the external (later-era) Nazario files.

    * No refitting: the selected Logistic Regression and its calibration choice
      are loaded exactly as saved.
    * Once only: protected by ``<locks_dir>/nazario_temporal_lock.json``.
    * Never computes a three-class macro score (legitimate and spam are absent).
    """
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    models_dir = Path(require(cfg, "paths.models_dir"))
    locks_dir = Path(require(cfg, "paths.locks_dir"))
    run_id = str(cfg.get("_run_id") or uuid.uuid4().hex)
    outputs = [results_dir / name for name in (
        "source_test_nazario_temporal.csv", "confusion_matrix_nazario_temporal.csv",
        "nazario_temporal_missed_terms.csv", "nazario_temporal_predictions.csv")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Temporal outputs already exist; choose a fresh results directory")
    ensure_dir(results_dir)
    with final_test_session(locks_dir):
        digest = reserve_experiment_lock("nazario_temporal", results_dir, cfg_path, force,
                                         reason, locks_dir=locks_dir, run_id=run_id)
        try:
            external = pd.read_pickle(processed_dir / "records_external.pkl")
            external = external.loc[external["parse_status"].eq("ok")].reset_index(drop=True)
            if external.empty:
                raise ValueError("No usable external Nazario records")
            if not external["partition_id"].eq("external_temporal").all():
                raise AssertionError("External records must carry partition_id external_temporal")
            if not external["class_id"].eq(2).all():
                raise AssertionError("External Nazario records must all be Phishing")
            predicted, assessed, _ = _predict_with_model("logreg", models_dir, external["model_text"])
            predicted = np.asarray(predicted, dtype=int)
            confidence = np.asarray(assessed)[np.arange(len(predicted)), predicted]
            table = _temporal_tables(external, predicted, confidence)
            matrix = confusion_matrix(np.full(len(predicted), 2), predicted, labels=LABELS)
            matrix_df = pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES)
            matrix_df.index.name = "true\\predicted"
            pipeline = joblib.load(models_dir / "logreg.joblib")
            missed = external.loc[predicted != 2, "model_text"].tolist()
            terms = _missed_terms(pipeline, missed)
            per_message = pd.DataFrame({"record_id": external["record_id"],
                                        "era_year": external["era_year"],
                                        "predicted_class": predicted, "confidence": confidence})
            table.to_csv(outputs[0], index=False, encoding="utf-8", mode="x")
            matrix_df.to_csv(outputs[1], encoding="utf-8", mode="x")
            terms.to_csv(outputs[2], index=False, encoding="utf-8", mode="x")
            per_message.to_csv(outputs[3], index=False, encoding="utf-8", mode="x")
            complete_experiment_lock("nazario_temporal", locks_dir, digest, run_id)
            logger.info("Saved temporal results; overall phishing recall = %.4f",
                        float(table.loc[0, "phishing_recall"]))
            return table
        except Exception:
            complete_experiment_lock("nazario_temporal", locks_dir, digest, run_id, status="failed")
            raise
