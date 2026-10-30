# src/calibrate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from src.config import require
from src.features import build_pipeline
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

LABELS = [0, 1, 2]


class SigmoidCalibratedClassifier:
    """Platt-scaled (sigmoid) multi-class calibration via one-vs-rest sigmoids on OOF decision scores."""

    def __init__(self, base_pipeline_params: Dict[str, Any], n_folds: int = 5, seed: int = 42):
        self.base_pipeline_params = base_pipeline_params
        self.n_folds = n_folds
        self.seed = seed
        self.calibrators_: List[LogisticRegression] = []
        self.base_pipeline_ = None

    def _make_base(self):
        return build_pipeline("logreg", self.base_pipeline_params, self.seed)

    def fit(self, X_text, y, groups=None):
        # Cap max_features to prevent MemoryError
        capped_params = copy.deepcopy(self.base_pipeline_params)
        current_max = capped_params.get("tfidf", {}).get("max_features", None)
        if current_max is None:
            capped_params["tfidf"]["max_features"] = 100000
        else:
            try:
                capped_params["tfidf"]["max_features"] = min(int(current_max), 100000)
            except (TypeError, ValueError):
                capped_params["tfidf"]["max_features"] = 100000
        
        self.base_pipeline_params = capped_params
        
        if groups is None:
            groups = np.arange(len(X_text))
        gkf = GroupKFold(n_splits=self.n_folds)
        oof_scores = np.zeros((len(X_text), len(LABELS)))

        for train_idx, test_idx in gkf.split(X_text, y, groups=groups):
            fold_pipe = self._make_base()
            fold_pipe.fit(X_text.iloc[train_idx], y.iloc[train_idx])
            scores = fold_pipe.decision_function(X_text.iloc[test_idx])
            oof_scores[test_idx] = scores

        self.calibrators_ = []
        for k in range(len(LABELS)):
            binary_y = (np.asarray(y) == LABELS[k]).astype(int)
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            cal.fit(oof_scores[:, [k]], binary_y)
            self.calibrators_.append(cal)

        self.base_pipeline_ = self._make_base()
        self.base_pipeline_.fit(X_text, y)
        return self

    def predict_proba(self, X_text):
        scores = self.base_pipeline_.decision_function(X_text)
        probs = np.zeros_like(scores)
        for k, cal in enumerate(self.calibrators_):
            probs[:, k] = cal.predict_proba(scores[:, [k]])[:, 1]
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        return probs

    def predict(self, X_text):
        probs = self.predict_proba(X_text)
        idx = np.argmax(probs, axis=1)
        return np.array([LABELS[i] for i in idx])


def reliability_table(y_true, proba, n_bins=10):
    pred_class = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred_class]
    correct = (pred_class == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})", "mean_confidence": float("nan"),
                         "accuracy": float("nan"), "count": 0})
        else:
            rows.append({"bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                         "mean_confidence": float(conf[mask].mean()),
                         "accuracy": float(correct[mask].mean()),
                         "count": count})
    ece = 0.0
    total = len(conf)
    for r in rows:
        if r["count"] > 0 and not np.isnan(r["mean_confidence"]):
            ece += abs(r["accuracy"] - r["mean_confidence"]) * (r["count"] / total)
    return pd.DataFrame(rows), float(ece)


def multiclass_brier(y_true, proba):
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(proba)
    for i, lbl in enumerate(LABELS):
        onehot[:, i] = (y_true == lbl).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compare_calibration(raw_probas, sig_probas, y_true, tag, out_path: Path):
    ll_raw = log_loss(y_true, raw_probas, labels=LABELS)
    ll_sig = log_loss(y_true, sig_probas, labels=LABELS)
    br_raw = multiclass_brier(y_true, raw_probas)
    br_sig = multiclass_brier(y_true, sig_probas)
    _, ece_raw = reliability_table(y_true, raw_probas)
    _, ece_sig = reliability_table(y_true, sig_probas)
    return {
        "split": tag,
        "log_loss_raw": round(ll_raw, 6),
        "log_loss_sigmoid": round(ll_sig, 6),
        "brier_raw": round(br_raw, 6),
        "brier_sigmoid": round(br_sig, 6),
        "ece_raw": round(ece_raw, 6),
        "ece_sigmoid": round(ece_sig, 6),
    }


def plot_reliability(reliab_df: pd.DataFrame, title: str, out_png: Path):
    df = reliab_df.dropna(subset=["mean_confidence"]).copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    width = 0.09
    centers = [(float(b.split(",")[0][1:]) + float(b.split(",")[1][:-1])) / 2 for b in df["bin"]]
    ax.bar(centers, df["accuracy"], width=width, alpha=0.6, color="steelblue", label="Observed accuracy")
    ax.scatter(df["mean_confidence"], df["accuracy"], color="darkred", zorder=5, label="Bin points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_calibration(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = require(cfg, "seed")
    models_dir = Path(require(cfg, "paths.models_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    ensure_dir(models_dir)
    ensure_dir(results_dir)

    with (models_dir / "selected_config.json").open("r", encoding="utf-8") as f:
        selected = json.load(f)
    best_logreg = selected["logreg"]

    n_folds = require(cfg, "calibration.n_folds")

    X_tr = train_df["model_text"]
    y_tr = train_df["class_id"]
    groups_tr = train_df["duplicate_group_id"]

    logger.info(f"Fitting SigmoidCalibratedClassifier with {n_folds} grouped folds...")
    calib = SigmoidCalibratedClassifier(best_logreg, n_folds=n_folds, seed=seed)
    calib.fit(X_tr, y_tr, groups=groups_tr)

    # Out-of-fold style comparison on TRAIN: use the base pipeline's own predict_proba vs sigmoid
    raw_train_proba = calib.base_pipeline_.predict_proba(X_tr)
    sig_train_proba = calib.predict_proba(X_tr)

    X_va = val_df["model_text"]
    y_va = val_df["class_id"]
    raw_val_proba = calib.base_pipeline_.predict_proba(X_va)
    sig_val_proba = calib.predict_proba(X_va)

    rows = [
        compare_calibration(raw_train_proba, sig_train_proba, y_tr, "train_oof_proxy", results_dir),
        compare_calibration(raw_val_proba, sig_val_proba, y_va, "validation", results_dir),
    ]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(results_dir / "calibration_comparison.csv", index=False, encoding="utf-8")
    logger.info("Saved results/calibration_comparison.csv")

    rel_raw_va, _ = reliability_table(y_va, raw_val_proba)
    rel_sig_va, _ = reliability_table(y_va, sig_val_proba)
    rel_raw_va.to_csv(results_dir / "reliability_raw.csv", index=False, encoding="utf-8")
    rel_sig_va.to_csv(results_dir / "reliability_sigmoid.csv", index=False, encoding="utf-8")
    logger.info("Saved reliability tables.")

    plot_reliability(rel_raw_va, "Reliability Diagram (raw)", results_dir / "reliability_raw.png")
    plot_reliability(rel_sig_va, "Reliability Diagram (sigmoid)", results_dir / "reliability_sigmoid.png")
    logger.info("Saved reliability plots.")

    val_row = rows[1]
    use_sigmoid = (val_row["log_loss_sigmoid"] < val_row["log_loss_raw"]
                   and val_row["brier_sigmoid"] < val_row["brier_raw"])
    choice = "sigmoid" if use_sigmoid else "raw"

    payload = {
        "choice": choice,
        "rule": "sigmoid only if it lowers BOTH log loss and Brier on validation",
        "validation_metrics": {
            "log_loss_raw": val_row["log_loss_raw"],
            "log_loss_sigmoid": val_row["log_loss_sigmoid"],
            "brier_raw": val_row["brier_raw"],
            "brier_sigmoid": val_row["brier_sigmoid"],
            "ece_raw": val_row["ece_raw"],
            "ece_sigmoid": val_row["ece_sigmoid"],
        },
        "n_folds": n_folds,
        "seed": seed,
    }
    with (models_dir / "calibration_choice.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Calibration choice: {choice}")

    joblib.dump(calib, models_dir / "logreg_calibrated.joblib")
    logger.info(f"Saved {models_dir / 'logreg_calibrated.joblib'}")

    return payload