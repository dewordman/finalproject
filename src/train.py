# src/train.py
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import require
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def build_pipeline(model_type: str, params: Dict[str, Any], seed: int) -> Pipeline:
    """Build a sklearn Pipeline with TF-IDF + classifier."""
    tfidf_params = params.get("tfidf", {})
    clf_params = params.get("clf", {})

    # Handle max_features: convert NaN/None safely
    max_features_raw = tfidf_params.get("max_features", None)
    if max_features_raw is None or (isinstance(max_features_raw, float) and np.isnan(max_features_raw)):
        max_features = None
    else:
        max_features = int(max_features_raw)

    tfidf = TfidfVectorizer(
        ngram_range=tuple(tfidf_params.get("ngram_range", [1, 2])),
        min_df=int(tfidf_params.get("min_df", 2)),
        sublinear_tf=bool(tfidf_params.get("sublinear_tf", False)),
        max_features=max_features,
    )

    if model_type == "logreg":
        # NOTE: multi_class parameter removed in scikit-learn 1.7+
        # lbfgs solver always uses multinomial by default
        clf = LogisticRegression(
            C=float(clf_params.get("C", 1.0)),
            class_weight=clf_params.get("class_weight", None),
            max_iter=int(clf_params.get("max_iter", 1000)),
            random_state=seed,
            solver="lbfgs",
        )
    elif model_type == "nb":
        clf = MultinomialNB(alpha=float(clf_params.get("alpha", 1.0)))
    elif model_type == "svm":
        clf = LinearSVC(
            C=float(clf_params.get("C", 1.0)),
            max_iter=int(clf_params.get("max_iter", 5000)),
            random_state=seed,
            dual="auto",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, macro F1, phishing recall, legit FPR from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = cm.sum()
    accuracy = np.trace(cm) / total if total > 0 else 0.0

    per_class_f1 = []
    for k in range(3):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = np.mean(per_class_f1)

    # Phishing recall (class 2)
    phishing_tp = cm[2, 2]
    phishing_fn = cm[2, :].sum() - phishing_tp
    phishing_recall = phishing_tp / (phishing_tp + phishing_fn) if (phishing_tp + phishing_fn) > 0 else 0.0

    # Legit FPR (class 0)
    legit_fp = cm[:, 0].sum() - cm[0, 0]
    legit_tn = cm.sum() - cm[0, 0] - legit_fp - (cm[0, :].sum() - cm[0, 0])
    legit_fpr = legit_fp / (legit_fp + legit_tn) if (legit_fp + legit_tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "phishing_recall": float(phishing_recall),
        "legit_fpr": float(legit_fpr),
    }


def run_grid(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run hyperparameter grid search and save results to grid_results.csv."""
    seed = require(cfg, "seed")
    results_dir = Path(require(cfg, "paths.results_dir"))
    ensure_dir(results_dir)

    grid_cfg = require(cfg, "grid")
    tfidf_grid = grid_cfg["tfidf"]
    logreg_grid = grid_cfg["logreg"]
    nb_grid = grid_cfg["nb"]
    svm_grid = grid_cfg["svm"]

    X_train = train_df["model_text"]
    y_train = train_df["class_id"].to_numpy()
    X_val = val_df["model_text"]
    y_val = val_df["class_id"].to_numpy()

    results = []

    # Logistic Regression
    logger.info("Grid search: Logistic Regression")
    for ngram_range, min_df, sublinear_tf, max_features, C, class_weight in itertools.product(
        tfidf_grid["ngram_range"],
        tfidf_grid["min_df"],
        tfidf_grid["sublinear_tf"],
        tfidf_grid["max_features"],
        logreg_grid["C"],
        logreg_grid["class_weight"],
    ):
        params = {
            "tfidf": {
                "ngram_range": ngram_range,
                "min_df": min_df,
                "sublinear_tf": sublinear_tf,
                "max_features": max_features,
            },
            "clf": {"C": C, "class_weight": class_weight, "max_iter": logreg_grid["max_iter"]},
        }
        pipe = build_pipeline("logreg", params, seed)
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics = _compute_metrics(y_val, y_pred)
            convergence_warning = False
        except Exception as e:
            logger.warning(f"LogReg fit failed: {e}")
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "phishing_recall": 0.0, "legit_fpr": 1.0}
            convergence_warning = True

        results.append({
            "model": "logreg",
            "ngram_range": str(ngram_range),
            "min_df": min_df,
            "sublinear_tf": sublinear_tf,
            "max_features": max_features if max_features is not None else "",
            "C": C,
            "class_weight": class_weight if class_weight is not None else "",
            "alpha": "",
            "convergence_warning": convergence_warning,
            **metrics,
        })

    # Naive Bayes
    logger.info("Grid search: Naive Bayes")
    for ngram_range, min_df, sublinear_tf, max_features, alpha in itertools.product(
        tfidf_grid["ngram_range"],
        tfidf_grid["min_df"],
        tfidf_grid["sublinear_tf"],
        tfidf_grid["max_features"],
        nb_grid["alpha"],
    ):
        params = {
            "tfidf": {
                "ngram_range": ngram_range,
                "min_df": min_df,
                "sublinear_tf": sublinear_tf,
                "max_features": max_features,
            },
            "clf": {"alpha": alpha},
        }
        pipe = build_pipeline("nb", params, seed)
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics = _compute_metrics(y_val, y_pred)
            convergence_warning = False
        except Exception as e:
            logger.warning(f"NB fit failed: {e}")
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "phishing_recall": 0.0, "legit_fpr": 1.0}
            convergence_warning = True

        results.append({
            "model": "nb",
            "ngram_range": str(ngram_range),
            "min_df": min_df,
            "sublinear_tf": sublinear_tf,
            "max_features": max_features if max_features is not None else "",
            "C": "",
            "class_weight": "",
            "alpha": alpha,
            "convergence_warning": convergence_warning,
            **metrics,
        })

    # SVM
    logger.info("Grid search: SVM")
    for ngram_range, min_df, sublinear_tf, max_features, C in itertools.product(
        tfidf_grid["ngram_range"],
        tfidf_grid["min_df"],
        tfidf_grid["sublinear_tf"],
        tfidf_grid["max_features"],
        svm_grid["C"],
    ):
        params = {
            "tfidf": {
                "ngram_range": ngram_range,
                "min_df": min_df,
                "sublinear_tf": sublinear_tf,
                "max_features": max_features,
            },
            "clf": {"C": C, "max_iter": svm_grid["max_iter"]},
        }
        pipe = build_pipeline("svm", params, seed)
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics = _compute_metrics(y_val, y_pred)
            convergence_warning = False
        except Exception as e:
            logger.warning(f"SVM fit failed: {e}")
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "phishing_recall": 0.0, "legit_fpr": 1.0}
            convergence_warning = True

        results.append({
            "model": "svm",
            "ngram_range": str(ngram_range),
            "min_df": min_df,
            "sublinear_tf": sublinear_tf,
            "max_features": max_features if max_features is not None else "",
            "C": C,
            "class_weight": "",
            "alpha": "",
            "convergence_warning": convergence_warning,
            **metrics,
        })

    grid_df = pd.DataFrame(results)
    grid_path = results_dir / "grid_results.csv"
    grid_df.to_csv(grid_path, index=False, encoding="utf-8")
    logger.info(f"Saved {grid_path} with {len(grid_df)} candidates")
    return grid_df


def select_best(grid_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Select the best model for each type based on selection criteria."""
    min_phishing_recall = float(require(cfg, "selection.min_phishing_recall"))
    max_legit_fpr = float(require(cfg, "selection.max_legit_fpr"))
    models_dir = Path(require(cfg, "paths.models_dir"))
    ensure_dir(models_dir)

    best = {}

    for model_type in ["logreg", "nb", "svm"]:
        candidates = grid_df[
            (grid_df["model"] == model_type)
            & (grid_df["phishing_recall"] >= min_phishing_recall)
            & (grid_df["legit_fpr"] <= max_legit_fpr)
            & (~grid_df["convergence_warning"])
        ].copy()

        if candidates.empty:
            logger.warning(
                f"No qualifying candidate for '{model_type}' with "
                f"phishing_recall >= {min_phishing_recall} and legit_fpr <= {max_legit_fpr}"
            )
            continue

        candidates = candidates.sort_values(
            by=["macro_f1", "phishing_recall"], ascending=[False, False]
        )
        best_row = candidates.iloc[0]

        # CRITICAL: Convert empty strings / NaN back to None for max_features
        max_features_raw = best_row["max_features"]
        if pd.isna(max_features_raw) or max_features_raw == "" or max_features_raw == "nan":
            max_features = None
        else:
            max_features = int(max_features_raw)

        # Convert empty strings back to None for class_weight
        class_weight_raw = best_row.get("class_weight", None)
        if pd.isna(class_weight_raw) or class_weight_raw == "" or class_weight_raw == "None":
            class_weight = None
        else:
            class_weight = class_weight_raw

        params = {
            "tfidf": {
                "ngram_range": eval(best_row["ngram_range"]),
                "min_df": int(best_row["min_df"]),
                "sublinear_tf": bool(best_row["sublinear_tf"]),
                "max_features": max_features,
            },
            "clf": {},
        }

        if model_type == "logreg":
            params["clf"] = {
                "C": float(best_row["C"]),
                "class_weight": class_weight,
                "max_iter": 1000,
            }
        elif model_type == "nb":
            params["clf"] = {"alpha": float(best_row["alpha"])}
        elif model_type == "svm":
            params["clf"] = {
                "C": float(best_row["C"]),
                "max_iter": 5000,
            }

        best[model_type] = params
        logger.info(
            f"Best {model_type}: macro_f1={best_row['macro_f1']:.4f}, "
            f"phishing_recall={best_row['phishing_recall']:.4f}, "
            f"legit_fpr={best_row['legit_fpr']:.4f}"
        )

    if not best:
        raise RuntimeError(
            f"No model qualified. Relax selection thresholds in config.yaml "
            f"(current: min_phishing_recall={min_phishing_recall}, max_legit_fpr={max_legit_fpr})"
        )

    cfg_path = models_dir / "selected_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Saved {cfg_path}")

    if not cfg_path.exists():
        raise RuntimeError(f"Failed to write {cfg_path}")

    return best


def fit_final(train_df: pd.DataFrame, best: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Fit final pipelines on full training data with best hyperparameters."""
    seed = require(cfg, "seed")
    models_dir = Path(require(cfg, "paths.models_dir"))
    ensure_dir(models_dir)

    X_train = train_df["model_text"]
    y_train = train_df["class_id"].to_numpy()

    for model_type, params in best.items():
        logger.info(f"Fitting final {model_type} on {len(train_df)} training records...")
        pipe = build_pipeline(model_type, params, seed)
        pipe.fit(X_train, y_train)

        model_path = models_dir / f"{model_type}.joblib"
        joblib.dump(pipe, model_path)
        logger.info(f"Saved {model_path}")
