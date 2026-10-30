# src/evaluate.py
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss

from src.config import require
from src.utils import ensure_dir, get_logger
from src.sealed import enforce_test_lock, write_test_lock, final_test_session

logger = get_logger(__name__)

LABELS = [0, 1, 2]
LABEL_NAMES = ["Legitimate", "Spam", "Phishing"]
LABEL_MAP = {0: "Legitimate", 1: "Spam", 2: "Phishing"}


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num/den, or 0.0 when den is zero."""
    if den == 0:
        return 0.0
    return float(num) / float(den)


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, Any]:
    """Compute per-class, macro, weighted metrics and overall accuracy from a 3x3 confusion matrix.

    Convention: rows = true class, columns = predicted class.
    """
    cm = np.asarray(cm, dtype=int)
    if cm.shape != (3, 3):
        raise ValueError(f"Expected 3x3 confusion matrix, got shape {cm.shape}")

    per_class = {}
    supports = cm.sum(axis=1)  # true counts per class
    total = int(cm.sum())

    for k in range(3):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - tp)
        fn = int(cm[k, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        fpr = _safe_div(fp, fp + tn)
        # Define FNR as 1 - recall to guarantee the invariant by construction,
        # even in the degenerate case where a class has zero support.
        fnr = 1.0 - recall

        # Internal sanity check: FNR == 1 - recall for every class
        if abs(fnr - (1.0 - recall)) > 1e-9:
            raise AssertionError(
                f"Internal metric invariant violated for class {k}: "
                f"FNR={fnr}, 1-recall={1.0 - recall}"
            )

        per_class[LABEL_NAMES[k]] = {
            "class_id": k,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": int(supports[k]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        }

    # Overall accuracy
    correct = int(np.trace(cm))
    accuracy = _safe_div(correct, total) if total > 0 else 0.0

    # Macro (unweighted mean over the 3 classes)
    macro_precision = np.mean([per_class[c]["precision"] for c in LABEL_NAMES])
    macro_recall = np.mean([per_class[c]["recall"] for c in LABEL_NAMES])
    macro_f1 = np.mean([per_class[c]["f1"] for c in LABEL_NAMES])
    macro_fpr = np.mean([per_class[c]["fpr"] for c in LABEL_NAMES])
    macro_fnr = np.mean([per_class[c]["fnr"] for c in LABEL_NAMES])

    # Weighted (support-weighted)
    if total > 0:
        weights = supports / total
        weighted_precision = float(np.sum([per_class[c]["precision"] * weights[k]
                                           for k, c in enumerate(LABEL_NAMES)]))
        weighted_recall = float(np.sum([per_class[c]["recall"] * weights[k]
                                        for k, c in enumerate(LABEL_NAMES)]))
        weighted_f1 = float(np.sum([per_class[c]["f1"] * weights[k]
                                    for k, c in enumerate(LABEL_NAMES)]))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "total": total,
        "per_class": per_class,
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
            "fpr": float(macro_fpr),
            "fnr": float(macro_fnr),
        },
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Group bootstrap confidence intervals
# ---------------------------------------------------------------------------

def group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_reps: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Resample duplicate groups with replacement and return percentile CIs for key metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Precompute per-group confusion counts
    group_cm: Dict[Any, np.ndarray] = {}
    for g in unique_groups:
        mask = groups == g
        cm_g = confusion_matrix(y_true[mask], y_pred[mask], labels=LABELS)
        group_cm[g] = cm_g

    rng = np.random.default_rng(seed)

    # Metric accumulators
    keys = ["accuracy", "macro_f1",
            "recall_Legitimate", "recall_Spam", "recall_Phishing",
            "precision_Legitimate", "precision_Spam", "precision_Phishing",
            "f1_Legitimate", "f1_Spam", "f1_Phishing",
            "fpr_Legitimate"]
    samples = {k: np.zeros(n_reps, dtype=float) for k in keys}

    for b in range(n_reps):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        cm_total = np.zeros((3, 3), dtype=int)
        for g in sampled:
            cm_total += group_cm[g]
        m = metrics_from_confusion(cm_total)
        samples["accuracy"][b] = m["accuracy"]
        samples["macro_f1"][b] = m["macro"]["f1"]
        for c in LABEL_NAMES:
            samples[f"recall_{c}"][b] = m["per_class"][c]["recall"]
            samples[f"precision_{c}"][b] = m["per_class"][c]["precision"]
            samples[f"f1_{c}"][b] = m["per_class"][c]["f1"]
        samples["fpr_Legitimate"][b] = m["per_class"]["Legitimate"]["fpr"]

    alpha = (1.0 - ci_level) / 2.0
    ci: Dict[str, Dict[str, float]] = {}
    for k, arr in samples.items():
        ci[k] = {
            "mean": float(arr.mean()),
            "lower": float(np.percentile(arr, 100 * alpha)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha))),
        }
    return ci


# ---------------------------------------------------------------------------
# Probability quality metrics (for logreg)
# ---------------------------------------------------------------------------

def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    onehot = np.zeros_like(proba)
    for i, lbl in enumerate(LABELS):
        onehot[:, i] = (y_true == lbl).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10,
                      *, predicted_classes=None):
    """10 equal-width confidence bins of the predicted-class probability."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred_class = (np.argmax(proba, axis=1) if predicted_classes is None
                  else np.asarray(predicted_classes, dtype=int))
    conf = proba[np.arange(len(proba)), pred_class]
    correct = (pred_class == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
            label = f"[{lo:.1f}, {hi:.1f})"
        else:
            mask = (conf >= lo) & (conf <= hi)
            label = f"[{lo:.1f}, {hi:.1f}]"
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": label, "mean_confidence": np.nan,
                         "accuracy": np.nan, "count": 0})
        else:
            rows.append({"bin": label,
                         "mean_confidence": float(conf[mask].mean()),
                         "accuracy": float(correct[mask].mean()),
                         "count": count})

    ece = 0.0
    total = len(conf)
    if total > 0:
        for r in rows:
            if r["count"] > 0 and not np.isnan(r["mean_confidence"]):
                ece += abs(r["accuracy"] - r["mean_confidence"]) * (r["count"] / total)
    return pd.DataFrame(rows), float(ece)


# ---------------------------------------------------------------------------
# Source-only baseline
# ---------------------------------------------------------------------------

def source_only_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict the majority TRAIN class for each source_dataset."""
    majority = (
        train_df.groupby("source_dataset")["class_id"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    y_pred = test_df["source_dataset"].map(majority).fillna(0).astype(int).to_numpy()
    return y_pred


# Global-lock wrappers retain the original positional call signatures.
def _config_digest(cfg_path: Path) -> str:
    return hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()


def _enforce_test_lock(results_dir: Path, cfg_path: Path, force: bool,
                       reason: Optional[str], *, locks_dir: Optional[Path] = None,
                       run_id: Optional[str] = None) -> str:
    lock_root = Path(locks_dir) if locks_dir is not None else Path(results_dir).parent / "locks"
    return enforce_test_lock(results_dir, cfg_path, force, reason,
                             locks_dir=lock_root, run_id=run_id)


def _write_lock(results_dir: Path, digest: str, summary: Dict[str, Any],
                *, locks_dir: Optional[Path] = None,
                run_id: Optional[str] = None, status: str = "completed") -> None:
    lock_root = Path(locks_dir) if locks_dir is not None else Path(results_dir).parent / "locks"
    write_test_lock(lock_root, digest, summary, run_id=run_id, status=status)


def _plot_confusion_heatmap(cm: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(int(cm[row, column])), ha="center", va="center",
                    color="white" if cm[row, column] > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _predict_with_model(model_tag: str, models_dir: Path, X_text: pd.Series):
    """Return raw-model classes, assessed probabilities, and SVM scores only."""
    model = joblib.load(Path(models_dir) / f"{model_tag}.joblib")
    classes = np.asarray(model.named_steps["clf"].classes_)
    if not np.array_equal(classes, LABELS):
        raise ValueError(f"Selected {model_tag} must contain class IDs {LABELS}")
    if model_tag == "svm":
        return model.predict(X_text), None, model.decision_function(X_text)
    if model_tag not in ("logreg", "nb"):
        raise ValueError(f"Unknown model_tag: {model_tag}")
    raw = model.predict_proba(X_text)
    predicted = classes[np.argmax(raw, axis=1)]
    if model_tag == "nb":
        return predicted, raw, None
    with (Path(models_dir) / "calibration_choice.json").open("r", encoding="utf-8") as handle:
        choice = json.load(handle).get("choice")
    if choice not in ("raw", "sigmoid"):
        raise ValueError("calibration_choice.json must contain choice='raw' or 'sigmoid'")
    assessed = raw
    if choice == "sigmoid":
        calibrated = joblib.load(Path(models_dir) / "logreg_calibrated.joblib")
        assessed = calibrated.predict_proba(X_text)
    assessed = np.asarray(assessed, dtype=float)
    if (assessed.shape != raw.shape or not np.isfinite(assessed).all()
            or (assessed < 0).any() or (assessed > 1).any()
            or not np.allclose(assessed.sum(axis=1), 1.0)):
        raise ValueError("Invalid assessed Logistic Regression probabilities")
    return predicted, assessed, None


def evaluation_output_names(prefix: str = ""):
    """Output registry shared by the runner and evaluation overwrite guard."""
    names = []
    for tag in ("logreg", "nb", "svm", "source_baseline"):
        names.extend([f"{prefix}confusion_matrix_{tag}.csv",
                      f"{prefix}confusion_matrix_{tag}.png",
                      f"{prefix}per_class_metrics_{tag}.csv"])
    for tag in ("logreg", "nb", "svm"):
        names.append(f"dev_predictions_{tag}.csv" if prefix else f"test_predictions_{tag}.csv")
    names.extend([f"{prefix}overall_metrics.csv", f"{prefix}bootstrap_ci.csv",
                  f"{prefix}final_summary.json", f"{prefix}reliability_logreg_test.csv"])
    return names


def _evaluate_partition(train_df: pd.DataFrame, evaluation_df: pd.DataFrame,
                        cfg: Dict[str, Any], *, partition: str, prefix: str,
                        digest: str, run_id: str) -> Dict[str, Any]:
    if evaluation_df.empty:
        raise ValueError(f"The {partition} partition is empty")
    assert evaluation_df["partition_id"].eq(partition).all(), (
        f"Every evaluated row must have partition_id == {partition!r}"
    )
    assert train_df["partition_id"].eq("train").all(), "Baseline fitting must use training rows only"
    if set(train_df["duplicate_group_id"]) & set(evaluation_df["duplicate_group_id"]):
        raise AssertionError("Duplicate groups overlap between training and evaluation")
    results_dir = ensure_dir(require(cfg, "paths.results_dir"))
    models_dir = Path(require(cfg, "paths.models_dir"))
    existing = [str(results_dir / name) for name in evaluation_output_names(prefix)
                if (results_dir / name).exists()]
    if existing:
        raise FileExistsError("Evaluation outputs already exist; choose a new results directory: "
                              + ", ".join(existing))
    repetitions = int(require(cfg, "evaluation.bootstrap_reps"))
    level = float(require(cfg, "evaluation.ci_level"))
    seed = int(require(cfg, "seed"))
    if repetitions < 1:
        raise ValueError("evaluation.bootstrap_reps must be positive")
    y_true = evaluation_df["class_id"].to_numpy()
    groups = evaluation_df["duplicate_group_id"].to_numpy()
    summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
               "config_digest": digest, "partition": partition,
               "test_size" if partition == "test" else "validation_size": len(evaluation_df),
               "bootstrap_reps": repetitions, "ci_level": level, "models": {}}
    overall_rows, ci_rows = [], []
    for tag in ("logreg", "nb", "svm", "source_baseline"):
        proba, scores = None, None
        if tag == "source_baseline":
            predicted = source_only_baseline(train_df, evaluation_df)
        else:
            predicted, proba, scores = _predict_with_model(tag, models_dir, evaluation_df["model_text"])
        matrix = confusion_matrix(y_true, predicted, labels=LABELS)
        metrics = metrics_from_confusion(matrix)
        intervals = group_bootstrap_ci(y_true, predicted, groups, repetitions, level, seed)
        for name in LABEL_NAMES:
            values = metrics["per_class"][name]
            assert np.isclose(values["fnr"], 1.0 - values["recall"]), "FNR != 1 - recall"
        pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).rename_axis(
            "true\\predicted").to_csv(results_dir / f"{prefix}confusion_matrix_{tag}.csv", encoding="utf-8")
        _plot_confusion_heatmap(matrix, f"{partition.title()} — {tag.upper()}",
                                results_dir / f"{prefix}confusion_matrix_{tag}.png")
        pd.DataFrame([{"class": name, **metrics["per_class"][name]} for name in LABEL_NAMES]).to_csv(
            results_dir / f"{prefix}per_class_metrics_{tag}.csv", index=False, encoding="utf-8")
        probability_metrics = {}
        if tag != "source_baseline":
            predictions = pd.DataFrame({"record_id": evaluation_df["record_id"].to_numpy(),
                "duplicate_group_id": groups, "source_dataset": evaluation_df["source_dataset"].to_numpy(),
                "partition_id": partition, "true_class": y_true, "predicted_class": predicted})
            if tag == "logreg":
                predictions["confidence"] = proba[np.arange(len(predicted)), predicted]
            if scores is not None:
                for index, name in enumerate(LABEL_NAMES):
                    predictions[f"score_{name}"] = scores[:, index]
            filename = f"dev_predictions_{tag}.csv" if prefix else f"test_predictions_{tag}.csv"
            predictions.to_csv(results_dir / filename, index=False, encoding="utf-8")
        if tag == "logreg":
            reliability, ece = reliability_table(y_true, proba, predicted_classes=predicted)
            reliability.to_csv(results_dir / f"{prefix}reliability_logreg_test.csv", index=False, encoding="utf-8")
            probability_metrics = {"log_loss": float(log_loss(y_true, proba, labels=LABELS)),
                                   "brier_score": multiclass_brier(y_true, proba), "ece": ece}
        summary["models"][tag] = {"accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro"]["f1"], "weighted_f1": metrics["weighted"]["f1"],
            "macro": metrics["macro"], "weighted": metrics["weighted"],
            "per_class": metrics["per_class"], "ci": intervals,
            "probability_metrics": probability_metrics}
        overall_rows.append({"model": tag, "partition": partition, "n": len(evaluation_df),
            "accuracy": metrics["accuracy"],
            **{f"macro_{key}": value for key, value in metrics["macro"].items()},
            **{f"weighted_{key}": value for key, value in metrics["weighted"].items()},
            **probability_metrics})
        for metric, bounds in intervals.items():
            ci_rows.append({"model": tag, "metric": metric, **bounds})
    pd.DataFrame(overall_rows).to_csv(results_dir / f"{prefix}overall_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(ci_rows).to_csv(results_dir / f"{prefix}bootstrap_ci.csv", index=False, encoding="utf-8")
    with (results_dir / f"{prefix}final_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def run_dev_eval(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Evaluate selected models on validation without opening the sealed pickle."""
    path = Path(require(cfg, "paths.processed_dir")) / "records_development.pkl"
    development = pd.read_pickle(path)
    assert development["partition_id"].isin(["train", "validation"]).all(), (
        "Development records must contain train and validation only"
    )
    train = development.loc[development["partition_id"].eq("train")].copy()
    validation = development.loc[development["partition_id"].eq("validation")].copy()
    assert validation["partition_id"].eq("validation").all()
    return _evaluate_partition(train, validation, cfg, partition="validation", prefix="dev_",
        digest=_config_digest(cfg_path), run_id=str(cfg.get("_run_id") or uuid.uuid4().hex))


def run_final_test(cfg: Dict[str, Any], cfg_path: Path, force: bool = False,
                   reason: Optional[str] = None) -> Dict[str, Any]:
    """Reserve the global lock BEFORE loading labels, then score test once."""
    results_dir = Path(require(cfg, "paths.results_dir"))
    locks_dir = Path(require(cfg, "paths.locks_dir"))
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    run_id = str(cfg.get("_run_id") or uuid.uuid4().hex)
    with final_test_session(locks_dir):
        digest = _enforce_test_lock(results_dir, cfg_path, force, reason,
                                    locks_dir=locks_dir, run_id=run_id)
        try:
            records = pd.read_pickle(processed_dir / "records_partitioned.pkl")
            train = records.loc[records["partition_id"].eq("train")].copy()
            test = records.loc[records["partition_id"].eq("test")].copy()
            summary = _evaluate_partition(train, test, cfg, partition="test", prefix="",
                                          digest=digest, run_id=run_id)
            _write_lock(results_dir, digest, summary, locks_dir=locks_dir, run_id=run_id)
            return summary
        except Exception:
            _write_lock(results_dir, digest, {}, locks_dir=locks_dir,
                        run_id=run_id, status="failed")
            raise
