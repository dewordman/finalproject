# tests/test_stage8.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    metrics_from_confusion,
    group_bootstrap_ci,
    multiclass_brier,
    reliability_table,
    source_only_baseline,
    _enforce_test_lock,
    _config_digest,
)


# ---------------------------------------------------------------------------
# Hand-computed 3x3 confusion matrix
# ---------------------------------------------------------------------------
# True\Pred  0  1  2
#     0      8  1  1   (support=10)
#     1      2  6  2   (support=10)
#     2      1  3  6   (support=10)
# Total = 30, correct = 20, accuracy = 2/3

HAND_CM = np.array([
    [8, 1, 1],
    [2, 6, 2],
    [1, 3, 6],
], dtype=int)


def test_metrics_from_confusion_per_class():
    m = metrics_from_confusion(HAND_CM)

    # Class 0 (Legitimate): TP=8, FP=3, FN=2, TN=17
    c0 = m["per_class"]["Legitimate"]
    assert c0["tp"] == 8 and c0["fp"] == 3 and c0["fn"] == 2 and c0["tn"] == 17
    assert c0["precision"] == pytest.approx(8 / 11)
    assert c0["recall"] == pytest.approx(8 / 10)
    assert c0["f1"] == pytest.approx(2 * (8/11) * (8/10) / ((8/11) + (8/10)))
    assert c0["fpr"] == pytest.approx(3 / 20)
    assert c0["fnr"] == pytest.approx(2 / 10)

    # Class 1 (Spam): TP=6, FP=4, FN=4, TN=16
    c1 = m["per_class"]["Spam"]
    assert c1["tp"] == 6 and c1["fp"] == 4 and c1["fn"] == 4 and c1["tn"] == 16
    assert c1["precision"] == pytest.approx(6 / 10)
    assert c1["recall"] == pytest.approx(6 / 10)
    assert c1["f1"] == pytest.approx(0.6)
    assert c1["fpr"] == pytest.approx(4 / 20)
    assert c1["fnr"] == pytest.approx(4 / 10)

    # Class 2 (Phishing): TP=6, FP=3, FN=4, TN=17
    c2 = m["per_class"]["Phishing"]
    assert c2["tp"] == 6 and c2["fp"] == 3 and c2["fn"] == 4 and c2["tn"] == 17
    assert c2["precision"] == pytest.approx(6 / 9)
    assert c2["recall"] == pytest.approx(6 / 10)
    assert c2["fpr"] == pytest.approx(3 / 20)
    assert c2["fnr"] == pytest.approx(4 / 10)


def test_metrics_from_confusion_overall_and_averages():
    m = metrics_from_confusion(HAND_CM)
    assert m["accuracy"] == pytest.approx(20 / 30)
    assert m["total"] == 30

    # Macro = unweighted mean
    macro_f1 = np.mean([
        m["per_class"]["Legitimate"]["f1"],
        m["per_class"]["Spam"]["f1"],
        m["per_class"]["Phishing"]["f1"],
    ])
    assert m["macro"]["f1"] == pytest.approx(macro_f1)

    # Weighted = support-weighted (all supports equal here, so same as macro)
    assert m["weighted"]["f1"] == pytest.approx(macro_f1)


def test_fnr_equals_one_minus_recall_for_every_class():
    """Internal invariant: FNR == 1 - recall for every class."""
    m = metrics_from_confusion(HAND_CM)
    for name in ["Legitimate", "Spam", "Phishing"]:
        c = m["per_class"][name]
        assert c["fnr"] == pytest.approx(1.0 - c["recall"])


def test_safe_division_returns_zero_when_denominator_zero():
    """Construct a confusion matrix with a class having zero support to exercise safe division."""
    cm = np.array([
        [10, 0, 0],
        [0, 0, 0],   # class 1 never appears
        [0, 0, 5],
    ], dtype=int)
    m = metrics_from_confusion(cm)
    # Class 1: TP=FP=FN=TN-degenerate; all metrics should be 0.0, no ZeroDivisionError
    c1 = m["per_class"]["Spam"]
    assert c1["precision"] == 0.0
    assert c1["recall"] == 0.0
    assert c1["f1"] == 0.0
    assert c1["fnr"] == pytest.approx(1.0 - c1["recall"])


def test_group_bootstrap_ci_returns_finite_intervals():
    """Bootstrap CIs should be finite and satisfy lower <= mean <= upper."""
    rng = np.random.default_rng(0)
    n = 120
    y_true = rng.integers(0, 3, size=n)
    # Predictor is right ~60% of the time
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.4
    y_pred[flip] = (y_pred[flip] + 1) % 3
    groups = np.repeat(np.arange(30), 4)  # 30 groups of 4

    ci = group_bootstrap_ci(y_true, y_pred, groups, n_reps=200, ci_level=0.95, seed=42)

    for key in ["accuracy", "macro_f1", "recall_Legitimate", "fpr_Legitimate"]:
        assert key in ci
        assert np.isfinite(ci[key]["lower"])
        assert np.isfinite(ci[key]["mean"])
        assert np.isfinite(ci[key]["upper"])
        assert ci[key]["lower"] <= ci[key]["mean"] <= ci[key]["upper"] + 1e-9


def test_multiclass_brier_perfect_and_worst():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.ones((3, 3)) / 3.0
    assert multiclass_brier(y, perfect) == pytest.approx(0.0)
    assert multiclass_brier(y, uniform) > 0.0


def test_reliability_table_shape_and_ece():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ])
    df, ece = reliability_table(y, proba)
    assert len(df) == 10
    assert 0.0 <= ece <= 1.0


def test_source_only_baseline_predicts_train_majority():
    train_df = pd.DataFrame({
        "source_dataset": ["A", "A", "A", "B", "B"],
        "class_id": [0, 0, 1, 2, 2],
    })
    test_df = pd.DataFrame({
        "source_dataset": ["A", "B", "A"],
    })
    y_pred = source_only_baseline(train_df, test_df)
    # A's majority = 0, B's majority = 2
    np.testing.assert_array_equal(y_pred, np.array([0, 2, 0]))


# ---------------------------------------------------------------------------
# Test-lock enforcement
# ---------------------------------------------------------------------------

def test_lock_blocks_second_run(tmp_path: Path):
    """A second call without --force must raise RuntimeError."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 42\n", encoding="utf-8")
    digest = _config_digest(cfg_path)

    # First call creates the lock
    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    lock_path = locks_dir / "final_test_lock.json"
    lock_path.write_text(json.dumps({"timestamp": "t0", "config_digest": digest}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Test lock already present"):
        _enforce_test_lock(results_dir, cfg_path, force=False, reason=None)


def test_lock_force_appends_revision(tmp_path: Path):
    """--force with --reason appends a revision entry and returns the digest."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 42\n", encoding="utf-8")
    digest = _config_digest(cfg_path)

    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    lock_path = locks_dir / "final_test_lock.json"
    lock_path.write_text(json.dumps({"timestamp": "t0", "config_digest": digest}), encoding="utf-8")

    returned = _enforce_test_lock(results_dir, cfg_path, force=True, reason="mask_terms update")
    assert returned == digest

    rev_log = locks_dir / "revision_log.json"
    assert rev_log.exists()
    data = [json.loads(line) for line in rev_log.read_text(encoding="utf-8").splitlines()]
    assert len(data) == 1
    assert data[0]["reason"] == "mask_terms update"
    assert data[0]["config_digest"] == digest
    assert data[0]["results_dir"] == str(results_dir.resolve())


def test_lock_force_requires_reason(tmp_path: Path):
    """--force without --reason must raise."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 42\n", encoding="utf-8")

    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    lock_path = locks_dir / "final_test_lock.json"
    lock_path.write_text(json.dumps({"timestamp": "t0", "config_digest": "x"}), encoding="utf-8")

    with pytest.raises(ValueError, match="--reason must contain non-blank text"):
        _enforce_test_lock(results_dir, cfg_path, force=True, reason=None)
