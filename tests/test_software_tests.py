# tests/test_software_tests.py
from __future__ import annotations

import random
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import scripts.software_tests as st
from src.text_build import text_settings_from_cfg

ROOT = Path(__file__).resolve().parents[1]
WORDS = {
    0: "project update meeting agenda budget schedule review team thursday actions attend thanks".split(),
    1: "congratulations won free prize winner gift cheap pills limited offer buy save order cash bonus unsubscribe".split(),
    2: "urgent verify account immediately suspension customer unusual activity password login access click secure".split(),
}


def _bundle(tmp_path: Path, vocabulary_present: bool = True) -> Path:
    rng = random.Random(1)
    texts, labels = [], []
    for label, words in WORDS.items():
        for _ in range(30):
            if vocabulary_present:
                texts.append("subject: x body: " + " ".join(rng.sample(words, 8)) + f" tok{rng.randint(1, 9999)}")
            else:  # a vocabulary that shares no word with any message, not even "subject" or "body"
                texts.append(f"zzz{label} yyy{label} xxx{label} qqq{rng.randint(1, 3)}")
            labels.append(label)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(C=10, max_iter=500))])
    pipeline.fit(texts, labels)
    cfg = yaml.safe_load((ROOT / "tests" / "config_template.yaml").read_text(encoding="utf-8"))
    cfg["text"]["mask_terms"] = ["monkey", "subject body"]
    bundle = {"pipeline": pipeline, "calibrator": None, "calibration_method": "raw",
              "class_map": {0: "Legitimate", 1: "Spam", 2: "Phishing"}, "model_version": "test",
              "max_input_chars": 20000, **text_settings_from_cfg(cfg)}
    path = tmp_path / "models" / "run" / "final_bundle.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump(bundle, path)
    return path


def _processed(tmp_path: Path, crossing: bool = False) -> Path:
    rows = []
    for partition in ("train", "validation", "test"):
        for label in (0, 1, 2):
            for i in range(3):
                rows.append({"duplicate_group_id": f"{partition}{label}{i}", "partition_id": partition, "class_id": label})
    if crossing:
        rows.append({"duplicate_group_id": "train00", "partition_id": "test", "class_id": 0})
        rows[0]["duplicate_group_id"] = "train00"
    folder = tmp_path / "data" / "processed" / "run"
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_pickle(folder / "records_partitioned.pkl")
    return folder


def _results(tmp_path: Path, macro_with_two_classes: bool = False) -> Path:
    folder = tmp_path / "results" / "run"
    folder.mkdir(parents=True)
    pd.DataFrame([{"experiment": "holdout_x", "present_classes": "Legitimate, Spam",
                   "macro_f1": 0.5 if macro_with_two_classes else None}]).to_csv(folder / "source_tests_summary.csv", index=False)
    return folder


def _run(tmp_path, bundle, processed, results, extra=()):
    return st.main(["--results-dir", str(results), "--bundle", str(bundle), "--processed-dir", str(processed),
                    "--table", str(ROOT / "configs" / "decision_table.yaml"), *extra])


def _statuses(results: Path) -> dict:
    csv = next(results.glob("software_test_results_*.csv"))
    frame = pd.read_csv(csv)
    return dict(zip(frame["test_id"], frame["status"]))


def test_all_tests_pass_on_a_healthy_bundle(tmp_path, capsys):
    bundle, processed, results = _bundle(tmp_path), _processed(tmp_path), _results(tmp_path)
    log = tmp_path / "predictions.csv"
    pd.DataFrame([{"record_uuid": "u", "predicted_class": "Spam", "text_sha256": "ab" * 32}]).to_csv(log, index=False)
    assert _run(tmp_path, bundle, processed, results, ("--log", str(log))) == 0
    status = _statuses(results)
    expected = {"T01a", "T01a-explain", "T01b", "T01b-explain", "T01c", "T01c-explain", "T02", "T03", "T04",
                "T05", "T06", "T07", "T08", "T09", "T10", "LOG"}
    assert set(status) == expected
    assert set(status.values()) == {"PASS"}, status
    assert "Summary" in capsys.readouterr().out


def test_missing_explanation_is_reported_as_a_failure(tmp_path):
    bundle, processed, results = _bundle(tmp_path, vocabulary_present=False), _processed(tmp_path), _results(tmp_path)
    assert _run(tmp_path, bundle, processed, results) == 1
    status = _statuses(results)
    assert status["T01a-explain"] == "FAIL" and status["T01b-explain"] == "FAIL"
    assert status["T03"] == "PASS" and status["T09"] == "PASS"


def test_crossing_duplicate_group_fails_t05(tmp_path):
    bundle, processed, results = _bundle(tmp_path), _processed(tmp_path, crossing=True), _results(tmp_path)
    assert _run(tmp_path, bundle, processed, results) == 1
    assert _statuses(results)["T05"] == "FAIL"


def test_macro_score_for_a_two_class_holdout_fails_t10(tmp_path):
    bundle, processed, results = _bundle(tmp_path), _processed(tmp_path), _results(tmp_path, macro_with_two_classes=True)
    assert _run(tmp_path, bundle, processed, results) == 1
    assert _statuses(results)["T10"] == "FAIL"


def test_log_with_message_text_fails(tmp_path):
    bundle, processed, results = _bundle(tmp_path), _processed(tmp_path), _results(tmp_path)
    log = tmp_path / "predictions.csv"
    pd.DataFrame([{"timestamp": "t", "subject": "hello", "body": st.LEGITIMATE[1]}]).to_csv(log, index=False)
    assert _run(tmp_path, bundle, processed, results, ("--log", str(log))) == 1
    assert _statuses(results)["LOG"] == "FAIL"


def test_missing_optional_inputs_are_skipped_not_failed(tmp_path):
    bundle, results = _bundle(tmp_path), tmp_path / "results" / "run"
    empty = tmp_path / "nothing"
    assert _run(tmp_path, bundle, empty, results) == 0
    status = _statuses(results)
    assert status["T05"] == "SKIPPED" and status["T10"] == "SKIPPED" and status["LOG"] == "SKIPPED"


def test_missing_bundle_returns_error_code(tmp_path, capsys):
    code = st.main(["--results-dir", str(tmp_path / "r"), "--bundle", str(tmp_path / "none.joblib"),
                    "--table", str(ROOT / "configs" / "decision_table.yaml")])
    assert code == 2 and "not found" in capsys.readouterr().out
