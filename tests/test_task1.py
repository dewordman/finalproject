# tests/test_task1.py
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import scripts.run_pipeline as runner
from src.evaluate import (run_dev_eval, run_final_test, _predict_with_model,
                          _enforce_test_lock, evaluation_output_names)
from src.results_index import generate_results_index
from src.train import select_best


class DisagreeingCalibrator:
    def predict_proba(self, texts):
        return np.tile([0.05, 0.90, 0.05], (len(texts), 1))


def _config(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {key: str(tmp_path / value) for key, value in cfg["paths"].items()}
    cfg["evaluation"]["bootstrap_reps"] = 20
    return cfg


@pytest.fixture
def evaluation_case(tmp_path):
    cfg = _config(tmp_path)
    processed, models = (Path(cfg["paths"][key]) for key in ("processed_dir", "models_dir"))
    processed.mkdir(parents=True)
    models.mkdir(parents=True)
    texts = ["meeting project office", "discount free winner", "password verify account"]
    sources = ["SyntheticA", "SyntheticB", "SyntheticC"]
    rows = []
    for partition_index, partition in enumerate(["train", "validation", "test"]):
        for class_id in range(3):
            rows.append({"record_id": f"R{partition_index}{class_id}", "model_text": texts[class_id],
                "class_id": class_id, "source_dataset": sources[class_id],
                "duplicate_group_id": partition_index * 3 + class_id, "partition_id": partition})
    records = pd.DataFrame(rows)
    train = records.loc[records.partition_id.eq("train")]
    for tag, classifier in (("logreg", LogisticRegression(C=10, max_iter=1000)),
                            ("nb", MultinomialNB()),
                            ("svm", LinearSVC(C=10, random_state=42, max_iter=5000))):
        pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", classifier)])
        pipeline.fit(train.model_text, train.class_id)
        joblib.dump(pipeline, models / f"{tag}.joblib")
    (models / "calibration_choice.json").write_text('{"choice":"raw"}', encoding="utf-8")
    (models / "selected_config.json").write_text('{"synthetic":true}', encoding="utf-8")
    records.to_pickle(processed / "records_partitioned.pkl")
    records.loc[records.partition_id.isin(["train", "validation"])].to_pickle(processed / "records_development.pkl")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, cfg_path, records


def test_declared_grid_and_required_settings():
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    assert cfg["grid"]["tfidf"] == {"ngram_range": [[1, 1], [1, 2]], "min_df": [2, 3, 5],
        "sublinear_tf": [False, True], "max_features": [None]}
    assert cfg["grid"]["logreg"] == {"C": [0.1, 1, 10], "class_weight": [None, "balanced"], "max_iter": 1000}
    assert cfg["grid"]["nb"]["alpha"] == [0.1, 0.5, 1.0]
    assert cfg["grid"]["svm"] == {"C": [0.1, 1, 10], "max_iter": 5000}
    assert cfg["split"]["n_candidates"] == 200
    assert cfg["dedup"]["review_sample_size"] == 200
    assert cfg["language"]["min_chars_for_detection"] == 30
    assert cfg["evaluation"]["variation_per_class"] == 150
    assert cfg["paths"]["locks_dir"] == "locks"
    assert "text" not in cfg["text"]


@pytest.mark.parametrize("key", ["min_phishing_recall", "max_legit_fpr"])
def test_null_selection_threshold_raises_value_error(tmp_path, key):
    cfg = _config(tmp_path)
    cfg["selection"] = {"min_phishing_recall": 0.5, "max_legit_fpr": 0.15}
    cfg["selection"][key] = None
    with pytest.raises(ValueError, match="is null"):
        select_best(pd.DataFrame(), cfg)


def test_two_result_tags_isolate_data_and_models(tmp_path):
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    first = runner._configure_run(cfg, tmp_path, cfg_path, str(tmp_path / "run_a"))
    second = runner._configure_run(cfg, tmp_path, cfg_path, str(tmp_path / "run_b"))
    for key in ("processed_dir", "models_dir"):
        assert Path(first["paths"][key]).name == "run_a"
        assert Path(second["paths"][key]).name == "run_b"
        assert first["paths"][key] != second["paths"][key]
    assert first["paths"]["locks_dir"] == second["paths"]["locks_dir"]
    with pytest.raises(ValueError, match="another config/results path"):
        runner._configure_run(cfg, tmp_path, cfg_path, str(tmp_path / "another_parent" / "run_a"))


def test_existing_outputs_are_preserved(tmp_path):
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg = runner._configure_run(cfg, tmp_path, cfg_path)
    old = Path(cfg["paths"]["results_dir"]) / "grid_results.csv"
    old.write_text("previous evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._protect_stage_outputs(tmp_path, "grid", cfg)
    assert old.read_text(encoding="utf-8") == "previous evidence\n"


def test_changed_config_cannot_reuse_a_result_namespace(tmp_path):
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    runner._configure_run(cfg, tmp_path, cfg_path)
    cfg_path.write_text(yaml.safe_dump(cfg) + "# changed settings\n", encoding="utf-8")
    with pytest.raises(ValueError, match="another config/results path"):
        runner._configure_run(cfg, tmp_path, cfg_path)


def test_locks_cannot_be_inside_results(tmp_path):
    cfg = _config(tmp_path)
    cfg["paths"]["locks_dir"] = str(Path(cfg["paths"]["results_dir"]) / "locks")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        runner._configure_run(cfg, tmp_path, cfg_path)


def test_dev_eval_never_opens_the_sealed_pickle(evaluation_case, monkeypatch):
    cfg, cfg_path, records = evaluation_case
    real_read = pd.read_pickle
    opened = []
    def guarded_read(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name != "records_partitioned.pkl"
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_pickle", guarded_read)
    summary = run_dev_eval(cfg, cfg_path)
    assert opened == ["records_development.pkl"]
    assert summary["partition"] == "validation"
    assert summary["validation_size"] == 3
    result_dir = Path(cfg["paths"]["results_dir"])
    assert all((result_dir / name).is_file() for name in evaluation_output_names("dev_"))
    assert not Path(cfg["paths"]["locks_dir"]).exists()
    metrics = pd.read_csv(result_dir / "dev_per_class_metrics_logreg.csv")
    assert np.allclose(metrics.fnr, 1 - metrics.recall)
    intervals = pd.read_csv(result_dir / "dev_bootstrap_ci.csv")
    assert set(intervals.model) == {"logreg", "nb", "svm", "source_baseline"}
    assert np.isfinite(intervals[["mean", "lower", "upper"]].to_numpy()).all()
    predictions = pd.read_csv(result_dir / "dev_predictions_logreg.csv")
    assert predictions.partition_id.eq("validation").all()
    assert set(predictions.record_id) == set(records.loc[records.partition_id.eq("validation"), "record_id"])


def test_dev_eval_rejects_test_rows_in_development_file(evaluation_case):
    cfg, cfg_path, records = evaluation_case
    records.to_pickle(Path(cfg["paths"]["processed_dir"]) / "records_development.pkl")
    with pytest.raises(AssertionError, match="train and validation only"):
        run_dev_eval(cfg, cfg_path)


def test_final_test_twice_fails_before_second_label_read(evaluation_case, monkeypatch):
    cfg, cfg_path, _ = evaluation_case
    run_final_test(cfg, cfg_path)
    lock = Path(cfg["paths"]["locks_dir"]) / "final_test_lock.json"
    previous = lock.read_bytes()
    other = copy.deepcopy(cfg)
    other["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]).parent / "different_run")
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("Second run loaded labels"))
    with pytest.raises(RuntimeError, match="Test lock already present"):
        run_final_test(other, cfg_path)
    assert lock.read_bytes() == previous
    assert not (Path(cfg["paths"]["results_dir"]) / "test_lock.json").exists()


def test_forced_revisions_are_append_only_and_keep_lock_history(evaluation_case):
    cfg, cfg_path, _ = evaluation_case
    run_final_test(cfg, cfg_path)
    locks = Path(cfg["paths"]["locks_dir"])
    old = json.loads((locks / "final_test_lock.json").read_text(encoding="utf-8"))
    for index in (1, 2):
        revised = copy.deepcopy(cfg)
        revised["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]).parent / f"forced_{index}")
        journal = locks / "revision_log.json"
        before = journal.read_bytes() if journal.exists() else b""
        run_final_test(revised, cfg_path, force=True, reason=f"Synthetic revision {index}")
        assert journal.read_bytes().startswith(before)
    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert all(set(entry) == {"timestamp", "run_id", "reason", "config_digest", "results_dir"} for entry in entries)
    current = json.loads((locks / "final_test_lock.json").read_text(encoding="utf-8"))
    assert current["status"] == "completed"
    assert len(current["history"]) == 2
    assert current["history"][0]["run_id"] == old["run_id"]


def test_forced_reason_cannot_be_blank(evaluation_case, monkeypatch):
    cfg, cfg_path, _ = evaluation_case
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("Invalid force loaded labels"))
    with pytest.raises(ValueError, match="non-blank"):
        run_final_test(cfg, cfg_path, force=True, reason="  ")


def test_failed_final_test_retains_reservation_and_revision(evaluation_case, monkeypatch):
    cfg, cfg_path, _ = evaluation_case
    locks = Path(cfg["paths"]["locks_dir"])
    def fail_load(*args, **kwargs):
        assert (locks / "revision_log.json").is_file()
        assert (locks / "final_test_lock.json").is_file()
        raise RuntimeError("Synthetic failure after reservation")
    monkeypatch.setattr(pd, "read_pickle", fail_load)
    with pytest.raises(RuntimeError, match="Synthetic failure"):
        run_final_test(cfg, cfg_path, force=True, reason="Synthetic failure check")
    payload = json.loads((locks / "final_test_lock.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert (locks / "revision_log.json").stat().st_size > 0
    with pytest.raises(RuntimeError, match="Test lock already present"):
        run_final_test(cfg, cfg_path)


def test_legacy_lock_is_adopted_without_deletion(tmp_path):
    result_dir = tmp_path / "results_previous"
    result_dir.mkdir()
    old = result_dir / "test_lock.json"
    old.write_text('{"timestamp":"old","config_digest":"original"}', encoding="utf-8")
    before = old.read_bytes()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Test lock already present"):
        _enforce_test_lock(tmp_path / "new_results", cfg_path, False, None, locks_dir=tmp_path / "locks")
    assert old.read_bytes() == before
    assert (tmp_path / "locks" / "final_test_lock.json").is_file()


def test_raw_argmax_is_kept_when_calibration_disagrees(evaluation_case):
    cfg, _, _ = evaluation_case
    models = Path(cfg["paths"]["models_dir"])
    (models / "calibration_choice.json").write_text('{"choice":"sigmoid"}', encoding="utf-8")
    joblib.dump(DisagreeingCalibrator(), models / "logreg_calibrated.joblib")
    predicted, assessed, scores = _predict_with_model("logreg", models, pd.Series(["meeting project office"]))
    assert predicted.tolist() == [0]
    assert np.argmax(assessed, axis=1).tolist() == [1]
    assert assessed[0, predicted[0]] == pytest.approx(0.05)
    assert scores is None


def test_grid_dispatch_uses_development_only(evaluation_case, monkeypatch):
    cfg, cfg_path, _ = evaluation_case
    (Path(cfg["paths"]["processed_dir"]) / "records_partitioned.pkl").write_bytes(b"sealed data must not be opened")
    seen = []
    def fake_grid(train, validation, effective):
        assert train.partition_id.eq("train").all()
        assert validation.partition_id.eq("validation").all()
        seen.append((len(train), len(validation)))
    monkeypatch.setattr(runner, "run_grid", fake_grid)
    runner._execute_stage("grid", cfg, cfg_path, cfg_path.parent, False, None)
    assert seen == [(3, 3)]


def test_split_writes_a_separate_development_file(evaluation_case, tmp_path):
    cfg, cfg_path, records = evaluation_case
    cfg = copy.deepcopy(cfg)
    processed = tmp_path / "fresh_split_data"
    results = tmp_path / "fresh_split_results"
    processed.mkdir()
    cfg["paths"]["processed_dir"] = str(processed)
    cfg["paths"]["results_dir"] = str(results)
    grouped = records.copy()
    grouped["parse_status"] = "ok"
    grouped["partition_id"] = ""
    grouped.to_pickle(processed / "records_grouped.pkl")
    runner._execute_stage("split", cfg, cfg_path, cfg_path.parent, False, None)
    development = pd.read_pickle(processed / "records_development.pkl")
    assert len(development) == 6
    assert development.partition_id.isin(["train", "validation"]).all()
    assert development.groupby("partition_id").class_id.nunique().to_dict() == {"train": 3, "validation": 3}


def test_all_stops_on_first_failure_and_preserves_logs(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "run_pipeline.py"))
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--config", str(cfg_path), "--stage", "all"])
    stages = []
    def fail(stage, *args):
        stages.append(stage)
        raise ValueError("Synthetic first-stage failure")
    monkeypatch.setattr(runner, "_execute_stage", fail)
    assert runner.main() == 1
    assert stages == ["parse"]
    assert runner.main() == 1
    logs = list((Path(cfg["paths"]["results_dir"]) / "run_logs").glob("*.json"))
    assert len(logs) == 2
    assert all(json.loads(path.read_text(encoding="utf-8"))["stages"][0]["status"] == "failed" for path in logs)


def test_index_resolves_custom_run_paths(evaluation_case):
    cfg, cfg_path, _ = evaluation_case
    result_dir = Path(cfg["paths"]["results_dir"])
    result_dir.mkdir()
    artifact = result_dir / "dev_confusion_matrix_logreg.csv"
    artifact.write_text("synthetic table\n", encoding="utf-8")
    output = result_dir / "RESULTS_INDEX.md"
    generate_results_index(cfg_path.parent, output, paths=cfg["paths"])
    line = next(line for line in output.read_text(encoding="utf-8").splitlines() if str(artifact) in line)
    assert "| present |" in line


def test_experiment_script_contains_only_the_allowed_stages():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.ps1"
    script = path.read_text(encoding="utf-8")
    assert "Remove-Item" not in script
    assert "final_test" not in script
    assert '$LASTEXITCODE -ne 0' in script
    assert '$stages = @("parse", "dedup", "split", "grid", "select", "audit", "calibrate", "bundle", "dev_eval")' in script
