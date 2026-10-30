# tests/test_task4.py
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import scripts.run_pipeline as runner
from src.diagnose import run_diagnose
from src.error_review import run_error_review
from src.experiments import run_nazario_temporal, run_source_tests
from src.sealed import complete_experiment_lock, reserve_experiment_lock
from src.variations import run_variations

VOCAB = {
    0: "meeting project office schedule report budget team lunch review manager contract invoice client agenda".split(),
    1: "discount free winner offer cheap pills casino prize bonus limited deal save lottery cash".split(),
    2: "password verify account security suspended login bank urgent confirm identity paypal billing alert".split(),
}
SOURCES = {0: ["Enron-Spam", "SpamAssassin"], 1: ["Enron-Spam", "SpamAssassin"], 2: ["Nazario"]}


def _cfg(tmp_path: Path) -> dict:
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {key: str(tmp_path / value) for key, value in cfg["paths"].items()}
    cfg["evaluation"].update(bootstrap_reps=10, variation_per_class=5)
    cfg["error_review"]["per_category"] = 2
    return cfg


def _text(label, rng):
    return " ".join(rng.sample(VOCAB[label], 8)) + f" tok{rng.randint(1000, 9999)}"


@pytest.fixture
def dev_case(tmp_path):
    rng = random.Random(3)
    cfg = _cfg(tmp_path)
    processed, models = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["models_dir"])
    processed.mkdir(parents=True)
    models.mkdir(parents=True)
    rows, group = [], 0
    for partition, per_cell in (("train", 12), ("validation", 8), ("test", 6)):
        for label in (0, 1, 2):
            for source in SOURCES[label]:
                for _ in range(per_cell // len(SOURCES[label]) + 1):
                    text = _text(label, rng)
                    rows.append({"record_id": f"R{len(rows):05d}", "model_text": text, "subject_raw": f"s{group}",
                                 "body_raw": text, "class_id": label, "source_dataset": source,
                                 "duplicate_group_id": group, "partition_id": partition, "parse_status": "ok",
                                 "class_label": ["Legitimate", "Spam", "Phishing"][label],
                                 "era_group": "2005-2007" if label == 2 else "unknown", "era_year": 2006 if label == 2 else -1})
                    group += 1
    # make two validation messages deliberately confusable (errors exist for the review sheet)
    frame = pd.DataFrame(rows)
    validation_idx = frame.index[frame.partition_id.eq("validation") & frame.class_id.eq(2)][:3]
    frame.loc[validation_idx, "model_text"] = "meeting project office schedule report"
    train = frame[frame.partition_id.eq("train")]
    pipeline = Pipeline([("tfidf", TfidfVectorizer(lowercase=False)), ("clf", LogisticRegression(C=10, max_iter=500))])
    pipeline.fit(train.model_text, train.class_id)
    joblib.dump(pipeline, models / "logreg.joblib")
    (models / "calibration_choice.json").write_text('{"choice":"raw"}', encoding="utf-8")
    params = {"tfidf": {"ngram_range": [1, 1], "min_df": 1, "sublinear_tf": False, "max_features": None},
              "clf": {"C": 1.0, "class_weight": None, "max_iter": 300}}
    (models / "selected_config.json").write_text(json.dumps({"logreg": params, "nb": params, "svm": params}), encoding="utf-8")
    frame.to_pickle(processed / "records_partitioned.pkl")
    frame[frame.partition_id.isin(["train", "validation"])].to_pickle(processed / "records_development.pkl")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, cfg_path, frame


def _forbid_sealed(monkeypatch):
    real = pd.read_pickle

    def guard(path, *args, **kwargs):
        assert "records_partitioned" not in str(path), "development stage opened the sealed pickle"
        return real(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_pickle", guard)


def test_run_pipeline_registers_and_unblocks_the_new_stages():
    for stage in ("diagnose", "source_tests", "variations", "error_review", "temporal"):
        assert stage in runner.STAGE_ORDER and stage in runner.STAGE_OUTPUTS and stage in runner.STAGE_REQUIRED_INPUTS
    assert runner.LOCKED_STAGES == {"final_test", "temporal"}
    assert not hasattr(runner, "BLOCKED_STAGES")
    assert "final_test" not in runner.DEV_STAGES and "temporal" not in runner.DEV_STAGES


def test_diagnose_computes_shares_and_protects_outputs(dev_case):
    cfg, _, frame = dev_case
    parsed = frame.copy()
    parsed["body_raw"] = ["Received: from x\n\nbody 2007 <div><p>hi</p></div> =3D http://a.b www a@b.org"] * len(parsed)
    parsed["subject_raw"] = "s"
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True)
    parsed.to_pickle(Path(cfg["paths"]["processed_dir"]) / "records_parsed.pkl")
    table = run_diagnose(cfg)
    assert {"share_year_1990_2009", "share_header_remnant", "share_html_tags",
            "share_url_after_prep", "share_email_after_prep", "share_http_word", "share_www_word",
            "share_at_sign"} <= set(table.columns)
    assert (table["share_header_remnant"] == 1.0).all() and (table["share_year_1990_2009"] == 1.0).all()
    assert (table["share_url"] == 1.0).all() and (table["share_email_address"] == 1.0).all()
    assert (table[["share_http_word", "share_www_word", "share_at_sign"]] == 1.0).all().all()
    assert table["n"].sum() == len(parsed)
    with pytest.raises(FileExistsError):
        run_diagnose(cfg)


def test_source_tests_use_development_data_only_and_hold_the_source_out(dev_case, monkeypatch):
    cfg, _, frame = dev_case
    _forbid_sealed(monkeypatch)
    seen = []
    import src.experiments as experiments
    real = experiments.build_pipeline

    def spy(name, params, seed):
        pipeline = real(name, params, seed)
        seen.append(pipeline)
        return pipeline
    monkeypatch.setattr(experiments, "build_pipeline", spy)
    summary = run_source_tests(cfg)
    results = Path(cfg["paths"]["results_dir"])
    assert (results / "source_tests_summary.csv").exists()
    assert len(summary) == 2 and seen, "both hold-outs must fit fresh pipelines"
    for name in ("holdout_enron", "holdout_spamassassin"):
        assert (results / f"source_test_{name}.csv").exists()
    # the sealed partition's rows never appear: every evaluated row comes from train/validation
    dev = pd.read_pickle(Path(cfg["paths"]["processed_dir"]) / "records_development.pkl")
    assert set(dev["partition_id"]) == {"train", "validation"}
    with pytest.raises(FileExistsError):
        run_source_tests(cfg)


def test_variations_use_validation_rows_and_sample_per_class(dev_case, monkeypatch):
    cfg, _, _ = dev_case
    _forbid_sealed(monkeypatch)
    report = run_variations(cfg)
    assert set(report["variation"]) == {"V1_misspelling", "V2_punctuation_whitespace", "V3_prefix",
                                        "V4_url_replacement", "V5_footer"}
    assert (report["n_spam"] <= 5).all() and (report["n_phishing"] <= 5).all()
    assert (report["n"] == report["n_spam"] + report["n_phishing"]).all()
    nulls = report.set_index("variation")["null_by_design"]
    assert bool(nulls["V2_punctuation_whitespace"]) and not bool(nulls["V5_footer"])
    pairs = pd.read_csv(Path(cfg["paths"]["results_dir"]) / "variation_pairs.csv")
    dev = pd.read_pickle(Path(cfg["paths"]["processed_dir"]) / "records_development.pkl")
    validation_ids = set(dev.loc[dev.partition_id.eq("validation"), "record_id"])
    assert set(pairs["record_id"]) <= validation_ids
    with pytest.raises(FileExistsError):
        run_variations(cfg)


def test_variations_reject_non_positive_sample_size(dev_case):
    cfg, _, _ = dev_case
    cfg["evaluation"]["variation_per_class"] = 0
    with pytest.raises(ValueError):
        run_variations(cfg)


def test_error_review_is_validation_only_one_row_per_group_without_bodies(dev_case, monkeypatch):
    cfg, _, _ = dev_case
    _forbid_sealed(monkeypatch)
    sheet = run_error_review(cfg)
    assert len(sheet) > 0
    assert sheet["record_id"].is_unique
    assert not any("body" in column for column in sheet.columns)
    coded = ["ambiguous_intent", "promotional_wording", "impersonation_or_urgency",
             "formatting_artefact", "plausible_source_dependence", "coder_note"]
    assert (sheet[coded].fillna("") == "").all().all()
    dev = pd.read_pickle(Path(cfg["paths"]["processed_dir"]) / "records_development.pkl")
    assert set(sheet["record_id"]) <= set(dev.loc[dev.partition_id.eq("validation"), "record_id"])
    assert set(sheet["category"]) <= {"legitimate_to_spam_or_phishing", "spam_to_legitimate",
                                      "phishing_to_legitimate", "phishing_to_spam"}


def _external(dev_case):
    cfg, cfg_path, frame = dev_case
    rng = random.Random(11)
    rows = [{"record_id": f"X{i}", "model_text": _text(2, rng) if i % 5 else "meeting project office schedule",
             "class_id": 2, "class_label": "Phishing", "parse_status": "ok", "partition_id": "external_temporal",
             "era_year": 2016 + i % 2, "source_dataset": "Nazario"} for i in range(10)]
    pd.DataFrame(rows).to_pickle(Path(cfg["paths"]["processed_dir"]) / "records_external.pkl")
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg, cfg_path


def test_temporal_never_refits_is_locked_and_writes_expected_tables(dev_case, monkeypatch):
    cfg, cfg_path = _external(dev_case)

    def no_fit(*args, **kwargs):
        raise AssertionError("the temporal experiment must not refit any model")
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    table = run_nazario_temporal(cfg, cfg_path)
    results = Path(cfg["paths"]["results_dir"])
    assert table.loc[0, "era_year"] == "overall" and table.loc[0, "n"] == 10
    assert set(table["era_year"]) == {"overall", "2016", "2017"}
    assert "temporal generalisation within one phishing source" in table.loc[0, "note"]
    matrix = pd.read_csv(results / "confusion_matrix_nazario_temporal.csv", index_col=0)
    assert matrix.shape == (3, 3) and int(matrix.to_numpy().sum()) == 10
    assert int(matrix.loc["Legitimate"].sum()) == 0 and int(matrix.loc["Spam"].sum()) == 0
    assert (results / "nazario_temporal_missed_terms.csv").exists()
    lock = json.loads((Path(cfg["paths"]["locks_dir"]) / "nazario_temporal_lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "completed"
    with pytest.raises(FileExistsError):
        run_nazario_temporal(cfg, cfg_path)


def test_temporal_second_run_refused_and_force_is_logged_with_history(dev_case):
    cfg, cfg_path = _external(dev_case)
    run_nazario_temporal(cfg, cfg_path)
    fresh = dict(cfg, paths={**cfg["paths"], "results_dir": str(Path(cfg["paths"]["results_dir"]).parent / "results_fresh")})
    with pytest.raises(RuntimeError, match="already run"):
        run_nazario_temporal(fresh, cfg_path)
    with pytest.raises(ValueError):
        run_nazario_temporal(fresh, cfg_path, force=True, reason="   ")
    run_nazario_temporal(fresh, cfg_path, force=True, reason="synthetic revision")
    locks = Path(cfg["paths"]["locks_dir"])
    log = [json.loads(line) for line in (locks / "revision_log.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [e["experiment"] for e in log] == ["nazario_temporal"] and log[0]["reason"] == "synthetic revision"
    lock = json.loads((locks / "nazario_temporal_lock.json").read_text(encoding="utf-8"))
    assert len(lock["history"]) == 1 and lock["status"] == "completed"


def test_temporal_failure_keeps_the_reservation(dev_case):
    cfg, cfg_path = _external(dev_case)
    frame = pd.read_pickle(Path(cfg["paths"]["processed_dir"]) / "records_external.pkl")
    frame["class_id"] = 0  # violates "all external records are Phishing"
    frame.to_pickle(Path(cfg["paths"]["processed_dir"]) / "records_external.pkl")
    with pytest.raises(AssertionError):
        run_nazario_temporal(cfg, cfg_path)
    lock = json.loads((Path(cfg["paths"]["locks_dir"]) / "nazario_temporal_lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "failed"


def test_experiment_lock_rules(tmp_path):
    results, locks, cfg_path = tmp_path / "results", tmp_path / "locks", tmp_path / "c.yaml"
    cfg_path.write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        reserve_experiment_lock("x", results, cfg_path, False, None, locks_dir=results / "locks")
    with pytest.raises(ValueError):
        reserve_experiment_lock("bad name!", results, cfg_path, False, None, locks_dir=locks)
    digest = reserve_experiment_lock("exp", results, cfg_path, False, None, locks_dir=locks, run_id="r1")
    with pytest.raises(RuntimeError):
        complete_experiment_lock("exp", locks, "wrong", "r1")
    complete_experiment_lock("exp", locks, digest, "r1")
    with pytest.raises(RuntimeError):
        reserve_experiment_lock("exp", results, cfg_path, False, None, locks_dir=locks)
    assert not (locks / "revision_log.json").exists()
