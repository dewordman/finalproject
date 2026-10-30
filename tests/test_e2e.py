# tests/test_e2e.py
"""End-to-end smoke test on tiny synthetic corpora: every stage through run_pipeline."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.run_pipeline as runner

VOCAB = {
    0: "meeting project office schedule report budget team lunch review manager contract "
       "invoice client agenda minutes forecast quarter department colleague deadline plan "
       "proposal training policy staff update budget approval travel calendar weekly notes".split(),
    1: "discount free winner offer cheap pills casino prize bonus limited deal save "
       "lottery cash credit loan mortgage viagra sale buy order guarantee click subscribe "
       "unsubscribe millions rich cheap bargain gift coupon".split(),
    2: "password verify account security suspended login bank urgent confirm identity "
       "paypal update billing alert unusual activity restore access secure ebay card "
       "expire immediately validate customer notice support".split(),
}


def _text(label: int, rng: random.Random, n_words: int = 14) -> str:
    return " ".join(rng.sample(VOCAB[label], n_words)) + f" ref{rng.randint(10_000, 99_999)}"


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["subject", "message", "label"]).to_csv(path, index=False, encoding="utf-8")


def _mbox(path: Path, texts, year: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, text in enumerate(texts):
        chunks.append(
            f"From x{index}@example.org Fri Mar 17 10:00:00 {year}\n"
            f"Subject: notice {index}\nDate: Fri, 17 Mar {year} 10:00:00 +0000\n"
            "Content-Type: text/plain; charset=utf-8\n\n" + text + "\n\n")
    path.write_text("".join(chunks), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    rng = random.Random(7)
    raw = tmp_path / "data" / "raw"
    _write_csv(raw / "enron_spam" / "enron.csv",
               [(f"s{i}", _text(0, rng), "ham") for i in range(70)]
               + [(f"s{i}", _text(1, rng), "spam") for i in range(70)])
    _write_csv(raw / "spamassassin" / "sa_ham.csv", [(f"h{i}", _text(0, rng), "ham") for i in range(60)])
    _write_csv(raw / "spamassassin" / "sa_spam.csv", [(f"p{i}", _text(1, rng), "spam") for i in range(60)])
    _mbox(raw / "nazario" / "phishing2.mbox", [_text(2, rng) for _ in range(80)], 2006)
    _mbox(raw / "nazario_external" / "phishing-2016.mbox", [_text(2, rng) for _ in range(20)], 2016)

    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {key: str(tmp_path / value) for key, value in cfg["paths"].items()}
    cfg["paths"]["raw_dir"] = str(raw)
    cfg["nazario"].update(main_dir=str(raw / "nazario"), external_dir=str(raw / "nazario_external"),
                          main_files=["phishing2.mbox"])
    cfg["language"]["min_chars_for_detection"] = 10 ** 6
    cfg["dedup"].update(near_dup_threshold=0.95, review_sample_size=5)
    cfg["split"]["n_candidates"] = 10
    cfg["grid"] = {
        "tfidf": {"ngram_range": [[1, 1]], "min_df": [1], "sublinear_tf": [False], "max_features": [None]},
        "logreg": {"C": [1.0], "class_weight": [None], "max_iter": 500},
        "nb": {"alpha": [1.0]}, "svm": {"C": [1.0], "max_iter": 2000}}
    cfg["selection"] = {"min_phishing_recall": 0.5, "max_legit_fpr": 0.5}
    cfg["audit"].update(top_n=10, min_source_count=2)
    cfg["calibration"]["n_folds"] = 3
    cfg["evaluation"].update(bootstrap_reps=20, variation_per_class=8)
    cfg["error_review"]["per_category"] = 3
    cfg["text"]["mask_terms"] = ["ebay"]
    cfg["decision"]["table_path"] = str(Path(__file__).resolve().parents[1] / "configs" / "decision_table.yaml")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts" / "run_pipeline.py"))

    def run(stage: str, *extra: str) -> int:
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--config", str(cfg_path), "--stage", stage, *extra])
        return runner.main()
    return cfg, tmp_path, run


def test_every_stage_runs_and_respects_the_protocol(project):
    cfg, root, run = project
    results = Path(cfg["paths"]["results_dir"])
    locks = Path(cfg["paths"]["locks_dir"])
    for stage in ("manifest", "parse", "diagnose", "dedup", "split", "grid", "select", "audit",
                  "calibrate", "bundle", "dev_eval", "source_tests", "variations", "error_review",
                  "external"):
        assert run(stage) == 0, f"stage {stage} failed"

    # Development-only stages produced their outputs
    for name in ("era_source_table.csv", "header_remnant_report.csv", "source_tests_summary.csv",
                 "variation_report.csv", "variation_pairs.csv", "error_review_sheet.csv",
                 "external_report.csv", "environment_freeze.txt"):
        assert (results / name).exists(), name
    assert not (locks / "final_test_lock.json").exists(), "dev stages must not lock or open the test set"

    sheet = pd.read_csv(results / "error_review_sheet.csv")
    assert "body" not in " ".join(sheet.columns).lower()
    assert sheet[["ambiguous_intent", "coder_note"]].isna().all().all()

    report = pd.read_csv(results / "variation_report.csv")
    assert set(report["variation"]) >= {"V2_punctuation_whitespace", "V3_prefix"}
    assert {"n_spam", "n_phishing", "null_by_design"} <= set(report.columns)

    # Once-only experiments are protected
    assert run("temporal") == 0
    assert (locks / "nazario_temporal_lock.json").exists()
    temporal = pd.read_csv(results / "source_test_nazario_temporal.csv")
    assert temporal.loc[0, "era_year"] == "overall" and temporal.loc[0, "n"] == 20
    assert run("temporal") == 1, "a second temporal run must be refused"
    assert run("final_test") == 0
    assert run("final_test") == 1, "a second final test must be refused"
    # A forced rerun into the same results folder is refused before any lock is touched
    assert run("temporal", "--force", "--reason", "synthetic revision") == 1
    lock = json.loads((locks / "nazario_temporal_lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "completed" and not lock["history"]
    assert run("final_test", "--force") == 1  # --force needs a non-blank --reason
    assert run("index") == 0
