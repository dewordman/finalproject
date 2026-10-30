# tests/test_task2.py
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import pytest
import yaml
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

import scripts.run_pipeline as runner
import src.parse as parser
from src.acquire import build_manifest
from src.dedup import assign_duplicate_groups, compute_exact_hash
from src.external import _duplicate_mask, run_external
from src.results_index import generate_results_index
from src.split import make_partitions, assert_no_metadata_in_features
from src.train import build_pipeline


def _config(tmp_path):
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {key: str(tmp_path / value) for key, value in cfg["paths"].items()}
    cfg["nazario"]["main_dir"] = str(tmp_path / "data/raw/nazario")
    cfg["nazario"]["external_dir"] = str(tmp_path / "data/raw/nazario_external")
    cfg["nazario"]["main_files"] = ["phishing2.mbox"]
    cfg["language"]["min_chars_for_detection"] = 100000
    cfg["split"]["n_candidates"] = 5
    return cfg


def _message(subject, body, date=None):
    headers = f"Subject: {subject}\nContent-Type: text/plain; charset=utf-8\n"
    if date is not None:
        headers += f"Date: {date}\n"
    return (headers + "\n" + body + "\n").encode("utf-8")


def _write_mbox(path, messages):
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = b"From synthetic@example.org Fri Mar 17 10:00:00 2006\n"
    path.write_bytes(b"".join(envelope + message + b"\n" for message in messages))


MALFORMED = b"Date: Fri, 17 Mar 2006 10:00:00 +0000\nSubject: Broken\nbad header without colon\n\nbody\n"
DATE_2006 = "Fri, 17 Mar 2006 10:00:00 +0000"
DATE_2018 = "Sat, 17 Mar 2018 10:00:00 +0000"


def test_allowed_mbox_only_and_malformed_message_is_retained(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    main = Path(cfg["nazario"]["main_dir"])
    _write_mbox(main / "phishing2.mbox", [
        _message("Dated", "verify your password account", DATE_2006),
        MALFORMED,
        _message("Undated", "secure login confirmation details"),
    ])
    (main / "README.txt").write_text("This is not email", encoding="utf-8")
    (main / "LICENSE.txt").write_text("This is not email either", encoding="utf-8")
    _write_mbox(main / "private-phishing4.mbox", [_message("Restricted", "do not read")])
    _write_mbox(main / "unlisted.mbox", [_message("Unlisted", "do not read")])
    opened = []
    real_parse = parser._parse_mbox
    def tracked(path, *args, **kwargs):
        opened.append(path.name)
        return real_parse(path, *args, **kwargs)
    monkeypatch.setattr(parser, "_parse_mbox", tracked)
    frame = parser.parse_nazario(main, cfg=cfg)
    assert opened == ["phishing2.mbox"]
    assert len(frame) == 4
    assert set(frame.source_dataset) == {"Nazario"}
    assert frame.original_label.eq("Phishing").all()
    assert frame.class_label.eq("Phishing").all()
    assert frame.class_id.eq(2).all()
    emails = frame.loc[frame.source_subset.eq("phishing2.mbox")]
    assert len(emails) == 3
    assert emails.era_year.tolist() == [2006, 2006, 2006]
    assert emails.era_group.eq("2005-2007").all()
    assert emails.parse_status.tolist() == ["ok", "excluded", "ok"]
    assert emails.iloc[1].exclusion_reason == "parse_failure"
    restricted = frame.loc[frame.source_subset.eq("private-phishing4.mbox")].iloc[0]
    assert restricted.exclusion_reason == "licence_restriction"
    assert restricted.body_raw == ""
    assert "README.txt" not in set(frame.source_file)
    assert "LICENSE.txt" not in set(frame.source_file)
    assert frame.era_year.dtype == np.dtype("int64")
    assert isinstance(frame.era_group.dtype, pd.StringDtype)
    parser.parse_all(cfg)
    report = pd.read_csv(Path(cfg["paths"]["results_dir"]) / "parse_report.csv")
    excluded_file = report.loc[report.source_file.eq("private-phishing4.mbox")]
    assert excluded_file.exclusion_reason.tolist() == ["licence_restriction"]
    assert report.exclusion_reason.eq("parse_failure").sum() == 1


@pytest.mark.parametrize("filename", ["stray.csv", "STRAY.CSV"])
def test_nazario_csv_is_rejected(tmp_path, filename):
    cfg = _config(tmp_path)
    main = Path(cfg["nazario"]["main_dir"])
    _write_mbox(main / "phishing2.mbox", [_message("Valid", "verify password")])
    (main / filename).write_text("label,text\n1,export\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allow_csv=false"):
        parser.parse_nazario(main, cfg=cfg)


@pytest.mark.parametrize("date,year,group", [
    ("Thu, 17 Mar 2005 10:00:00 +0000", 2005, "2005-2007"),
    ("Tue, 17 Mar 2015 10:00:00 +0000", 2015, "2015+"),
    ("Mon, 17 Mar 2003 10:00:00 +0000", 2003, "other"),
    ("Sun, 17 Mar 2002 10:00:00 +0000", 2006, "2005-2007"),
    ("Wed, 17 Mar 2027 10:00:00 +0000", 2006, "2005-2007"),
    ("invalid date", 2006, "2005-2007"),
])
def test_nazario_date_bounds_and_fallback(tmp_path, date, year, group):
    cfg = _config(tmp_path)
    main = Path(cfg["nazario"]["main_dir"])
    _write_mbox(main / "phishing2.mbox", [_message("Date check", "verify password", date)])
    row = parser.parse_nazario(main, cfg=cfg).iloc[0]
    assert row.parse_status == "ok"
    assert row.era_year == year
    assert row.era_group == group


def test_missing_file_year_default_gives_minus_one(tmp_path):
    cfg = _config(tmp_path)
    cfg["nazario"]["file_year_default"] = {}
    main = Path(cfg["nazario"]["main_dir"])
    _write_mbox(main / "phishing2.mbox", [_message("Undated", "verify password")])
    row = parser.parse_nazario(main, cfg=cfg).iloc[0]
    assert row.era_year == -1 and row.era_group == "other"


@pytest.mark.parametrize("key", ["main_files", "excluded_files", "allow_csv", "file_year_default", "main_dir"])
def test_required_nazario_settings_cannot_be_null(tmp_path, key):
    cfg = _config(tmp_path)
    cfg["nazario"][key] = None
    with pytest.raises(ValueError, match="null"):
        parser.parse_nazario(tmp_path, cfg=cfg)


def test_missing_declared_mbox_fails_clearly(tmp_path):
    cfg = _config(tmp_path)
    main = Path(cfg["nazario"]["main_dir"])
    main.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Configured Nazario main mbox"):
        parser.parse_nazario(main, cfg=cfg)


def test_other_sources_receive_era_fields(tmp_path):
    enron = tmp_path / "enron"
    enron.mkdir()
    pd.DataFrame({"subject": ["Dated", "Undated"], "body": ["meeting budget", "office notes"],
                  "label": ["ham", "spam"], "Date": [DATE_2006, ""]}).to_csv(
        enron / "emails.csv", index=False, encoding="utf-8")
    frame = parser.parse_enron(enron, [], False)
    assert frame.era_year.tolist() == [2006, -1]
    assert frame.era_group.tolist() == ["2005-2007", "unknown"]
    spamassassin = tmp_path / "spamassassin"
    ham = spamassassin / "easy_ham"
    ham.mkdir(parents=True)
    (ham / "email.eml").write_bytes(_message("Dated ham", "meeting notes", DATE_2018))
    row = parser.parse_spamassassin(spamassassin, [], False).iloc[0]
    assert row.era_year == 2018 and row.era_group == "2015+"


@pytest.fixture
def main_case(tmp_path):
    cfg = _config(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    enron = raw / "enron_spam"
    enron.mkdir(parents=True)
    pd.DataFrame({
        "subject": ["Office", "Delivery", "Family", "Casino", "Sale", "Prize"],
        "body": ["board minutes budget meeting", "supplier warehouse shipping invoice",
                 "family weekend picnic garden", "casino jackpot bonus roulette",
                 "discount clearance coupon shopping", "prize raffle lottery winner"],
        "label": ["ham", "ham", "ham", "spam", "spam", "spam"],
    }).to_csv(enron / "emails.csv", index=False, encoding="utf-8")
    _write_mbox(Path(cfg["nazario"]["main_dir"]) / "phishing2.mbox", [
        _message("Account", "password verify account identity security", DATE_2006),
        _message("Wire", "wire urgent bank transfer beneficiary", DATE_2006),
        _message("Portal", "login credentials portal authentication session", DATE_2006),
    ])
    external = Path(cfg["nazario"]["external_dir"])
    _write_mbox(external / "later.mbox", [
        _message("Account", "password verify account identity security", DATE_2018),
        _message("Account", "security identity account verify password", DATE_2018),
        _message("Astronomy", "spaceship telescope planet galaxy observatory", DATE_2018),
        MALFORMED.replace(DATE_2006.encode("utf-8"), DATE_2018.encode("utf-8")),
    ])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, cfg_path


def test_external_preparation_is_identical_but_main_parse_never_calls_it(main_case, monkeypatch):
    cfg, _ = main_case
    cfg = copy.deepcopy(cfg)
    cfg["text"].update(mask_terms=["identity"], mask_dates=True, max_input_chars=45)
    external = parser.parse_nazario_external(cfg)
    monkeypatch.setattr(parser, "parse_nazario_external",
                        lambda *args: pytest.fail("Main parse called external parser"))
    main = parser.parse_all(cfg)
    first = main.loc[main.subject_raw.eq("Account")].iloc[0]
    counterpart = external.loc[external.subject_raw.eq("Account") & external.body_raw.str.startswith("password")].iloc[0]
    assert first.model_text == counterpart.model_text
    assert len(first.model_text) <= 45
    assert "identity" not in first.model_text
    assert external.partition_id.eq("external_temporal").all()
    assert external.era_group.eq("2015+").all()
    assert not main.partition_id.eq("external_temporal").any()
    report = pd.read_csv(Path(cfg["paths"]["results_dir"]) / "parse_report.csv")
    assert {"era_year", "era_group"} <= set(report.columns)
    assert not (Path(cfg["paths"]["processed_dir"]) / "manifest.csv").exists()
    assert not set(main.source_file) & {"later.mbox"}


def test_main_duplicate_detection_does_not_fit_tfidf(main_case, monkeypatch):
    cfg, _ = main_case
    main = parser.parse_all(cfg)
    monkeypatch.setattr(TfidfVectorizer, "fit", lambda *args, **kwargs: pytest.fail("Main dedup TF-IDF fit"))
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", lambda *args, **kwargs: pytest.fail("Main dedup TF-IDF fit"))
    assign_duplicate_groups(main, cfg)
    reference = joblib.load(Path(cfg["paths"]["processed_dir"]) / "duplicate_reference.joblib")
    assert isinstance(reference["vectorizer"], HashingVectorizer)
    assert set(reference["records"].columns) == {"record_id", "model_text", "duplicate_group_id", "exact_hash"}


def test_external_duplicates_removed_without_labels_or_refitting_and_never_split(main_case, monkeypatch):
    cfg, cfg_path = main_case
    main = parser.parse_all(cfg)
    grouped, _ = assign_duplicate_groups(main, cfg)
    processed = Path(cfg["paths"]["processed_dir"])
    before = (processed / "records_grouped.pkl").read_bytes()
    with monkeypatch.context() as guarded:
        guarded.setattr(pd, "read_pickle", lambda *args, **kwargs: pytest.fail("External stage opened main labels"))
        guarded.setattr(TfidfVectorizer, "fit", lambda *args, **kwargs: pytest.fail("External TF-IDF fit"))
        guarded.setattr(TfidfVectorizer, "fit_transform", lambda *args, **kwargs: pytest.fail("External TF-IDF fit"))
        runner._execute_stage("external", cfg, cfg_path, cfg_path.parent, False, None)
    assert (processed / "records_grouped.pkl").read_bytes() == before
    external = pd.read_pickle(processed / "records_external.pkl")
    assert len(external) == 2
    assert external.loc[external.parse_status.eq("ok"), "subject_raw"].tolist() == ["Astronomy"]
    report = pd.read_csv(Path(cfg["paths"]["results_dir"]) / "external_report.csv")
    assert report.iloc[0].to_dict() == {"era_year": 2018, "parsed": 4, "excluded": 1,
                                      "removed_as_duplicate": 2, "retained": 1}
    runner._execute_stage("split", cfg, cfg_path, cfg_path.parent, False, None)
    partitioned = pd.read_pickle(processed / "records_partitioned.pkl")
    assert not set(external.record_id) & set(grouped.record_id)
    assert not set(external.record_id) & set(partitioned.record_id)
    assert not grouped.partition_id.eq("external_temporal").any()
    assert not partitioned.partition_id.eq("external_temporal").any()
    injected = pd.concat([main, external], ignore_index=True)
    for function in (assign_duplicate_groups, make_partitions):
        with pytest.raises(ValueError, match="External temporal records"):
            function(injected, cfg)
    model = build_pipeline("nb", {"tfidf": {"min_df": 1}}, 42)
    assert_no_metadata_in_features(model, main)


def test_external_near_duplicate_connections_are_transitive():
    angles = np.deg2rad([20, 40, 90])
    matrix = csr_matrix(np.column_stack([np.cos(angles), np.sin(angles)]))
    class FrozenVectors:
        def transform(self, texts):
            return matrix
    external = pd.DataFrame({"parse_status": ["ok"] * 3, "model_text": ["one", "two", "three"]})
    reference = {"records": pd.DataFrame({"exact_hash": [compute_exact_hash("main")]}),
                 "vectorizer": FrozenVectors(), "vectors": csr_matrix([[1.0, 0.0]])}
    np.testing.assert_array_equal(_duplicate_mask(external, reference, 0.90), [True, True, False])


def test_exact_overlap_is_removed_even_for_a_language_excluded_record():
    external = pd.DataFrame({"parse_status": ["excluded", "ok", "excluded"],
                             "model_text": ["main text", "new text", ""]})
    reference = {"records": pd.DataFrame({"exact_hash": [compute_exact_hash("main text")]}),
                 "vectorizer": None, "vectors": None}
    np.testing.assert_array_equal(_duplicate_mask(external, reference, 0.90), [True, False, False])


def test_external_reads_all_mboxes_and_forces_temporal_metadata(tmp_path):
    cfg = _config(tmp_path)
    directory = Path(cfg["nazario"]["external_dir"])
    _write_mbox(directory / "early.mbox", [_message("Early date", "verify password", DATE_2006)])
    _write_mbox(directory / "undated.mbox", [_message("No date", "verify login")])
    _write_mbox(directory / "private-phishing4.mbox", [_message("Restricted", "do not parse")])
    (directory / "README.txt").write_text("Not email", encoding="utf-8")
    frame = parser.parse_nazario_external(cfg)
    assert len(frame) == 3
    assert set(frame.source_file) == {"early.mbox", "undated.mbox", "private-phishing4.mbox"}
    assert frame.era_group.eq("2015+").all()
    assert frame.partition_id.eq("external_temporal").all()
    assert frame.record_id.str.startswith("E").all()
    assert frame.loc[frame.source_file.eq("early.mbox"), "era_year"].tolist() == [2006]
    assert frame.loc[frame.source_file.eq("undated.mbox"), "era_year"].tolist() == [-1]
    assert frame.loc[frame.source_file.eq("private-phishing4.mbox"),
                     "exclusion_reason"].tolist() == ["licence_restriction"]


def test_manifest_hashes_every_file_and_carries_source_metadata(main_case):
    cfg, cfg_path = main_case
    raw = Path(cfg["paths"]["raw_dir"])
    (raw / "README.txt").write_text("Inventory root file", encoding="utf-8")
    (raw / "nazario" / "private-phishing4.mbox").write_bytes(b"Restricted raw bytes")
    cfg["sources"]["enron_spam"] = {"url": "https://example.org/enron", "note": "Synthetic conversion"}
    runner._execute_stage("manifest", cfg, cfg_path, cfg_path.parent, False, None)
    output = Path(cfg["paths"]["processed_dir"]) / "manifest.csv"
    frame = pd.read_csv(output, keep_default_na=False)
    files = frame.loc[frame.item_type.ne("directory")]
    assert len(files) == len([path for path in raw.rglob("*") if path.is_file()])
    for row in files.itertuples():
        assert row.sha256 == hashlib.sha256((raw / row.path).read_bytes()).hexdigest()
    nazario = files.loc[files.path.eq("nazario/phishing2.mbox")].iloc[0]
    assert nazario.source_url == cfg["sources"]["nazario"]["url"]
    assert nazario.source_licence == "CC-BY-4.0"
    enron = files.loc[files.path.eq("enron_spam/emails.csv")].iloc[0]
    assert enron.source_note == "Synthetic conversion"
    assert enron.source_url == "https://example.org/enron"
    assert {"nazario/private-phishing4.mbox", "README.txt"} <= set(files.path)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_manifest(raw, output, cfg=cfg)
    assert output.read_bytes() == before


def test_external_output_protection_preserves_prior_files(main_case):
    cfg, _ = main_case
    processed = Path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True)
    output = processed / "records_external.pkl"
    output.write_bytes(b"previous external artifact")
    with pytest.raises(FileExistsError):
        run_external(cfg)
    assert output.read_bytes() == b"previous external artifact"


def test_external_and_main_directories_must_be_separate(main_case):
    cfg, _ = main_case
    cfg["nazario"]["external_dir"] = cfg["nazario"]["main_dir"]
    with pytest.raises(ValueError, match="separate"):
        parser.parse_nazario_external(cfg)
    with pytest.raises(ValueError, match="separate"):
        parser.parse_all(cfg)


def test_results_index_direct_call_preserves_previous_index(tmp_path):
    output = tmp_path / "results" / "RESULTS_INDEX.md"
    generate_results_index(tmp_path, output)
    before = output.read_bytes()
    assert b"records_external.pkl" in before
    assert b"duplicate_reference.joblib" in before
    assert b"External records are stored separately" in before
    with pytest.raises(FileExistsError, match="Results index already exists"):
        generate_results_index(tmp_path, output)
    assert output.read_bytes() == before
