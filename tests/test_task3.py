# tests/test_task3.py
from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.inference as inference
import src.parse as parser
from src.mask_suggest import suggest_terms, yaml_list
from src.text_build import (LEGACY_OPTIONS, build_model_text, decode_html_and_qp, delete_terms,
                            get_text_options, normalise, strip_email_addresses, strip_urls,
                            strip_header_remnants, strip_temporal_cues, text_settings_from_cfg)

OPTS = {"mask_mode": "delete", "strip_temporal_cues": True, "strip_email_addresses": True,
        "strip_header_remnants": True, "decode_html_in_csv_sources": True}


def _cfg():
    return yaml.safe_load((Path(__file__).resolve().parent / "config_template.yaml").read_text(encoding="utf-8"))


def test_delete_mode_removes_host_without_leaving_fragments():
    text = "mail from jose@monkey.org via www.monkey.org and monkey business"
    out = delete_terms(text, ["monkey"], "delete")
    assert "monkey" not in out and "org" not in out.split() and "maskedterm" not in out
    assert out.split()[:2] == ["mail", "from"]


def test_multiword_and_dotted_terms_and_token_mode():
    assert "free  money" not in delete_terms("get free  money now", ["free money"], "delete")
    assert delete_terms("visit monkey.org today", ["monkey.org"], "delete").split() == ["visit", "today"]
    assert delete_terms("monkey here", ["monkey"], "token") == "maskedterm here"
    assert delete_terms("convince me", ["vince"], "delete") == "convince me"  # whole words only


@pytest.mark.parametrize("raw", [
    "Mon, 5 Nov 2007 10:11:12 +0100", "5th of November 2007", "Nov 5, 2007 10:11", "sep 2002",
    "2007-11-05", "05/11/2007", "10:11:12", "meet in 2016 please", "Monday, November 5th 2007"])
def test_dates_times_and_years_are_removed(raw):
    out = strip_temporal_cues(raw.lower())
    assert not re.search(r"\d{4}|\d{1,2}:\d{2}", out), out


def test_may_march_and_other_numbers_survive():
    sentence = "you may march to room 4127 on floor 12 with 3 people"
    assert strip_temporal_cues(sentence) == sentence


def test_email_addresses_removed():
    assert strip_email_addresses("write to a.b+c@host.example.org now").split() == ["write", "to", "now"]


def test_header_block_and_xspam_lines_removed():
    raw = "Received: from a.example (localhost [127.0.0.1])\n by b.example with ESMTP id 1\nX-Spam-Status: No, hits=0\n\nHello there\nX-Mailer: tool 1.0\nbye"
    cleaned, removed = strip_header_remnants(raw)
    assert removed and "received" not in cleaned.lower() and "x-spam" not in cleaned.lower()
    assert "Hello there" in cleaned and "bye" in cleaned and "x-mailer" not in cleaned.lower()
    plain, flag = strip_header_remnants("Hello: this is just text\nmore")
    assert not flag and plain.startswith("Hello:")
    assert strip_header_remnants("") == ("", False)


def test_html_and_quoted_printable_decoding_never_raises():
    html_qp = "<html><body><div>Caf=C3=A9 =3D ok =20 yes</div></body></html>"
    out = decode_html_and_qp(html_qp)
    assert "<" not in out and "=3D" not in out and "=20" not in out
    assert decode_html_and_qp("plain text only") == "plain text only"


def test_options_none_reproduces_legacy_behaviour():
    subject, body = "Hello ENRON", "Visit http://x.org on 5 Nov 2007 at 10:11. enron rocks"
    legacy = build_model_text(subject, body, ["enron"])
    assert legacy == "subject: hello maskedterm body: visit http://x.org on 5 nov 2007 at 10:11. maskedterm rocks"
    assert build_model_text(subject, body, ["enron"], options=dict(LEGACY_OPTIONS)) == legacy


def test_full_options_pipeline_order():
    text = build_model_text("RE: Fwd", "Received: from a (b)\n\n<div><p>Pay now</p></div> at jose@monkey.org on 5 Nov 2007, monkey.org",
                            ["monkey"], options=OPTS)
    assert text.startswith("subject: re: fwd body: ")
    assert "pay now at" in text
    for forbidden in ("2007", "nov", "monkey", "@", "received", "<div>", "org"):
        assert forbidden not in text.split(), (forbidden, text)


def test_get_text_options_defaults_nulls_and_validation():
    cfg = _cfg()
    assert get_text_options(cfg)["mask_mode"] == "delete"
    legacy_cfg = {"text": {"mask_terms": [], "mask_dates": False}}
    assert get_text_options(legacy_cfg) == LEGACY_OPTIONS
    for key in LEGACY_OPTIONS:
        broken = {"text": {key: None}}
        with pytest.raises(ValueError):
            get_text_options(broken)
    with pytest.raises(ValueError):
        get_text_options({"text": {"mask_mode": "erase"}})
    with pytest.raises(ValueError):
        get_text_options({"text": {"strip_temporal_cues": "yes"}})
    with pytest.raises(ValueError):
        build_model_text("a", "b", options={"unknown": True})


def test_text_settings_from_cfg_requires_masks():
    cfg = _cfg()
    cfg["text"]["mask_terms"] = ["x"]
    settings = text_settings_from_cfg(cfg)
    assert settings["mask_terms"] == ["x"] and settings["text_options"]["strip_temporal_cues"] is True
    cfg["text"]["mask_terms"] = None
    with pytest.raises(ValueError):
        text_settings_from_cfg(cfg)


def test_training_and_prediction_paths_produce_identical_model_text(monkeypatch, tmp_path):
    cfg = _cfg()
    cfg["text"]["mask_terms"] = ["monkey"]
    subject = "Update from JOSE@monkey.org"
    body = "Received: from a\n\nPlease verify <b>now</b> before Nov 5, 2007 10:11. monkey"
    # training path
    record = parser._make_excluded_record("Test", "s", "Phishing", "f", "")
    record.update(subject_raw=subject, body_raw=body, parse_status="ok", record_id="R1")
    frame = pd.DataFrame([record])
    prepared = parser.prepare_records(frame, cfg)
    training_text = prepared.loc[0, "model_text"]
    # prediction path
    captured = {}
    real = inference.build_model_text

    def spy(*args, **kwargs):
        text = real(*args, **kwargs)
        captured["text"] = text
        return text
    monkeypatch.setattr(inference, "build_model_text", spy)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(["meeting budget plan", "free winner prize", "verify password account"] * 2, [0, 1, 2] * 2)
    bundle = {"pipeline": pipeline, "calibrator": None, "calibration_method": "raw",
              "class_map": {0: "Legitimate", 1: "Spam", 2: "Phishing"}, "model_version": "t",
              "max_input_chars": cfg["text"]["max_input_chars"], **text_settings_from_cfg(cfg)}
    table = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "decision_table.yaml").read_text(encoding="utf-8"))
    result = inference.predict_email(subject, body, bundle, table)
    assert result["status"] == "success"
    assert captured["text"] == training_text
    assert "monkey" not in training_text and "2007" not in training_text


def test_suggest_terms_filters_and_never_edits_config(tmp_path):
    flagged = pd.DataFrame({
        "class": ["Legitimate", "Phishing", "Phishing", "Spam"],
        "term": ["kaminski", "monkey org", "2007", "sep"],
        "top_source": ["Enron-Spam", "Nazario", "Nazario", "SpamAssassin"],
        "top_source_share": [1.0, 0.99, 0.99, 0.95]})
    csv = tmp_path / "flagged.csv"
    flagged.to_csv(csv, index=False)
    config = tmp_path / "config.yaml"
    config.write_text("text:\n  mask_terms: []\n", encoding="utf-8")
    before = config.read_text(encoding="utf-8")
    out = tmp_path / "cands.csv"
    cands = suggest_terms(csv, 0.9, out)
    assert set(cands["term"]) == {"kaminski", "monkey", "org"}  # years and month abbreviations are covered elsewhere
    assert 'mask_terms:' in yaml_list(cands) and '"kaminski"' in yaml_list(cands)
    assert config.read_text(encoding="utf-8") == before
    with pytest.raises(FileExistsError):
        suggest_terms(csv, 0.9, out)
    assert len(suggest_terms(csv, 1.0, tmp_path / "strict.csv")) == 1


def test_strip_urls_removes_links_but_keeps_surrounding_words():
    text = "click https://evil.example.com/login?id=1&x=2 now or visit www.bank.example.org/secure (urgent) ftp://files.example.net/a.zip end"
    out = strip_urls(text).split()
    assert out == ["click", "now", "or", "visit", "(urgent)", "end"]
    assert strip_urls("no links here") == "no links here"


def test_strip_urls_option_is_off_by_default_and_works_through_build_model_text():
    raw = ("Pay now", "Go to http://bad.example.org/x and www.bad.example.org today")
    legacy = build_model_text(*raw)
    assert "http" in legacy and "www" in legacy
    cleaned = build_model_text(*raw, options={"strip_urls": True})
    assert "http" not in cleaned and "www" not in cleaned and "go to" in cleaned and "today" in cleaned
    assert get_text_options(_cfg())["strip_urls"] is False
    assert get_text_options({"text": {}})["strip_urls"] is False


def test_tokenised_enron_style_urls_and_emails_are_not_matched_by_the_strippers():
    """Enron-Spam is distributed tokenised; stripping natural-form links elsewhere would create a source cue."""
    enron_style = "write me back here : http : / / datesaturday . com / d / 1 . php or royal _ b @ reliantenergy . com"
    assert strip_urls(enron_style) == enron_style and strip_email_addresses(enron_style) == enron_style
    cfg = _cfg()
    assert cfg["text"]["strip_urls"] is False and cfg["text"]["strip_email_addresses"] is False
