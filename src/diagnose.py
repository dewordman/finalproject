# src/diagnose.py
"""Read-only confound diagnostics: how much do source, era and e-mail debris
separate the classes before any model is fitted?

Uses ``records_parsed.pkl`` (written before partitioning), so no partition
labels, and in particular no test-partition information, are involved.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import require
from src.text_build import strip_header_remnants
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_MODERN_YEAR = re.compile(r"\b20(?:1\d|2\d)\b")
_OLD_YEAR = re.compile(r"\b(?:199\d|200\d)\b")
_ANY_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_HTML_TAG = re.compile(r"<\s*/?\s*(?:html|body|div|br|table|font|p|span|a|td|tr)\b", re.I)
_QP = re.compile(r"=3D|=20", re.I)
_URL = re.compile(r"https?://|www\.", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# Enron-Spam is tokenised with spaces ("http : / / x . com", "name @ host . com"), so the natural-text
# patterns above never match it. These word-level patterns work for every source.
_HTTP_WORD = re.compile(r"\bhttps?\b", re.I)
_WWW_WORD = re.compile(r"\bwww\b", re.I)
_AT_SIGN = re.compile(r"@")

ERA_COLUMNS = [
    "source_dataset", "class_label", "era_group", "n",
    "share_year_2010_2029", "share_year_1990_2009", "share_any_year",
    "share_any_year_after_prep", "median_chars_model_text",
    "share_html_tags", "share_header_remnant", "share_qp_remnant",
    "share_url", "share_email_address", "share_url_after_prep", "share_email_after_prep",
    "share_http_word", "share_www_word", "share_at_sign",
]


def _share(series: pd.Series, pattern: re.Pattern) -> float:
    return float(series.map(lambda text: bool(pattern.search(text))).mean())


def run_diagnose(cfg: Dict[str, Any]) -> pd.DataFrame:
    processed = Path(require(cfg, "paths.processed_dir"))
    results = Path(require(cfg, "paths.results_dir"))
    era_path = results / "era_source_table.csv"
    header_path = results / "header_remnant_report.csv"
    if era_path.exists() or header_path.exists():
        raise FileExistsError("Diagnostic outputs already exist; choose a fresh results directory")
    ensure_dir(results)
    frame = pd.read_pickle(processed / "records_parsed.pkl")
    frame = frame.loc[frame["parse_status"].eq("ok")].copy()
    if "era_group" not in frame:
        frame["era_group"] = "unknown"
    raw = (frame["subject_raw"].fillna("") + " " + frame["body_raw"].fillna(""))
    frame["_raw"] = raw
    frame["_header"] = frame["body_raw"].fillna("").map(lambda text: strip_header_remnants(text)[1])
    rows = []
    for (source, label, era), part in frame.groupby(
            ["source_dataset", "class_label", "era_group"], sort=True):
        rows.append({
            "source_dataset": source, "class_label": label, "era_group": era, "n": len(part),
            "share_year_2010_2029": _share(part["_raw"], _MODERN_YEAR),
            "share_year_1990_2009": _share(part["_raw"], _OLD_YEAR),
            "share_any_year": _share(part["_raw"], _ANY_YEAR),
            "share_any_year_after_prep": _share(part["model_text"], _ANY_YEAR),
            "median_chars_model_text": float(part["model_text"].str.len().median()),
            "share_html_tags": _share(part["body_raw"].fillna(""), _HTML_TAG),
            "share_header_remnant": float(part["_header"].mean()),
            "share_qp_remnant": _share(part["_raw"], _QP),
            "share_url": _share(part["_raw"], _URL),
            "share_email_address": _share(part["_raw"], _EMAIL),
            "share_url_after_prep": _share(part["model_text"], _URL),
            "share_email_after_prep": _share(part["model_text"], _EMAIL),
            "share_http_word": _share(part["_raw"], _HTTP_WORD),
            "share_www_word": _share(part["_raw"], _WWW_WORD),
            "share_at_sign": _share(part["_raw"], _AT_SIGN),
        })
    table = pd.DataFrame(rows, columns=ERA_COLUMNS).round(4)
    header = (frame.groupby("source_dataset")["_header"].agg(["size", "sum"]).reset_index()
              .rename(columns={"size": "rows", "sum": "rows_with_header_remnants"}))
    header["share"] = (header["rows_with_header_remnants"] / header["rows"]).round(4)
    table.to_csv(era_path, index=False, encoding="utf-8", mode="x")
    header.to_csv(header_path, index=False, encoding="utf-8", mode="x")
    print(table.to_string(index=False))
    logger.info("Saved %s and %s", era_path, header_path)
    return table
