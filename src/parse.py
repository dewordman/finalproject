# src/parse.py
from __future__ import annotations

import email
import mailbox
from email import policy
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langdetect import detect, DetectorFactory

from src.acquire import build_manifest
from src.config import load_config, require
from src.text_build import html_to_text, build_model_text, get_text_options
from src.utils import ensure_dir, sha256_text, get_logger

logger = get_logger(__name__)
CLASS_MAP = {"Legitimate": 0, "Spam": 1, "Phishing": 2}
RECORD_COLUMNS = [
    "record_id", "source_dataset", "source_subset", "original_label",
    "class_label", "class_id", "subject_raw", "body_raw", "model_text",
    "exact_hash", "duplicate_group_id", "parse_status", "exclusion_reason",
    "partition_id", "era_year", "era_group", "source_file",
    "source_row_index", "input_label",
]
INTEGER_COLUMNS = {"class_id", "duplicate_group_id", "era_year", "source_row_index"}


def _records_frame(records) -> pd.DataFrame:
    frame = pd.DataFrame(records).reindex(columns=RECORD_COLUMNS)
    for column in RECORD_COLUMNS:
        if column in INTEGER_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").fillna(-1).astype("int64")
        else:
            frame[column] = frame[column].fillna("").astype("string")
    return frame


def _date_year(value: Any, fallback: int = -1) -> int:
    try:
        year = parsedate_to_datetime(str(value or "")).year
        if 2003 <= year <= 2026:
            return year
    except (TypeError, ValueError, OverflowError, AttributeError):
        pass
    return fallback


def _era_group(year: int, unknown: str = "unknown") -> str:
    if 2005 <= year <= 2007:
        return "2005-2007"
    if year >= 2015:
        return "2015+"
    return unknown if year == -1 else "other"


def _make_excluded_record(
    dataset: str, subset: str, original_label: str, file_name: str,
    reason: str, details: str = ""
) -> Dict[str, Any]:
    class_id = CLASS_MAP.get(original_label, -1)
    return {
        "record_id": "", "source_dataset": dataset, "source_subset": subset,
        "original_label": original_label,
        "class_label": original_label if class_id != -1 else "", "class_id": class_id,
        "subject_raw": "", "body_raw": "", "model_text": "",
        "exact_hash": sha256_text(file_name + details), "duplicate_group_id": -1,
        "parse_status": "excluded", "exclusion_reason": reason, "partition_id": "",
        "era_year": -1, "era_group": "other" if dataset == "Nazario" else "unknown",
        "source_file": file_name, "source_row_index": -1, "input_label": original_label,
    }


def _message_text(msg) -> tuple[str, str]:
    subject = str(msg.get("Subject", "") or "").strip()
    body = ""
    for content_type in ("text/plain", "text/html"):
        for part in msg.walk():
            if part.is_multipart() or part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() != content_type:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                payload = part.get_payload() or ""
            if isinstance(payload, bytes):
                payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            body = str(payload)
            if content_type == "text/html":
                body = html_to_text(body)
            break
        if body:
            break
    defects = [type(defect).__name__ for part in msg.walk() for defect in part.defects]
    if defects:
        raise ValueError("Malformed email: " + ", ".join(defects))
    return subject, body.strip()


def _message_record(msg, dataset, subset, label, file_name, index,
                    mask_terms, mask_dates, fallback=-1, era_override=None):
    year = _date_year(msg.get("Date"), fallback)
    group = era_override or _era_group(year, "other" if dataset == "Nazario" else "unknown")
    try:
        subject, body = _message_text(msg)
        record = _make_excluded_record(dataset, subset, label, file_name, "")
        record.update(subject_raw=subject, body_raw=body, parse_status="ok",
                      model_text=build_model_text(subject, body, mask_terms=mask_terms,
                                                  mask_dates=mask_dates),
                      exact_hash=sha256_text(subject + body))
    except Exception as exc:
        record = _make_excluded_record(dataset, subset, label, file_name,
                                       "parse_failure", f"{index}: {exc}")
    record.update(era_year=year, era_group=group, source_row_index=index)
    return record


def _parse_file(file_path: Path, dataset: str, subset: str, original_label: str,
                mask_terms: Optional[List[str]] = None, mask_dates: bool = True) -> Dict[str, Any]:
    file_path = Path(file_path)
    try:
        msg = email.message_from_bytes(file_path.read_bytes(), policy=policy.default)
        return _message_record(msg, dataset, subset, original_label, file_path.name, 0,
                               mask_terms, mask_dates)
    except Exception as exc:
        return _make_excluded_record(dataset, subset, original_label, file_path.name,
                                      "parse_failure", str(exc))


def _parse_mbox(file_path: Path, dataset: str, subset: str, original_label: str,
                mask_terms: Optional[List[str]] = None, mask_dates: bool = True,
                *, file_year_default: int = -1, era_group: Optional[str] = None,
                partition_id: str = "") -> List[Dict[str, Any]]:
    records = []
    box = None
    file_path = Path(file_path)
    try:
        box = mailbox.mbox(
            str(file_path), create=False,
            factory=lambda handle: email.message_from_binary_file(handle, policy=policy.default),
        )
        for index, key in enumerate(box.keys()):
            try:
                record = _message_record(box[key], dataset, subset, original_label,
                    file_path.name, index, mask_terms, mask_dates, file_year_default, era_group)
                if record["parse_status"] == "ok" and not (
                        record["subject_raw"] or record["body_raw"]):
                    record.update(parse_status="excluded", exclusion_reason="parse_failure")
            except Exception as exc:
                record = _make_excluded_record(dataset, subset, original_label, file_path.name,
                                               "parse_failure", f"{index}: {exc}")
                record.update(source_row_index=index, era_year=file_year_default,
                    era_group=era_group or _era_group(file_year_default,
                        "other" if dataset == "Nazario" else "unknown"))
            record["partition_id"] = partition_id
            records.append(record)
        if not records:
            raise ValueError("No mbox messages found")
    except Exception as exc:
        record = _make_excluded_record(dataset, subset, original_label, file_path.name,
                                       "parse_failure", str(exc))
        record.update(era_year=file_year_default,
            era_group=era_group or _era_group(file_year_default,
                "other" if dataset == "Nazario" else "unknown"), partition_id=partition_id)
        records.append(record)
    finally:
        if box is not None:
            box.close()
    return records


def _parse_csv(csv_path: Path, dataset: str, subset: str, mask_terms, mask_dates):
    records = []
    try:
        frame = pd.read_csv(csv_path, encoding="utf-8", dtype="string", on_bad_lines="skip")
    except Exception as exc:
        logger.error("Failed to read %s: %s", csv_path, exc)
        return records
    subject_words = ("subject", "title", "topic") if dataset == "Enron-Spam" else ("subject", "title")
    subject_col = next((c for c in frame if any(word in c.lower() for word in subject_words)), None)
    body_words = (("message", "body", "text", "content", "email", "mail")
                  if dataset == "Enron-Spam" else ("text", "body", "message", "content"))
    body_col = next((c for c in frame if any(word in c.lower() for word in body_words)), None)
    label_col = next((c for c in frame if any(word in c.lower().replace("_", "")
        .replace(" ", "").replace("/", "") for word in ("label", "class", "type", "spam", "ham"))), None)
    date_col = next((c for c in frame if c.strip().lower() == "date"), None)
    if label_col is None:
        logger.error("%s: Could not find a label column in %s", dataset, csv_path.name)
        return records
    default_label = "Legitimate" if dataset == "SpamAssassin" and "ham" in subset.lower() else "Spam"
    for index, row in frame.iterrows():
        label = default_label
        try:
            subject = str(row[subject_col]).strip() if subject_col and pd.notna(row[subject_col]) else ""
            body = str(row[body_col]).strip() if body_col and pd.notna(row[body_col]) else ""
            raw_label = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else default_label
            if "ham" in raw_label or raw_label in ("0", "legitimate"):
                label = "Legitimate"
            elif "spam" in raw_label or raw_label == "1":
                label = "Spam"
            date = row[date_col] if date_col and pd.notna(row[date_col]) else ""
            if not date:
                date = email.message_from_string(body, policy=policy.default).get("Date", "")
            year = _date_year(date)
            record = _make_excluded_record(dataset, subset, label, csv_path.name, "")
            record.update(subject_raw=subject, body_raw=body, parse_status="ok",
                model_text=build_model_text(subject, body, mask_terms=mask_terms, mask_dates=mask_dates),
                exact_hash=sha256_text(subject + body), source_row_index=int(index),
                input_label=raw_label, era_year=year, era_group=_era_group(year))
        except Exception as exc:
            record = _make_excluded_record(dataset, subset, label, csv_path.name,
                                           "parse_failure", f"{index}: {exc}")
        records.append(record)
    return records


def parse_enron(dir: Path, mask_terms: Optional[List[str]] = None,
                mask_dates: bool = True) -> pd.DataFrame:
    csv_files = list(Path(dir).glob("*.csv"))
    records = _parse_csv(csv_files[0], "Enron-Spam", "csv", mask_terms, mask_dates) if csv_files else []
    return _records_frame(records)


def parse_spamassassin(dir: Path, mask_terms: Optional[List[str]] = None,
                       mask_dates: bool = True) -> pd.DataFrame:
    dir = Path(dir)
    records = []
    csv_files = list(dir.glob("*.csv"))
    if csv_files:
        for path in csv_files:
            records.extend(_parse_csv(path, "SpamAssassin", path.stem, mask_terms, mask_dates))
    else:
        if any((dir / name).exists() for name in ("easy_ham", "hard_ham")):
            subsets = ["easy_ham", "hard_ham", "spam"]
        elif (dir / "ham").exists() and (dir / "spam").exists():
            subsets = ["ham", "spam"]
        else:
            subsets = [path.name for path in dir.iterdir() if path.is_dir()] if dir.exists() else []
        for subset in subsets:
            label = "Legitimate" if "ham" in subset.lower() else "Spam"
            for path in sorted((dir / subset).rglob("*")):
                if path.is_file() and path.name.lower() != "cmds":
                    records.append(_parse_file(path, "SpamAssassin", subset, label, mask_terms, mask_dates))
    return _records_frame(records)


def _nazario_settings(cfg):
    settings = require(cfg, "nazario")
    for key in ("main_dir", "external_dir"):
        if not isinstance(require(cfg, f"nazario.{key}"), str) or not settings[key].strip():
            raise ValueError(f"nazario.{key} must be a non-empty path")
    for key in ("main_files", "excluded_files"):
        names = require(cfg, f"nazario.{key}")
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError(f"nazario.{key} must be a list of unique file names")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"nazario.{key} must contain plain file names")
            if Path(name).suffix.lower() != ".mbox":
                raise ValueError(f"nazario.{key} entries must end in .mbox: {name}")
    if not require(cfg, "nazario.main_files"):
        raise ValueError("nazario.main_files must not be empty")
    if not isinstance(require(cfg, "nazario.allow_csv"), bool):
        raise ValueError("nazario.allow_csv must be a boolean")
    defaults = require(cfg, "nazario.file_year_default")
    if not isinstance(defaults, dict):
        raise ValueError("nazario.file_year_default must be a mapping")
    for name, value in defaults.items():
        if not isinstance(value, int) or isinstance(value, bool) or not 2003 <= value <= 2026:
            raise ValueError(f"nazario.file_year_default.{name} must be an integer in 2003..2026")
    return settings


def _input_path(cfg, dotted_key):
    path = Path(require(cfg, dotted_key))
    root = Path(cfg.get("_project_root") or Path(__file__).resolve().parents[1])
    return path if path.is_absolute() else root / path


def _check_nazario_dir(dir, cfg):
    if not dir.is_dir():
        raise FileNotFoundError(f"Nazario directory not found: {dir}")
    csv_files = sorted(path.name for path in dir.iterdir()
                       if path.is_file() and path.suffix.lower() == ".csv")
    if csv_files and not require(cfg, "nazario.allow_csv"):
        raise ValueError("Nazario CSV input is forbidden by nazario.allow_csv=false: "
                         + ", ".join(csv_files) + ". Move CSV exports outside this directory.")


def _restricted_records(dir, settings, partition=""):
    records = []
    for name in settings["excluded_files"]:
        if (dir / name).is_file():
            record = _make_excluded_record("Nazario", name, "Phishing", name, "licence_restriction")
            year = settings["file_year_default"].get(name, -1)
            record.update(era_year=year, era_group="2015+" if partition else _era_group(year, "other"),
                          partition_id=partition)
            records.append(record)
    return records


def parse_nazario(dir: Path, mask_terms: Optional[List[str]] = None,
                  mask_dates: Optional[bool] = None, *, cfg=None) -> pd.DataFrame:
    """Read only configured main mboxes; old positional arguments remain accepted."""
    if cfg is None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "config_rev2.yaml")
    settings = _nazario_settings(cfg)
    dir = Path(dir)
    _check_nazario_dir(dir, cfg)
    terms = list(require(cfg, "text.mask_terms")) if mask_terms is None else mask_terms
    dates = require(cfg, "text.mask_dates") if mask_dates is None else mask_dates
    records = _restricted_records(dir, settings)
    for name in settings["main_files"]:
        if name in settings["excluded_files"]:
            continue
        path = dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Configured Nazario main mbox is missing: {path}")
        records.extend(_parse_mbox(path, "Nazario", name, "Phishing", terms, dates,
                                  file_year_default=settings["file_year_default"].get(name, -1)))
    return _records_frame(records)


def prepare_records(frame: pd.DataFrame, cfg: Dict[str, Any], prefix="R") -> pd.DataFrame:
    """Apply identical content, language, masking and length rules to both corpora."""
    frame = _records_frame(frame)
    frame = frame.sort_values(["source_dataset", "source_subset", "original_label",
                              "exact_hash", "source_row_index"]).reset_index(drop=True)
    frame["record_id"] = pd.Series([f"{prefix}{i:06d}" for i in range(1, len(frame) + 1)], dtype="string")
    terms = list(require(cfg, "text.mask_terms"))
    dates = require(cfg, "text.mask_dates")
    options = get_text_options(cfg)
    max_chars = int(require(cfg, "text.max_input_chars"))
    min_chars = int(require(cfg, "language.min_chars_for_detection"))
    allowed = require(cfg, "language.allowed")
    if max_chars < 1 or min_chars < 0:
        raise ValueError("Text length limits must be positive (language minimum may be zero)")
    DetectorFactory.seed = int(require(cfg, "seed"))
    for index, row in frame.iterrows():
        if row["parse_status"] != "ok":
            continue
        subject, body = row["subject_raw"].strip(), row["body_raw"].strip()
        reason = ""
        if not subject and not body:
            reason = "empty_content"
        elif len(subject + " " + body) >= min_chars:
            try:
                if detect(subject + " " + body) not in allowed:
                    reason = "non_english"
            except Exception:
                pass
        frame.at[index, "model_text"] = build_model_text(
            subject, body, mask_terms=terms, mask_dates=dates, options=options)[:max_chars]
        if reason:
            frame.at[index, "parse_status"] = "excluded"
            frame.at[index, "exclusion_reason"] = reason
    return frame


def _year_from_filename(name: str) -> int:
    """Year in a file name such as ``phishing-2023.mbox`` (else -1).

    Used when a message has no usable Date header, so that it still lands in the
    correct year row of the temporal table instead of year -1.
    """
    import re
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", name)
    if match and 2003 <= int(match.group(1)) <= 2026:
        return int(match.group(1))
    return -1


def parse_nazario_external(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Parse external mboxes only; this function is never called by parse_all."""
    settings = _nazario_settings(cfg)
    dir = _input_path(cfg, "nazario.external_dir")
    _check_nazario_dir(dir, cfg)
    main = _input_path(cfg, "nazario.main_dir").resolve()
    if dir.resolve() == main or dir.resolve().is_relative_to(main) or main.is_relative_to(dir.resolve()):
        raise ValueError("nazario.external_dir and nazario.main_dir must be separate directories")
    records = _restricted_records(dir, settings, "external_temporal")
    for path in sorted(dir.glob("*.mbox")):
        if path.is_file() and path.name not in settings["excluded_files"]:
            records.extend(_parse_mbox(path, "Nazario", path.name, "Phishing",
                list(require(cfg, "text.mask_terms")), require(cfg, "text.mask_dates"),
                file_year_default=settings["file_year_default"].get(path.name, _year_from_filename(path.name)),
                era_group="2015+", partition_id="external_temporal"))
    return prepare_records(_records_frame(records), cfg, prefix="E")


def _write_manifest(cfg, raw_dir, dfs, final_df, output_path) -> None:
    """Compatibility wrapper for the original manifest helper signature."""
    build_manifest(raw_dir, output_path, cfg=cfg)


def parse_all(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw = Path(require(cfg, "paths.raw_dir"))
    processed = Path(require(cfg, "paths.processed_dir"))
    results = Path(require(cfg, "paths.results_dir"))
    outputs = [processed / "records_parsed.pkl", results / "parse_report.csv"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Parse outputs already exist; choose a fresh results basename")
    terms, dates = list(require(cfg, "text.mask_terms")), require(cfg, "text.mask_dates")
    external = _input_path(cfg, "nazario.external_dir").resolve()
    for main_root in (raw / "enron_spam", raw / "spamassassin", _input_path(cfg, "nazario.main_dir")):
        if external.is_relative_to(main_root.resolve()) or main_root.resolve().is_relative_to(external):
            raise ValueError("The external directory must be separate from every main source directory")
    frames = []
    for folder, parser in (("enron_spam", parse_enron), ("spamassassin", parse_spamassassin)):
        if (raw / folder).is_dir():
            frames.append(parser(raw / folder, terms, dates))
    frames.append(parse_nazario(_input_path(cfg, "nazario.main_dir"), terms, dates, cfg=cfg))
    frame = prepare_records(pd.concat(frames, ignore_index=True), cfg)
    assert not frame["partition_id"].eq("external_temporal").any()
    ensure_dir(processed)
    ensure_dir(results)
    frame.to_pickle(outputs[0])
    frame.drop(columns=["subject_raw", "body_raw", "model_text"]).to_csv(
        outputs[1], index=False, encoding="utf-8", mode="x")
    logger.info("Parsed %d main-corpus rows; manifest is generated by --stage manifest", len(frame))
    return frame
