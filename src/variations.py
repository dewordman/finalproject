# src/variations.py
"""
Experiment 3: Surface Variation Sensitivity Analysis

This module applies deterministic, minor surface-level changes to correctly
classified Spam and Phishing VALIDATION messages to measure the model's sensitivity
to superficial text alterations.

NOTE: This measures sensitivity to surface changes, NOT adversarial robustness.
The transformations are not optimized to evade the classifier; they simply
simulate common, benign formatting variations or minor typos.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import require
from src.text_build import build_model_text, text_settings_from_cfg
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

CORPORATE_FOOTER = (
    "\n\n--\nBest regards,\nCorporate Communications Team\n"
    "This is an automated message."
)
URL_PATTERN = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_top_contributing_word(pipeline: Any, text: str) -> str:
    """Find the eligible term with the largest absolute class contribution.

    Mask placeholders and representation markers are not raw message words.
    Multiword features cannot receive a single-word spelling change.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])
    if X.nnz == 0:
        return ""

    pred_class = int(pipeline.predict([text])[0])
    coefs = clf.coef_[pred_class]
    feature_names = tfidf.get_feature_names_out()

    cols = X.nonzero()[1]
    contributions = []
    for col in cols:
        term = str(feature_names[col])
        if (
            len(term) < 5
            or term in {"maskedterm", "subject", "body"}
            or any(char.isspace() for char in term)
        ):
            continue
        val = float(X[0, col]) * float(coefs[col])
        contributions.append((val, term))

    if not contributions:
        return ""

    contributions.sort(key=lambda r: abs(r[0]), reverse=True)
    return contributions[0][1]


def _get_contributions(pipeline: Any, text: str) -> Dict[str, float]:
    """Get feature contributions for the predicted class."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])
    if X.nnz == 0:
        return {}

    pred_class = int(pipeline.predict([text])[0])
    coefs = clf.coef_[pred_class]
    feature_names = tfidf.get_feature_names_out()

    contribs: Dict[str, float] = {}
    cols = X.nonzero()[1]
    for col in cols:
        contribs[str(feature_names[col])] = float(X[0, col]) * float(coefs[col])
    return contribs


def _normalize_text(subject: str, body: str, settings: Dict[str, Any], max_chars: int) -> str:
    """Apply the SAME preprocessing used at training time.

    ``settings`` is the dictionary from ``text_settings_from_cfg`` (mask_terms,
    mask_dates, text_options). For backward compatibility a plain list is
    treated as ``mask_terms`` only.
    """
    if isinstance(settings, dict):
        text = build_model_text(subject, body, settings["mask_terms"],
                                mask_dates=settings["mask_dates"],
                                options=settings["text_options"])
    else:
        text = build_model_text(subject, body, list(settings))
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Variation functions — each returns (new_subject, new_body).
# Preprocessing (mask_terms + length cap) is applied UNIFORMLY in the main
# loop so that the transformed model_text is directly comparable to the
# original model_text.
# ---------------------------------------------------------------------------

def apply_v1_misspelling(
    subject: str,
    body: str,
    pipeline: Any,
    original_model_text: str,
) -> Tuple[str, str]:
    """V1: Misspell a term selected from the evaluated original model text.

    Use the stored masked, length-limited representation for term selection.
    Replace one whole-word occurrence in the raw subject or body, preserving
    its case. The main loop applies the same masking and length limit again.
    """
    target_word = _get_top_contributing_word(pipeline, original_model_text)
    if not target_word:
        return subject, body

    pattern = re.compile(r"\b" + re.escape(target_word) + r"\b", re.IGNORECASE)

    # Select the first adjacent inner-letter pair that actually changes the word.
    # Equal adjacent letters would otherwise create a spelling no-op.
    swap_index = next(
        (i for i in range(1, len(target_word) - 2)
         if target_word[i].casefold() != target_word[i + 1].casefold()),
        None,
    )
    if swap_index is None:
        return subject, body

    def replace_word(match: re.Match) -> str:
        chars = list(match.group(0))
        chars[swap_index], chars[swap_index + 1] = (
            chars[swap_index + 1], chars[swap_index]
        )
        return "".join(chars)

    new_subject, replacements = pattern.subn(replace_word, subject, count=1)
    if replacements:
        return new_subject, body
    new_body, replacements = pattern.subn(replace_word, body, count=1)
    if replacements:
        return subject, new_body
    return subject, body


def apply_v2_punctuation_whitespace(subject: str, body: str) -> Tuple[str, str]:
    """V2: Punctuation and whitespace alteration."""
    new_subject = subject + "!" if subject else "!"
    new_body = body.replace(" ", "  ").replace(",", " ,")
    return new_subject, new_body


def apply_v3_prefix(subject: str, body: str) -> Tuple[str, str]:
    """V3: Prefix the subject with 'Re: '."""
    return "Re: " + subject if subject else "Re: ", body


def apply_v4_url_replacement(subject: str, body: str) -> Tuple[str, str]:
    """V4: Replace every URL in the text with the literal 'http://example.org'."""
    return (
        URL_PATTERN.sub("http://example.org", subject),
        URL_PATTERN.sub("http://example.org", body),
    )


def apply_v5_footer(subject: str, body: str) -> Tuple[str, str]:
    """V5: Append a benign corporate-style footer."""
    return subject, body + CORPORATE_FOOTER


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _vector_changed(pipeline: Any, original: str, transformed: str) -> bool:
    """True if the TF-IDF vector differs, i.e. the model can possibly react."""
    tfidf = pipeline.named_steps["tfidf"]
    difference = tfidf.transform([original]) != tfidf.transform([transformed])
    return bool(difference.nnz)


def _sample_per_class(subset: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for class_id, name in ((1, "Spam"), (2, "Phishing")):
        rows = subset[subset["class_id"] == class_id]
        if len(rows) < per_class:
            logger.warning("Only %d correctly classified %s messages available "
                           "(requested %d); using all of them.", len(rows), name, per_class)
        take = min(per_class, len(rows))
        if take:
            parts.append(rows.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else subset.iloc[0:0].copy()


def run_variations(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Surface-variation sensitivity on correctly classified Spam/Phishing
    messages from the VALIDATION partition (the sealed test set is never read).

    Messages are sampled per class (``evaluation.variation_per_class``). Every
    transformed message goes through the SAME text preparation as training
    (mask terms, date/header/HTML options, length cap). Messages whose
    transformation leaves the model text unchanged are counted as unchanged
    and excluded from flip-rate / recall / contribution metrics. A variation is
    flagged ``null_by_design`` when no transformed message changes the TF-IDF
    vector at all, because then the model cannot react by construction.
    """
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    models_dir = Path(require(cfg, "paths.models_dir"))
    seed = int(require(cfg, "seed"))
    per_class = int(require(cfg, "evaluation.variation_per_class"))
    if per_class < 1:
        raise ValueError("evaluation.variation_per_class must be at least 1")
    settings = text_settings_from_cfg(cfg)
    max_chars = int(require(cfg, "text.max_input_chars"))

    ensure_dir(results_dir)
    pairs_path = results_dir / "variation_pairs.csv"
    report_path = results_dir / "variation_report.csv"
    if pairs_path.exists() or report_path.exists():
        raise FileExistsError("Variation outputs already exist; choose a fresh results directory")

    development = pd.read_pickle(processed_dir / "records_development.pkl")
    validation = development[development["partition_id"] == "validation"].copy()
    if not validation["partition_id"].eq("validation").all() or validation.empty:
        raise AssertionError("Variation analysis must use validation rows only")

    pipeline_path = models_dir / "logreg.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model not found at {pipeline_path}. Run --stage select first.")
    pipeline = joblib.load(pipeline_path)

    validation["pred_class"] = pipeline.predict(validation["model_text"])
    probs = pipeline.predict_proba(validation["model_text"])
    validation["confidence"] = [float(probs[i, int(pred)])
                                for i, pred in enumerate(validation["pred_class"])]

    mask = (validation["class_id"] == validation["pred_class"]) & validation["class_id"].isin([1, 2])
    subset = _sample_per_class(validation[mask].copy(), per_class, seed)
    if subset.empty:
        logger.warning("No correctly classified Spam/Phishing validation messages found.")
        return pd.DataFrame()
    n_spam = int((subset["class_id"] == 1).sum())
    n_phishing = int((subset["class_id"] == 2).sum())
    logger.info("Running variations on %d validation messages (Spam=%d, Phishing=%d).",
                len(subset), n_spam, n_phishing)

    variations = {
        "V1_misspelling": apply_v1_misspelling,
        "V2_punctuation_whitespace": apply_v2_punctuation_whitespace,
        "V3_prefix": apply_v3_prefix,
        "V4_url_replacement": apply_v4_url_replacement,
        "V5_footer": apply_v5_footer,
    }
    pairs_records: List[Dict[str, Any]] = []
    report_records: List[Dict[str, Any]] = []

    for var_name, transform_func in variations.items():
        flips = correct_after = changed_count = unchanged_count = vector_changed = 0
        conf_changes: List[float] = []
        contrib_diffs: Dict[str, List[float]] = {}

        for _, row in subset.iterrows():
            true_class = int(row["class_id"])
            orig_pred = int(row["pred_class"])
            orig_conf = float(row["confidence"])
            orig_text = row["model_text"]
            if var_name == "V1_misspelling":
                new_subject, new_body = apply_v1_misspelling(
                    row["subject_raw"], row["body_raw"], pipeline, orig_text)
            else:
                new_subject, new_body = transform_func(row["subject_raw"], row["body_raw"])
            new_text = _normalize_text(new_subject, new_body, settings, max_chars)
            pairs_records.append({
                "record_id": row["record_id"], "variation": var_name,
                "original_model_text": orig_text, "transformed_model_text": new_text,
                "changed": new_text != orig_text,
            })
            if new_text == orig_text:
                unchanged_count += 1
                continue
            changed_count += 1
            if _vector_changed(pipeline, orig_text, new_text):
                vector_changed += 1
            new_pred = int(pipeline.predict([new_text])[0])
            new_conf = float(pipeline.predict_proba([new_text])[0][new_pred])
            flips += int(new_pred != orig_pred)
            correct_after += int(new_pred == true_class)
            conf_changes.append(new_conf - orig_conf)
            orig_contribs = _get_contributions(pipeline, orig_text)
            new_contribs = _get_contributions(pipeline, new_text)
            for feat in set(orig_contribs) | set(new_contribs):
                diff = abs(new_contribs.get(feat, 0.0) - orig_contribs.get(feat, 0.0))
                if diff > 0:
                    contrib_diffs.setdefault(feat, []).append(diff)

        top_terms = [feat for feat, _ in sorted(
            ((f, float(np.mean(d))) for f, d in contrib_diffs.items()),
            key=lambda item: item[1], reverse=True)[:5]]
        record = {
            "variation": var_name, "n": len(subset), "n_spam": n_spam, "n_phishing": n_phishing,
            "changed_count": changed_count, "unchanged_count": unchanged_count,
            "vector_changed_count": vector_changed,
            "null_by_design": vector_changed == 0,
            "prediction_flips": flips,
            "flip_rate": round(flips / changed_count, 4) if changed_count else np.nan,
            "recall_after_change": round(correct_after / changed_count, 4) if changed_count else np.nan,
            "mean_confidence_change": round(float(np.mean(conf_changes)), 4) if conf_changes else np.nan,
            "top_5_changed_terms": " | ".join(top_terms),
        }
        report_records.append(record)
        logger.info("%s: %d/%d changed, vector changed in %d, %d flips, null_by_design=%s",
                    var_name, changed_count, len(subset), vector_changed, flips, record["null_by_design"])

    pd.DataFrame(pairs_records).to_csv(pairs_path, index=False, encoding="utf-8", mode="x")
    report_df = pd.DataFrame(report_records)
    report_df.to_csv(report_path, index=False, encoding="utf-8", mode="x")
    logger.info("Saved %s and %s", pairs_path, report_path)
    return report_df
