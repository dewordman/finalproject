"""Source-cue audits for fitted TF-IDF Logistic Regression, NB and LinearSVC.

The audit describes source concentration of influential features. It does not
measure prediction accuracy, prove leakage, or change a fitted model.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import require
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)
MODEL_NAMES = ("logreg", "nb", "svm")
LEGACY_FILES = (
    "audit_top_terms.csv", "audit_source_share.csv",
    "audit_flagged_terms.csv", "audit_summary.txt",
)
MODEL_FILES = tuple(
    f"audit_{kind}_{model}.{extension}"
    for model in MODEL_NAMES
    for kind, extension in (
        ("top_terms", "csv"), ("source_share", "csv"),
        ("flagged_terms", "csv"), ("summary", "txt"),
    )
)
AUDIT_OUTPUT_FILES = LEGACY_FILES + MODEL_FILES + (
    "audit_comparison.csv", "audit_summary_all_models.txt",
    "audit_results_all_models.zip",
)
TOP_COLUMNS = ["model", "class", "class_id", "rank", "term", "score", "score_type"]
SHARE_COLUMNS = ["term", "total", "top_source", "top_source_count", "top_source_share"]
COMPARISON_COLUMNS = [
    "model", "class", "class_id", "score_type", "requested_top_n",
    "audited_top_terms", "flagged_terms", "flagged_proportion",
]


def _identify_model(clf: Any, model_name: Optional[str]) -> str:
    inferred = (
        "logreg" if isinstance(clf, LogisticRegression)
        else "svm" if isinstance(clf, LinearSVC)
        else "nb" if isinstance(clf, MultinomialNB)
        else None
    )
    if inferred is None:
        raise TypeError(f"Unsupported audit classifier: {type(clf).__name__}")
    if model_name is not None and model_name != inferred:
        raise ValueError(f"Expected {model_name} but the saved model contains {inferred}")
    return inferred


def _feature_scores(pipeline: Any, model_name: Optional[str]):
    """Return fitted class IDs, feature scores and an explicit score type."""
    clf = pipeline.named_steps["clf"]
    model_name = _identify_model(clf, model_name)
    classes = np.asarray(clf.classes_)
    if len(classes) < 2:
        raise ValueError("An audit requires a fitted model with at least two classes")

    if model_name != "nb":
        scores = np.asarray(clf.coef_, dtype=float)
        # A binary coefficient row belongs to classes_[1]; its negative belongs
        # to classes_[0]. Multiclass rows follow clf.classes_, not dict order.
        if scores.shape[0] == 1 and len(classes) == 2:
            scores = np.vstack([-scores[0], scores[0]])
        score_type = "coefficient"
    else:
        counts = np.asarray(clf.feature_count_, dtype=float)
        log_prob = np.asarray(clf.feature_log_prob_, dtype=float)
        if counts.ndim != 2 or counts.shape != log_prob.shape or counts.shape[0] != len(classes):
            raise ValueError("Naive Bayes feature arrays have inconsistent shapes")
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Naive Bayes feature counts must be finite and non-negative")
        alpha = np.asarray(clf.alpha, dtype=float)
        if alpha.ndim == 0:
            alpha = np.full(counts.shape[1], float(alpha))
        if alpha.shape != (counts.shape[1],) or not np.isfinite(alpha).all() or (alpha < 0).any():
            raise ValueError("Naive Bayes alpha must be a non-negative scalar or feature vector")
        # Match sklearn's effective smoothing when force_alpha=False.
        if not getattr(clf, "force_alpha", False):
            alpha = np.maximum(alpha, 1e-10)
        scores = np.empty_like(log_prob)
        for k in range(len(classes)):
            other_counts = counts[np.arange(len(classes)) != k].sum(axis=0)
            smoothed = other_counts + alpha
            if (smoothed <= 0).any() or not np.isfinite(smoothed.sum()) or smoothed.sum() <= 0:
                raise ValueError("NB audit needs finite complement probabilities; use positive smoothing")
            # Pooled other-class distribution, smoothed once with the fitted
            # alpha. Counts can be fractional because this model uses TF-IDF.
            log_other = np.log(smoothed / smoothed.sum())
            scores[k] = log_prob[k] - log_other
        score_type = "nb_log_probability_contrast"

    n_features = len(pipeline.named_steps["tfidf"].get_feature_names_out())
    if scores.shape != (len(classes), n_features) or not np.isfinite(scores).all():
        raise ValueError("Feature scores must be finite and aligned with fitted classes and vocabulary")
    return model_name, classes, scores, score_type


def top_terms(
    pipeline: Any,
    class_names: Union[Mapping[str, int], Sequence[str]],
    n: int,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Rank up to n positive class-supporting scores using fitted class IDs.

    For NB the score is log P(term|class) - log P(term|pooled other classes).
    This is a feature association measure, not a regression coefficient or a
    full posterior prediction contribution. Legacy lists assume IDs 0..K-1.
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("audit.top_n must be a positive integer")
    mapping = dict(class_names) if isinstance(class_names, Mapping) else {
        name: i for i, name in enumerate(class_names)
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Configured class IDs must be unique")
    model_name, classes, scores, score_type = _feature_scores(pipeline, model_name)
    names_by_id = {class_id: name for name, class_id in mapping.items()}
    if set(classes.tolist()) != set(names_by_id):
        raise ValueError("Configured class IDs do not match the fitted model classes")
    features = pipeline.named_steps["tfidf"].get_feature_names_out()
    rows = []
    for class_id, class_scores in zip(classes, scores):
        indices = np.argsort(class_scores)[::-1]
        indices = [i for i in indices if class_scores[i] > 0][:int(n)]
        for rank, i in enumerate(indices, start=1):
            rows.append({
                "model": model_name, "class": names_by_id[class_id],
                "class_id": class_id, "rank": rank, "term": features[i],
                "score": float(class_scores[i]), "score_type": score_type,
            })
    return pd.DataFrame(rows, columns=TOP_COLUMNS)


def _validate_records(records: pd.DataFrame) -> None:
    required = {"model_text", "source_dataset"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Audit records lack columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("No development records are available for the audit")
    if "partition_id" in records and not records["partition_id"].isin(["train", "validation"]).all():
        raise ValueError("Single-model audits accept train and validation records only")
    for col in required:
        if not records[col].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Audit column {col} must contain non-missing strings")
    if records["source_dataset"].str.strip().eq("").any():
        raise ValueError("Audit sources must not be blank")


def source_share(
    train_df: pd.DataFrame,
    terms: List[str],
    ngram_range: tuple = (1, 1),
    *,
    fitted_vectorizer: Any = None,
) -> pd.DataFrame:
    """Count messages containing each term, using the model's own analyser.

    Source dominance is a raw document share; it is not adjusted for unequal
    source sizes and is not a causal measure of source dependence.
    """
    _validate_records(train_df)
    terms = list(dict.fromkeys(terms))
    if not terms:
        return pd.DataFrame(columns=SHARE_COLUMNS)
    if fitted_vectorizer is None:
        vectorizer = CountVectorizer(
            vocabulary=terms, ngram_range=ngram_range, binary=True, lowercase=False,
        )
    else:
        vectorizer = CountVectorizer(
            vocabulary=terms, analyzer=fitted_vectorizer.build_analyzer(), binary=True,
        )
    X = vectorizer.fit_transform(train_df["model_text"])
    sources = train_df["source_dataset"].to_numpy()
    source_names = pd.unique(sources)
    # Aggregate sparse rows directly: never densify the document-term matrix.
    counts = np.vstack([
        np.asarray(X[sources == source].sum(axis=0)).ravel()
        for source in source_names
    ])
    totals = counts.sum(axis=0)
    rows = []
    for i, term in enumerate(vectorizer.get_feature_names_out()):
        total = int(totals[i])
        if total == 0:
            continue
        source_idx = int(np.argmax(counts[:, i]))
        top_count = int(counts[source_idx, i])
        rows.append({
            "term": term, "total": total, "top_source": source_names[source_idx],
            "top_source_count": top_count, "top_source_share": top_count / total,
        })
    return pd.DataFrame(rows, columns=SHARE_COLUMNS)


def flag_source_cues(top_df: pd.DataFrame, share_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply the same source-share rule to each model's selected features."""
    min_count = require(cfg, "audit.min_source_count")
    threshold = float(require(cfg, "audit.dominance_threshold"))
    if isinstance(min_count, bool) or int(min_count) != min_count or min_count < 1:
        raise ValueError("audit.min_source_count must be a positive integer")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("audit.dominance_threshold must be between 0 and 1")
    flagged = share_df.loc[
        (share_df["total"] >= min_count) & (share_df["top_source_share"] >= threshold)
    ].copy()
    return flagged.merge(top_df, on="term", how="inner", validate="one_to_many")


def run_audit(
    pipeline: Any, train_val_df: pd.DataFrame, cfg: Dict[str, Any],
    results_dir: Path, model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Write one model's audit; maintain the original LR file schemas."""
    _validate_records(train_val_df)
    results_dir = ensure_dir(results_dir)
    model_name = _identify_model(pipeline.named_steps["clf"], model_name)
    score_type = "nb_log_probability_contrast" if model_name == "nb" else "coefficient"
    class_map = require(cfg, "classes")
    n = require(cfg, "audit.top_n")
    top_df = top_terms(pipeline, class_map, n, model_name)
    share_df = source_share(
        train_val_df, top_df["term"].unique().tolist(),
        fitted_vectorizer=pipeline.named_steps["tfidf"],
    )
    missing_terms = set(top_df["term"]).difference(share_df["term"])
    if missing_terms:
        raise ValueError("Ranked model terms are absent from development records; check model/data versions")
    flagged_df = flag_source_cues(top_df, share_df, cfg)
    summaries = []
    lines = [
        f"AUDIT SUMMARY: {model_name}", "=" * 40,
        f"Feature score: {score_type}", "Records: train and validation only",
        f"Source dominance threshold: {require(cfg, 'audit.dominance_threshold')}",
        f"Minimum messages containing a term: {require(cfg, 'audit.min_source_count')}",
        "Source shares are not adjusted for source sizes.",
        "Flags indicate possible source cues; they do not prove leakage.",
        "Feature score magnitudes are not comparable between model families.", "",
    ]
    for class_name, class_id in class_map.items():
        class_top = top_df.loc[top_df["class_id"] == class_id]
        class_flags = flagged_df.loc[flagged_df["class_id"] == class_id]
        total, count = len(class_top), len(class_flags)
        proportion = count / total if total else float("nan")
        summaries.append({
            "model": model_name, "class": class_name, "class_id": class_id,
            "score_type": score_type, "requested_top_n": n,
            "audited_top_terms": total, "flagged_terms": count,
            "flagged_proportion": proportion,
        })
        display = f"{proportion:.2%}" if total else "n/a"
        lines += [f"Class: {class_name}", f"  Flagged terms proportion: {display} ({count}/{total})"]
        if total < n:
            lines.append(f"  Only {total} positive-scoring features were available (requested {n}).")
        for row in class_flags.itertuples(index=False):
            lines.append(
                f"    - '{row.term}' (score: {row.score:.4f}, "
                f"dominates '{row.top_source}' at {row.top_source_share:.2%})"
            )
        lines.append("")
    summary_text = "\n".join(lines)
    top_df.to_csv(results_dir / f"audit_top_terms_{model_name}.csv", index=False, encoding="utf-8")
    share_df.to_csv(results_dir / f"audit_source_share_{model_name}.csv", index=False, encoding="utf-8")
    flagged_df.to_csv(results_dir / f"audit_flagged_terms_{model_name}.csv", index=False, encoding="utf-8")
    (results_dir / f"audit_summary_{model_name}.txt").write_text(summary_text, encoding="utf-8")

    if model_name == "logreg":
        top_df[["class", "rank", "term", "score"]].rename(
            columns={"score": "coefficient"}
        ).to_csv(results_dir / LEGACY_FILES[0], index=False, encoding="utf-8")
        share_df.to_csv(results_dir / LEGACY_FILES[1], index=False, encoding="utf-8")
        flagged_df[SHARE_COLUMNS + ["class", "score"]].rename(
            columns={"score": "coefficient"}
        ).to_csv(results_dir / LEGACY_FILES[2], index=False, encoding="utf-8")
        (results_dir / LEGACY_FILES[3]).write_text(summary_text, encoding="utf-8")
    logger.info(f"Audit {model_name}: saved per-model files ({len(top_df)} class-term entries)")
    return pd.DataFrame(summaries, columns=COMPARISON_COLUMNS)


def run_all_audits(
    partitioned_df: pd.DataFrame, cfg: Dict[str, Any], models_dir: Path, results_dir: Path,
) -> pd.DataFrame:
    """Audit saved models sequentially and package all current audit outputs."""
    if "partition_id" not in partitioned_df:
        raise ValueError("All-model audit requires partition_id to exclude the test set")
    # make_partitions retains excluded rows in the pickle and leaves their
    # partition_id blank. Validate the eligible rows, not the whole saved table.
    if "parse_status" in partitioned_df:
        bad_status = ~partitioned_df["parse_status"].isin(["ok", "excluded"])
        if bad_status.any():
            values = partitioned_df.loc[bad_status, "parse_status"].value_counts(dropna=False).head(10).to_dict()
            raise ValueError(f"Unknown or missing parse_status in audit input: {values}")
        retained = partitioned_df.loc[partitioned_df["parse_status"] == "ok"]
    else:
        # Compatibility with callers that already provide a retained-only table.
        retained = partitioned_df
    bad_partition = ~retained["partition_id"].isin(["train", "validation", "test"])
    if bad_partition.any():
        values = retained.loc[bad_partition, "partition_id"].value_counts(dropna=False).head(10).to_dict()
        raise ValueError(
            f"Unknown or missing partition_id on retained rows ({int(bad_partition.sum())}): {values}"
        )
    development = retained.loc[
        retained["partition_id"].isin(["train", "validation"])
    ].copy()
    _validate_records(development)
    logger.info(
        f"Audit input: {len(development)} train/validation records; "
        f"omitted {int(retained['partition_id'].eq('test').sum())} test records and "
        f"{len(partitioned_df) - len(retained)} excluded records"
    )
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    model_paths = {name: models_dir / f"{name}.joblib" for name in MODEL_NAMES}
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fitted audit models: " + ", ".join(missing))
    # One saved model in memory at a time; no fitting or prediction is needed.
    frames = []
    for name, path in model_paths.items():
        logger.info(f"Auditing saved {name}: {path}")
        pipeline = joblib.load(path)
        frames.append(run_audit(pipeline, development, cfg, results_dir, name))
        del pipeline
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(results_dir / "audit_comparison.csv", index=False, encoding="utf-8")
    combined = "\n\n".join(
        (results_dir / f"audit_summary_{name}.txt").read_text(encoding="utf-8")
        for name in MODEL_NAMES
    )
    (results_dir / "audit_summary_all_models.txt").write_text(combined, encoding="utf-8")
    archive_path = results_dir / "audit_results_all_models.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in AUDIT_OUTPUT_FILES:
            if filename != archive_path.name:
                archive.write(results_dir / filename, arcname=filename)
    logger.info(f"Saved {archive_path}")
    return comparison
