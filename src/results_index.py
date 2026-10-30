# src/results_index.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.utils import get_logger
from src.audit import AUDIT_OUTPUT_FILES
from src.evaluate import evaluation_output_names

logger = get_logger(__name__)

# Mapping of output file → Chapter 5 section
FILE_TO_SECTION: Dict[str, str] = {
    # Stage 1: Parse
    "results/parse_report.csv": "5.1 Dataset Acquisition and Preparation",
    "data/processed/manifest.csv": "5.1 Dataset Acquisition and Preparation",
    # Stage 2: Dedup
    "results/dedup_report.csv": "5.2 Data Cleaning and Deduplication",
    "results/near_dup_review_sample.csv": "5.2 Data Cleaning and Deduplication",
    # Stage 3: Split
    "results/partition_report.csv": "5.3 Partitioning",
    "results/split_manifest.csv": "5.3 Partitioning",
    # Stage 4–5: Grid and Select
    "results/grid_results.csv": "5.4 Model Selection and Training",
    "models/selected_config.json": "5.4 Model Selection and Training",
    "models/logreg.joblib": "5.4 Model Selection and Training",
    "models/nb.joblib": "5.4 Model Selection and Training",
    "models/svm.joblib": "5.4 Model Selection and Training",
    # Stage 6: Audit
    "results/audit_top_terms.csv": "5.5 Source-Cue Audit",
    "results/audit_source_share.csv": "5.5 Source-Cue Audit",
    "results/audit_flagged_terms.csv": "5.5 Source-Cue Audit",
    "results/audit_summary.txt": "5.5 Source-Cue Audit",
    # Stage 7: Calibration
    "results/calibration_comparison.csv": "5.6 Probability Calibration",
    "results/reliability_raw.csv": "5.6 Probability Calibration",
    "results/reliability_sigmoid.csv": "5.6 Probability Calibration",
    "results/reliability_raw.png": "5.6 Probability Calibration",
    "results/reliability_sigmoid.png": "5.6 Probability Calibration",
    "models/calibration_choice.json": "5.6 Probability Calibration",
    "models/logreg_calibrated.joblib": "5.6 Probability Calibration",
    # Stage 8: Final Test
    "results/final_summary.json": "5.7 Final Evaluation",
    "results/test_lock.json": "5.7 Final Evaluation",
    "results/test_predictions_logreg.csv": "5.7 Final Evaluation",
    "results/test_predictions_nb.csv": "5.7 Final Evaluation",
    "results/test_predictions_svm.csv": "5.7 Final Evaluation",
    "results/confusion_matrix_logreg.csv": "5.7 Final Evaluation",
    "results/confusion_matrix_nb.csv": "5.7 Final Evaluation",
    "results/confusion_matrix_svm.csv": "5.7 Final Evaluation",
    "results/confusion_matrix_logreg.png": "5.7 Final Evaluation",
    "results/confusion_matrix_nb.png": "5.7 Final Evaluation",
    "results/confusion_matrix_svm.png": "5.7 Final Evaluation",
    "results/confusion_matrix_source_baseline.csv": "5.7 Final Evaluation",
    "results/confusion_matrix_source_baseline.png": "5.7 Final Evaluation",
    "results/per_class_metrics_logreg.csv": "5.7 Final Evaluation",
    "results/per_class_metrics_nb.csv": "5.7 Final Evaluation",
    "results/per_class_metrics_svm.csv": "5.7 Final Evaluation",
    "results/reliability_logreg_test.csv": "5.7 Final Evaluation",
    # Stage 9: Source Tests
    "results/source_tests_summary.csv": "5.8 Source-Conditioned Tests",
    "results/source_test_holdout_enron.csv": "5.8 Source-Conditioned Tests",
    "results/source_test_holdout_spamassassin.csv": "5.8 Source-Conditioned Tests",
    "results/source_test_nazario_partial.csv": "5.8 Source-Conditioned Tests",
    "results/confusion_matrix_holdout_enron.csv": "5.8 Source-Conditioned Tests",
    "results/confusion_matrix_holdout_spamassassin.csv": "5.8 Source-Conditioned Tests",
    "results/confusion_matrix_holdout_enron_baseline.csv": "5.8 Source-Conditioned Tests",
    "results/confusion_matrix_holdout_spamassassin_baseline.csv": "5.8 Source-Conditioned Tests",
    "results/confusion_matrix_nazario_partial.csv": "5.8 Source-Conditioned Tests",
    # Stage 10: Variations
    "results/variation_report.csv": "5.9 Surface Variation Sensitivity",
    "results/variation_pairs.csv": "5.9 Surface Variation Sensitivity",
    # Stage 11: Error Review
    "results/error_review_sheet.csv": "5.10 Error Review and Coding",
    # Stage 12: Bundle / Deployment
    "models/final_bundle.joblib": "5.11 Deployment Bundle",
    "configs/decision_table.yaml": "5.11 Deployment Bundle",
    "logs/predictions.csv": "5.11 Deployment Bundle",
    # Meta
    "results/run_log.json": "Appendix A: Run Log",
    "results/revision_log.json": "Appendix B: Revision Log",
}

# Keep the all-model audit files aligned with the runner's output registry.
FILE_TO_SECTION.update({
    f"results/{filename}": "5.5 Source-Cue Audit"
    for filename in AUDIT_OUTPUT_FILES
})
FILE_TO_SECTION.update({
    f"results/{filename}": "5.6 Development Evaluation (Validation Only)"
    for filename in evaluation_output_names("dev_")
})
FILE_TO_SECTION.update({
    "locks/final_test_lock.json": "Appendix B: Global Final-Test Lock",
    "locks/revision_log.json": "Appendix B: Append-Only Revision Journal",
    "results/run_context.json": "Appendix A: Run Context",
    "data/processed/duplicate_reference.joblib": "5.2 Data Cleaning and Deduplication",
    "data/processed/records_external.pkl": "5.8 External Temporal Corpus",
    "results/external_report.csv": "5.8 External Temporal Corpus",
})
FILE_TO_SECTION.update({
    "results/era_source_table.csv": "5.1 Confound Diagnostics (source, era, e-mail debris)",
    "results/header_remnant_report.csv": "5.1 Confound Diagnostics (source, era, e-mail debris)",
    "results/mask_term_candidates.csv": "5.5 Source-Cue Audit (candidate mask terms)",
    "results/environment_freeze.txt": "Appendix A: Environment (pip freeze)",
    "results/source_test_holdout_enron.csv": "5.8 Source-Conditioned Tests (development data)",
    "results/source_test_holdout_spamassassin.csv": "5.8 Source-Conditioned Tests (development data)",
    "results/source_tests_summary.csv": "5.8 Source-Conditioned Tests (development data)",
    "results/source_test_nazario_temporal.csv": "5.8 Temporal Generalisation (external Nazario files)",
    "results/confusion_matrix_nazario_temporal.csv": "5.8 Temporal Generalisation (external Nazario files)",
    "results/nazario_temporal_missed_terms.csv": "5.8 Temporal Generalisation (external Nazario files)",
    "results/nazario_temporal_predictions.csv": "5.8 Temporal Generalisation (external Nazario files)",
    "results/variation_report.csv": "5.9 Surface Variation Sensitivity (validation data)",
    "results/variation_pairs.csv": "5.9 Surface Variation Sensitivity (validation data)",
    "results/error_review_sheet.csv": "5.10 Error Review and Coding (validation data)",
    "locks/nazario_temporal_lock.json": "Appendix B: Temporal-Test Lock",
})
for _stale in ("results/source_test_nazario_partial.csv",
               "results/confusion_matrix_nazario_partial.csv"):
    FILE_TO_SECTION.pop(_stale, None)


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_results_index(project_root: Path, out_path: Path,
                           *, paths: Optional[Dict[str, str]] = None) -> None:
    """Scan the project and write a Markdown index of every output file."""
    if out_path.exists():
        raise FileExistsError(f"Results index already exists: {out_path}")
    rows: List[Tuple[str, str, str, str, str]] = []

    for rel_path, section in FILE_TO_SECTION.items():
        full_path = project_root / rel_path
        if paths is not None:
            for prefix, key in (("data/processed/", "processed_dir"), ("results/", "results_dir"),
                                ("models/", "models_dir"), ("configs/", "configs_dir"),
                                ("logs/", "logs_dir"), ("locks/", "locks_dir")):
                if rel_path.startswith(prefix):
                    full_path = Path(paths[key]) / rel_path[len(prefix):]
                    break
        if full_path.exists():
            stat = full_path.stat()
            size = _human_size(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            status = "present"
        else:
            size = "-"
            mtime = "-"
            status = "missing"
        rows.append((str(full_path) if paths is not None else rel_path, section, status, size, mtime))

    if paths is not None:
        for path in sorted((Path(paths["results_dir"]) / "run_logs").glob("*.json")):
            stat = path.stat()
            rows.append((str(path), "Appendix A: Run Log", "present", _human_size(stat.st_size),
                         datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")))

    # Sort by section then by path
    rows.sort(key=lambda r: (r[1], r[0]))

    lines = [
        "# Results Index",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "This file maps every pipeline output to the Chapter Five section it feeds.",
        "It is generated by `--stage index`; existing indexes are preserved.",
        "",
        "| Output file | Chapter 5 section | Status | Size | Modified |",
        "|---|---|---|---|---|",
    ]
    for rel, section, status, size, mtime in rows:
        lines.append(f"| `{rel}` | {section} | {status} | {size} | {mtime} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Stage Execution Order")
    lines.append("")
    lines.append("1. `parse` — raw data → `data/processed/records_parsed.pkl`")
    lines.append("2. `dedup` — duplicate grouping → `records_grouped.pkl`")
    lines.append("3. `split` — train/val/test partitioning → `records_partitioned.pkl`")
    lines.append("4. `grid` — hyperparameter search → `grid_results.csv`")
    lines.append("5. `select` — pick best model → `models/*.joblib`")
    lines.append("6. `audit` — source-cue detection → `audit_*.csv`")
    lines.append("7. `calibrate` — Platt scaling → `calibration_choice.json`")
    lines.append("8. `bundle` — package for deployment → `final_bundle.joblib`")
    lines.append("9. `dev_eval` — selected models evaluated on validation only → `dev_*.csv`")
    lines.append("10. `final_test` — explicit sealed evaluation; global lock under `locks_dir`")
    lines.append("`manifest` is optional and inventories every raw file; `external` is optional and runs after main deduplication.")
    lines.append("External records are stored separately and never enter the main partitions.")
    lines.append("`diagnose`, `source_tests`, `variations` and `error_review` use development data only (training + validation).")
    lines.append("`temporal` scores the saved models once on the external Nazario files, under its own lock in `locks_dir`.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info(f"Saved results index to {out_path}")
