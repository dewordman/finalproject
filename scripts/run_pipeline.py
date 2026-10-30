# scripts/run_pipeline.py
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.config import load_config, require
from src.parse import parse_all
from src.acquire import build_manifest
from src.external import run_external
from src.dedup import assign_duplicate_groups
from src.split import make_partitions, assert_partition_integrity
from src.train import run_grid, select_best, fit_final
from src.audit import AUDIT_OUTPUT_FILES, run_all_audits
from src.calibrate import run_calibration
from src.evaluate import run_final_test, run_dev_eval, evaluation_output_names
from src.diagnose import run_diagnose
from src.error_review import run_error_review
from src.experiments import run_source_tests, run_nazario_temporal
from src.variations import run_variations
from src.text_build import text_settings_from_cfg
from src.run_log import (init_run_log, start_stage, end_stage, finalise_run_log,
                         save_run_log, write_environment_freeze)
from src.results_index import generate_results_index
from src.utils import get_logger

logger = get_logger(__name__)
DEV_STAGES = ["parse", "dedup", "split", "grid", "select", "audit", "calibrate", "bundle", "dev_eval"]
STAGE_ORDER = DEV_STAGES + ["final_test", "external", "manifest", "diagnose", "source_tests",
                            "variations", "error_review", "temporal"]
LOCKED_STAGES = {"final_test", "temporal"}
DEVELOPMENT = "data/processed/records_development.pkl"
STAGE_REQUIRED_INPUTS: Dict[str, List[str]] = {
    "parse": [],
    "dedup": ["data/processed/records_parsed.pkl"],
    "split": ["data/processed/records_grouped.pkl"],
    "grid": [DEVELOPMENT],
    "select": ["results/grid_results.csv", DEVELOPMENT],
    "audit": [DEVELOPMENT, "models/logreg.joblib", "models/nb.joblib", "models/svm.joblib"],
    "calibrate": [DEVELOPMENT, "models/selected_config.json", "models/logreg.joblib"],
    "bundle": ["models/logreg.joblib", "models/calibration_choice.json"],
    "dev_eval": [DEVELOPMENT, "models/selected_config.json", "models/logreg.joblib",
                 "models/nb.joblib", "models/svm.joblib", "models/calibration_choice.json"],
    "final_test": ["data/processed/records_partitioned.pkl", "models/logreg.joblib",
                   "models/nb.joblib", "models/svm.joblib", "models/calibration_choice.json"],
    "index": [],
    "manifest": [],
    "external": ["data/processed/duplicate_reference.joblib"],
    "diagnose": ["data/processed/records_parsed.pkl"],
    "source_tests": [DEVELOPMENT, "models/selected_config.json"],
    "variations": [DEVELOPMENT, "models/logreg.joblib"],
    "error_review": [DEVELOPMENT, "models/logreg.joblib", "models/calibration_choice.json",
                     "models/logreg_calibrated.joblib"],
    "temporal": ["data/processed/records_external.pkl", "models/logreg.joblib",
                 "models/calibration_choice.json", "models/logreg_calibrated.joblib"],
}
STAGE_OUTPUTS: Dict[str, List[str]] = {
    "parse": ["data/processed/records_parsed.pkl", "results/parse_report.csv"],
    "dedup": ["data/processed/records_grouped.pkl", "data/processed/duplicate_reference.joblib",
              "results/dedup_report.csv", "results/near_dup_review_sample.csv"],
    "split": ["data/processed/records_partitioned.pkl", DEVELOPMENT,
              "results/partition_report.csv", "results/split_manifest.csv"],
    "grid": ["results/grid_results.csv"],
    "select": ["models/selected_config.json", "models/logreg.joblib", "models/nb.joblib", "models/svm.joblib"],
    "audit": [f"results/{name}" for name in AUDIT_OUTPUT_FILES],
    "calibrate": ["results/calibration_comparison.csv", "results/reliability_raw.csv",
        "results/reliability_sigmoid.csv", "results/reliability_raw.png", "results/reliability_sigmoid.png",
        "models/calibration_choice.json", "models/logreg_calibrated.joblib"],
    "bundle": ["models/final_bundle.joblib"],
    "dev_eval": [f"results/{name}" for name in evaluation_output_names("dev_")],
    "final_test": [f"results/{name}" for name in evaluation_output_names()],
    "index": ["results/RESULTS_INDEX.md"],
    "manifest": ["data/processed/manifest.csv"],
    "external": ["data/processed/records_external.pkl", "results/external_report.csv"],
    "diagnose": ["results/era_source_table.csv", "results/header_remnant_report.csv"],
    "source_tests": ["results/source_tests_summary.csv"],
    "variations": ["results/variation_pairs.csv", "results/variation_report.csv"],
    "error_review": ["results/error_review_sheet.csv"],
    "temporal": ["results/source_test_nazario_temporal.csv",
                 "results/confusion_matrix_nazario_temporal.csv",
                 "results/nazario_temporal_missed_terms.csv",
                 "results/nazario_temporal_predictions.csv"],
}


def _path_from_root(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _configure_run(cfg: Dict[str, Any], project_root: Path, cfg_path: Path,
                   results_override: str | None = None) -> Dict[str, Any]:
    """Bind this run to a result basename, configuration digest and private models."""
    effective = copy.deepcopy(cfg)
    result_value = results_override if results_override is not None else require(cfg, "paths.results_dir")
    results = _path_from_root(project_root, result_value)
    run_tag = results.name
    if not run_tag:
        raise ValueError("The results directory must have a non-empty basename")
    for key in ("raw_dir", "processed_dir", "models_dir", "configs_dir", "logs_dir", "locks_dir"):
        effective["paths"][key] = str(_path_from_root(project_root, require(cfg, f"paths.{key}")))
    if "nazario" in effective:
        for key in ("main_dir", "external_dir"):
            effective["nazario"][key] = str(_path_from_root(project_root, require(cfg, f"nazario.{key}")))
    locks = Path(effective["paths"]["locks_dir"])
    if locks.is_relative_to(results):
        raise ValueError("paths.locks_dir must be outside the results directory")
    effective["paths"]["results_dir"] = str(results)
    for key in ("processed_dir", "models_dir"):
        effective["paths"][key] = str(Path(effective["paths"][key]) / run_tag)
    results.mkdir(parents=True, exist_ok=True)
    owner = {"results_dir": str(results), "run_tag": run_tag,
             "config_digest": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
             "paths": effective["paths"]}
    owner_paths = [results / "run_context.json",
                   Path(effective["paths"]["processed_dir"]) / "run_context.json",
                   Path(effective["paths"]["models_dir"]) / "run_context.json"]
    # Check all namespaces before claiming any: equal basenames cannot collide.
    for path in owner_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous != owner:
                raise ValueError(f"Run namespace already belongs to another config/results path: {path}")
        elif path.parent.exists() and any(path.parent.iterdir()):
            raise FileExistsError(f"Unowned output directory is not empty: {path.parent}; choose a fresh basename")
    for path in owner_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                json.dump(owner, handle, indent=2)
    effective["_config_path"] = str(cfg_path)
    effective["_project_root"] = str(project_root)
    effective["_run_tag"] = run_tag
    return effective


def _artifact_path(project_root: Path, relative: str, cfg: Dict[str, Any] | None = None) -> Path:
    if cfg is not None:
        for prefix, key in (("data/processed/", "processed_dir"), ("results/", "results_dir"),
                            ("models/", "models_dir")):
            if relative.startswith(prefix):
                return Path(require(cfg, f"paths.{key}")) / relative[len(prefix):]
    return project_root / relative


def _check_inputs(project_root: Path, stage: str, cfg: Dict[str, Any] | None = None) -> bool:
    missing = [str(_artifact_path(project_root, name, cfg))
               for name in STAGE_REQUIRED_INPUTS.get(stage, [])
               if not _artifact_path(project_root, name, cfg).is_file()]
    if missing:
        logger.error("Stage '%s' cannot run. Missing required inputs:\n  %s", stage, "\n  ".join(missing))
        return False
    return True


def _protect_stage_outputs(project_root: Path, stage: str, cfg: Dict[str, Any]) -> None:
    existing = [str(_artifact_path(project_root, name, cfg))
                for name in STAGE_OUTPUTS.get(stage, [])
                if _artifact_path(project_root, name, cfg).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing stage outputs. Choose a fresh results basename: "
                              + ", ".join(existing))


def _load_development(processed_dir: Path) -> pd.DataFrame:
    records = pd.read_pickle(processed_dir / "records_development.pkl")
    if not records["partition_id"].isin(["train", "validation"]).all():
        raise AssertionError("Development file must contain training and validation rows only")
    return records


def _build_bundle(cfg: Dict[str, Any], models_dir: Path, cfg_path: Path) -> None:
    pipeline = joblib.load(models_dir / "logreg.joblib")
    with (models_dir / "calibration_choice.json").open("r", encoding="utf-8") as handle:
        method = json.load(handle).get("choice")
    if method not in ("raw", "sigmoid"):
        raise ValueError("calibration_choice.json must specify raw or sigmoid")
    calibrator = joblib.load(models_dir / "logreg_calibrated.joblib") if method == "sigmoid" else None
    packages = {"python": platform.python_version()}
    for package in ("scikit-learn", "pandas", "numpy", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unknown"
    digest = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle = {"pipeline": pipeline, "calibrator": calibrator, "calibration_method": method,
        "class_map": {0: "Legitimate", 1: "Spam", 2: "Phishing"},
        **text_settings_from_cfg(cfg),
        "max_input_chars": require(cfg, "text.max_input_chars"), "model_version": f"{timestamp}_{digest}",
        "config_digest": digest, "packages": packages,
        "created_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump(bundle, models_dir / "final_bundle.joblib")


def _execute_stage(stage: str, cfg: Dict[str, Any], cfg_path: Path,
                   project_root: Path, force: bool, reason: str | None) -> None:
    processed = Path(require(cfg, "paths.processed_dir"))
    results = Path(require(cfg, "paths.results_dir"))
    models = Path(require(cfg, "paths.models_dir"))
    if stage == "parse":
        parse_all(cfg)
    elif stage == "manifest":
        build_manifest(Path(require(cfg, "paths.raw_dir")), processed / "manifest.csv", cfg=cfg)
    elif stage == "external":
        run_external(cfg)
    elif stage == "dedup":
        assign_duplicate_groups(pd.read_pickle(processed / "records_parsed.pkl"), cfg)
    elif stage == "split":
        records, _ = make_partitions(pd.read_pickle(processed / "records_grouped.pkl"), cfg)
        assert_partition_integrity(records)
        development = records.loc[records["partition_id"].isin(["train", "validation"])].copy()
        development.to_pickle(processed / "records_development.pkl")
    elif stage in ("grid", "select", "audit", "calibrate"):
        records = _load_development(processed)
        train = records.loc[records["partition_id"].eq("train")].copy()
        validation = records.loc[records["partition_id"].eq("validation")].copy()
        if stage == "grid":
            run_grid(train, validation, cfg)
        elif stage == "select":
            selected = select_best(pd.read_csv(results / "grid_results.csv"), cfg)
            fit_final(train, selected, cfg)
        elif stage == "audit":
            run_all_audits(records, cfg, models, results)
        else:
            run_calibration(train, validation, cfg)
    elif stage == "bundle":
        _build_bundle(cfg, models, cfg_path)
    elif stage == "dev_eval":
        run_dev_eval(cfg, cfg_path)
    elif stage == "final_test":
        run_final_test(cfg, cfg_path, force=force, reason=reason)
    elif stage == "diagnose":
        run_diagnose(cfg)
    elif stage == "source_tests":
        run_source_tests(cfg)
    elif stage == "variations":
        run_variations(cfg)
    elif stage == "error_review":
        run_error_review(cfg)
    elif stage == "temporal":
        run_nazario_temporal(cfg, cfg_path, force=force, reason=reason)
    elif stage == "index":
        generate_results_index(project_root, results / "RESULTS_INDEX.md", paths=cfg["paths"])
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated email-classification stages")
    parser.add_argument("--stage", choices=STAGE_ORDER + ["all", "index"], default="parse")
    parser.add_argument("--config", default="config.yaml", metavar="PATH")
    parser.add_argument("--results-dir", default=None, metavar="PATH")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.force and (args.stage not in LOCKED_STAGES or not args.reason or not args.reason.strip()):
        logger.error("--force requires --stage final_test or --stage temporal and a non-blank --reason")
        return 1
    cfg_path = _path_from_root(project_root, args.config)
    try:
        cfg = _configure_run(load_config(cfg_path), project_root, cfg_path, args.results_dir)
    except Exception as exc:
        logger.error("Run configuration failed: %s", exc)
        return 1
    write_environment_freeze(Path(cfg["paths"]["results_dir"]))
    run_log = init_run_log(cfg_path, seed=require(cfg, "seed"))
    cfg["_run_id"] = run_log["run_id"]
    log_path = Path(cfg["paths"]["results_dir"]) / "run_logs" / f"{run_log['run_id']}.json"
    stages = DEV_STAGES + ["index"] if args.stage == "all" else [args.stage]
    exit_code = 0
    try:
        for stage in stages:
            logger.info("STAGE: %s", stage)
            run_log = start_stage(run_log, stage)
            try:
                if not _check_inputs(project_root, stage, cfg):
                    raise FileNotFoundError(f"Stage '{stage}' has missing inputs")
                _protect_stage_outputs(project_root, stage, cfg)
                _execute_stage(stage, cfg, cfg_path, project_root, args.force, args.reason)
                outputs = [str(_artifact_path(project_root, name, cfg))
                           for name in STAGE_OUTPUTS.get(stage, [])
                           if _artifact_path(project_root, name, cfg).is_file()]
                run_log = end_stage(run_log, outputs=outputs, status="ok")
            except Exception as exc:
                logger.exception("Stage '%s' failed: %s", stage, exc)
                run_log = end_stage(run_log, outputs=[], status="failed")
                exit_code = 1
                break
    finally:
        save_run_log(finalise_run_log(run_log), log_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
