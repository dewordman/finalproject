# tests/test_task5.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import src.run_log as run_log
from src.decision import load_decision_table
from src.results_index import FILE_TO_SECTION, generate_results_index

ROOT = Path(__file__).resolve().parents[1]


def test_package_marker_is_named_correctly():
    assert (ROOT / "src" / "__init__.py").exists()
    assert not (ROOT / "src" / "_init_.py").exists()


def test_decision_table_is_clean_valid_and_has_no_duplicate_keys():
    path = ROOT / "configs" / "decision_table.yaml"

    class NoDuplicates(yaml.SafeLoader):
        pass

    def construct(loader, node, deep=False):
        keys = [loader.construct_object(key, deep=deep) for key, _ in node.value]
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
        return yaml.SafeLoader.construct_mapping(loader, node, deep)
    NoDuplicates.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
    yaml.load(path.read_text(encoding="utf-8"), Loader=NoDuplicates)
    table = load_decision_table(path)
    assert isinstance(table["t_low"], float) and isinstance(table["t_high"], float)
    assert table["t_low"] < table["t_high"]
    assert "intentionally left null" not in path.read_text(encoding="utf-8")


def test_run_log_reports_real_versions():
    versions = run_log._package_versions()
    for package in ("pandas", "numpy", "scikit-learn", "PyYAML", "pytest"):
        assert versions[package] not in ("unknown", "not installed", ""), package
    assert versions["python"].count(".") == 2


def test_environment_freeze_is_written_once_and_never_overwritten(tmp_path):
    first = run_log.write_environment_freeze(tmp_path / "results")
    assert first.name == "environment_freeze.txt" and first.read_text(encoding="utf-8").startswith("# python ")
    first.write_text("# my edit\n", encoding="utf-8")
    again = run_log.write_environment_freeze(tmp_path / "results")
    assert again.read_text(encoding="utf-8") == "# my edit\n"


def test_results_index_lists_new_outputs_and_drops_partial_experiment(tmp_path):
    for name in ("era_source_table.csv", "header_remnant_report.csv", "mask_term_candidates.csv",
                 "source_test_nazario_temporal.csv", "confusion_matrix_nazario_temporal.csv",
                 "variation_report.csv", "error_review_sheet.csv", "environment_freeze.txt"):
        assert f"results/{name}" in FILE_TO_SECTION, name
    assert "results/source_test_nazario_partial.csv" not in FILE_TO_SECTION
    cfg = yaml.safe_load((ROOT / "tests" / "config_template.yaml").read_text(encoding="utf-8"))
    paths = {key: str(tmp_path / value) for key, value in cfg["paths"].items()}
    Path(paths["results_dir"]).mkdir(parents=True)
    out = Path(paths["results_dir"]) / "RESULTS_INDEX.md"
    generate_results_index(tmp_path, out, paths=paths)
    text = out.read_text(encoding="utf-8")
    assert "disabled pending" not in text and "temporal" in text and "development data only" in text


def test_readme_has_balanced_fences_and_the_required_workflow():
    text = (ROOT / "README_RUN.md").read_text(encoding="utf-8")
    assert text.count("```") % 2 == 0
    for needle in ("manifest", "parse", "diagnose", "dedup", "split", "grid", "select", "audit",
                   "suggest_mask_terms", "calibrate", "bundle", "dev_eval", "final_test", "--force",
                   "--reason", "source_tests", "temporal", "software_tests", "variations", "error_review", "index",
                   "locks", "revision_log"):
        assert needle in text, needle
    assert "Remove-Item" not in text


def test_change_log_is_filled_in():
    text = (ROOT / "docs" / "CHANGES_REV2.md").read_text(encoding="utf-8")
    assert len(text.split()) > 300
    for needle in ("Chapter Three", "delete", "temporal", "lock", "dev_eval"):
        assert needle in text


def test_config_has_no_duplicate_keys_and_required_nulls():
    class NoDuplicates(yaml.SafeLoader):
        pass

    def construct(loader, node, deep=False):
        keys = [loader.construct_object(key, deep=deep) for key, _ in node.value]
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
        return yaml.SafeLoader.construct_mapping(loader, node, deep)
    NoDuplicates.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
    cfg = yaml.load((ROOT / "tests" / "config_template.yaml").read_text(encoding="utf-8"), Loader=NoDuplicates)
    assert cfg["selection"] == {"min_phishing_recall": None, "max_legit_fpr": None}
    assert (ROOT / "config_rev2.yaml").exists()
    assert cfg["text"]["mask_mode"] == "delete" and cfg["evaluation"]["variation_per_class"] == 150
    assert "variation_sample_size" not in cfg["evaluation"]
