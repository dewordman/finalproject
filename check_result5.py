"""
check_results_rev5.py
=====================
Checks what files exist in results_rev5 and identifies what's missing
based on the stages you've already run.
"""

from pathlib import Path

results_dir = Path("results_rev5")

if not results_dir.exists():
    print(f"❌ results_rev5 folder does not exist!")
    exit(1)

print("=" * 70)
print("DIAGNOSTIC: Checking results_rev5 folder")
print("=" * 70)

# Expected files based on stages you've run
expected_files = {
    "parse": [
        "records_parsed.pkl",
        "records_manifest.csv",
    ],
    "manifest": [
        "records_manifest.csv",
    ],
    "diagnose": [
        "era_source_table.csv",
        "header_remnant_report.csv",
    ],
    "dedup": [
        "records_deduped.pkl",
    ],
    "split": [
        "split_manifest.json",
    ],
    "grid": [
        "grid_results.csv",
    ],
    "select": [
        "selected_config.json",
    ],
    "audit": [
        "audit_top_terms_logreg.csv",
        "audit_top_terms_nb.csv",
        "audit_top_terms_svm.csv",
        "audit_source_share_logreg.csv",
        "audit_flagged_terms_logreg.csv",
        "audit_summary_logreg.txt",
        "audit_summary_nb.txt",
        "audit_summary_svm.txt",
    ],
    "calibrate": [
        "calibration_comparison.csv",
        "calibration_choice.json",
    ],
    "bundle": [
        "pipeline.joblib",
    ],
    "dev_eval": [
        "dev_final_summary.json",
        "dev_overall_metrics.csv",
        "dev_per_class_metrics_logreg.csv",
        "dev_per_class_metrics_nb.csv",
        "dev_per_class_metrics_svm.csv",
        "dev_confusion_matrix_logreg.csv",
        "dev_confusion_matrix_nb.csv",
        "dev_confusion_matrix_svm.csv",
    ],
    "final_test": [
        "test_final_summary.json",
        "test_overall_metrics.csv",
        "test_per_class_metrics_logreg.csv",
        "test_confusion_matrix_logreg.csv",
    ],
    "temporal": [
        "temporal_summary.json",
        "temporal_by_year.csv",
    ],
    "error_review": [
        "error_review_sheet.csv",
    ],
}

# Check each stage
missing_by_stage = {}
found_by_stage = {}

for stage, files in expected_files.items():
    missing = []
    found = []
    for file in files:
        file_path = results_dir / file
        if file_path.exists():
            found.append(file)
        else:
            missing.append(file)
    
    if found:
        found_by_stage[stage] = found
    if missing:
        missing_by_stage[stage] = missing

# Report findings
print("\n✅ FOUND FILES:")
for stage, files in found_by_stage.items():
    print(f"\n  {stage}:")
    for f in files:
        print(f"    ✓ {f}")

print("\n" + "=" * 70)
print("❌ MISSING FILES:")
print("=" * 70)

if not missing_by_stage:
    print("\n✅ All expected files are present!")
else:
    for stage, files in missing_by_stage.items():
        print(f"\n  {stage}:")
        for f in files:
            print(f"    ✗ {f}")

# Check for models subfolder
models_dir = results_dir / "models"
if models_dir.exists():
    model_files = list(models_dir.glob("*.joblib"))
    print(f"\n📦 Models folder contains {len(model_files)} files:")
    for f in model_files:
        print(f"    ✓ {f.name}")
else:
    print(f"\n❌ Models folder does not exist!")

# Check for run_logs
logs_dir = results_dir / "run_logs"
if logs_dir.exists():
    log_files = list(logs_dir.glob("*.json"))
    print(f"\n📝 Run logs folder contains {len(log_files)} files")
else:
    print(f"\n❌ Run logs folder does not exist!")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if missing_by_stage:
    print("\nYou need to re-run the following stages:")
    for stage in missing_by_stage.keys():
        print(f"  - {stage}")
    
    print("\n⚠️  IMPORTANT: final_test and temporal stages have locks.")
    print("   You'll need to use --force --reason for those stages.")
else:
    print("\n✅ Your results_rev5 folder is complete!")