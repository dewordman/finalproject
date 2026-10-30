"""
rebuild_results_rev5.py
=======================
Re-runs missing stages to rebuild results_rev5 folder.
"""

import subprocess
import sys

config = "config_rev5.yaml"
results_dir = "results_rev5"

# Stages to run (in order)
stages = [
    "parse",
    "manifest", 
    "diagnose",
    "dedup",
    "split",
    "grid",
    "select",
    "audit",
    "calibrate",
    "bundle",
    "dev_eval",
    "source_tests",
    "variations",
    "error_review",
]

# Stages that need --force (already have locks)
force_stages = ["final_test", "temporal"]

print("=" * 70)
print("REBUILDING results_rev5")
print("=" * 70)

# Run normal stages
for stage in stages:
    print(f"\n{'='*70}")
    print(f"Running stage: {stage}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--config", config,
        "--results-dir", results_dir,
        "--stage", stage
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Stage '{stage}' failed!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

# Run force stages
for stage in force_stages:
    print(f"\n{'='*70}")
    print(f"Running stage: {stage} (with --force)")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--config", config,
        "--results-dir", results_dir,
        "--stage", stage,
        "--force",
        "--reason", "Rebuilding results_rev5 after index stage cleanup"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Stage '{stage}' failed!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

print("\n" + "=" * 70)
print("✅ REBUILD COMPLETE")
print("=" * 70)
print(f"\nAll stages have been re-run for {results_dir}")
print("Run 'python check_results_rev5.py' to verify all files are present.")