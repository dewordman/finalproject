# scripts/diagnose_confounds.py
"""Convenience wrapper: python scripts/diagnose_confounds.py --config ... --results-dir ..."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--stage", "diagnose"]
    sys.exit(main())
