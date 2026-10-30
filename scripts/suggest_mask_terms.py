# scripts/suggest_mask_terms.py
"""Print candidate mask_terms from an audit; NEVER edits any config file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, require
from src.mask_suggest import suggest_terms, yaml_list


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest mask_terms from the audit (read-only)")
    parser.add_argument("--config", default="config_rev2.yaml")
    parser.add_argument("--results-dir", required=True,
                        help="Results directory that holds audit_flagged_terms_logreg.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = Path(args.results_dir)
    candidates = suggest_terms(results / "audit_flagged_terms_logreg.csv",
                               float(require(cfg, "audit.dominance_threshold")),
                               results / "mask_term_candidates.csv")
    print(yaml_list(candidates))
    print(f"\nSaved {results / 'mask_term_candidates.csv'}. Review by hand; "
          "nothing was written to any config file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
