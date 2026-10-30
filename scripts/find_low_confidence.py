# scripts/find_low_confidence.py
"""Find REAL validation messages the model is least sure about, to demonstrate test T08 in the app.

Reads <results>/dev_predictions_logreg.csv (validation predictions only) and the development records of
the same run, then writes the lowest-confidence messages to a text file you can paste into the app.
The sealed test partition is never read. The messages come from public corpora; do not print them in the thesis.

  python scripts/find_low_confidence.py --results-dir results_rev5 --n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

CLASSES = {0: "Legitimate", 1: "Spam", 2: "Phishing"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List the lowest-confidence validation messages")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--processed-dir", help="default: data/processed/<results folder name>")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--below", type=float, default=0.4, help="the t_low of the decision table in use")
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    results = results if results.is_absolute() else ROOT / results
    processed = Path(args.processed_dir) if args.processed_dir else ROOT / "data" / "processed" / results.name
    predictions_path = results / "dev_predictions_logreg.csv"
    records_path = processed / "records_development.pkl"
    for path in (predictions_path, records_path):
        if not path.exists():
            print(f"ERROR: not found: {path}")
            return 2
    predictions = pd.read_csv(predictions_path)
    if not predictions["partition_id"].eq("validation").all():
        print("ERROR: the predictions file must contain validation rows only")
        return 2
    lowest = predictions.nsmallest(args.n, "confidence")
    records = pd.read_pickle(records_path).set_index("record_id")
    out = results / "low_confidence_examples.txt"
    if out.exists():
        print(f"ERROR: {out} already exists; remove or rename it first")
        return 2
    blocks = []
    for _, row in lowest.iterrows():
        record = records.loc[row["record_id"]]
        blocks.append(
            f"=== {row['record_id']} | source {row['source_dataset']} | true {CLASSES[int(row['true_class'])]} | "
            f"predicted {CLASSES[int(row['predicted_class'])]} | confidence {row['confidence']:.4f}\n"
            f"SUBJECT: {record['subject_raw']}\nBODY:\n{record['body_raw']}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(lowest[["record_id", "source_dataset", "true_class", "predicted_class", "confidence"]].to_string(index=False))
    n_low = int((predictions["confidence"] < args.below).sum())
    print(f"\n{n_low} of {len(predictions)} validation predictions fall below t_low={args.below}.")
    print(f"Messages written to {out}. Paste one with confidence below {args.below} into the app for the T08 screenshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
