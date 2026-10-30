# scripts/software_tests.py
"""Run the Chapter Three software tests against the REAL final bundle and write an evidence table.

Covers the app-facing tests through ``src.inference.predict_email`` (the function the Streamlit app
calls) and the data/model checks on the real processed files:

  T01  valid inputs (legitimate, spam, phishing-style) -> status success, complete fields, explanation present
  T02  missing subject, valid body                       -> processed
  T03  subject and body both empty                       -> rejected with a message, no prediction
  T04  HTML, script tag and URL text                     -> processed as inert text, no exception
  T05  duplicate groups never cross partitions (real processed data)
  T06  feature audit: the model sees text only; masked terms in the vocabulary are listed
  T07  fresh-process reload gives the same prediction and the same terms
  T08  low-confidence routing and band boundaries (decision rules)
  T09  oversized input                                   -> rejected
  T10  source-conditioned tests never report a three-class macro score for a missing class
  LOG  optional: the prediction log stores hashes, never message text

Usage (PowerShell, project root):
  python scripts/software_tests.py --results-dir results_rev5 --table configs/decision_table.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.decision import decide, load_decision_table, lookup_action  # noqa: E402
from src.inference import load_bundle, predict_email  # noqa: E402

PHISHING = ("URGENT: Verify your account immediately to prevent suspension",
            "Dear customer, we detected unusual activity on your account. Verify your password and "
            "login details within 24 hours or your account access will be suspended. Click here to "
            "verify your account: https://example.com/secure-login")
LEGITIMATE = ("Project update and scheduled meeting",
              "Hi team, please find the agenda for Thursday's project review. We will go through the "
              "budget, the schedule and the open actions. Let me know if you cannot attend. Thanks.")
SPAM = ("Congratulations, you have won a free prize!",
        "You are our lucky winner! Claim your FREE gift card and cheap pills now. Limited offer, buy now "
        "and save 80 percent. Click to unsubscribe or order today. Guaranteed cash bonus!")
HTML_URL = ("Invoice <b>attached</b>",
            "<html><body><div><p>Dear user,</p><script>alert('x')</script><a href='http://bad.example.org/x'>"
            "Open</a> or visit www.bad.example.org/login now.</p></div></body></html>")
CLASSES = {"Legitimate", "Spam", "Phishing"}
BANDS = {"low", "moderate", "high"}

Row = Dict[str, str]


def _row(test_id: str, name: str, expected: str, actual: str, status: str, notes: str = "") -> Row:
    return {"test_id": test_id, "name": name, "expected": expected, "actual": actual,
            "status": status, "notes": notes}


def _safe(test_id: str, name: str, expected: str, func: Callable[[], Row]) -> Row:
    try:
        return func()
    except Exception as exc:  # a crash is a failed test, never a silent skip
        return _row(test_id, name, expected, f"exception: {type(exc).__name__}: {exc}", "FAIL")


def _valid_result_fields(result: Dict[str, Any]) -> List[str]:
    problems = []
    if result.get("status") != "success":
        return [f"status={result.get('status')}"]
    if result.get("predicted_class") not in CLASSES:
        problems.append("predicted_class invalid")
    confidence = result.get("confidence")
    if not isinstance(confidence, float) or not 0.0 <= confidence <= 1.0:
        problems.append("confidence not a float in [0,1]")
    if result.get("band") not in BANDS:
        problems.append("band invalid")
    for key in ("action", "model_version", "rule_version", "timestamp"):
        if not result.get(key):
            problems.append(f"{key} missing")
    return problems


def run_app_tests(bundle: Dict[str, Any], table: Dict[str, Any], bundle_path: Path,
                  table_path: Path) -> List[Row]:
    rows: List[Row] = []
    cases = {"T01a": ("valid phishing-style email", PHISHING), "T01b": ("valid legitimate email", LEGITIMATE),
             "T01c": ("valid spam-style email", SPAM)}
    results: Dict[str, Dict[str, Any]] = {}
    for test_id, (name, (subject, body)) in cases.items():
        def case(subject=subject, body=body, test_id=test_id, name=name) -> Row:
            result = predict_email(subject, body, bundle, table)
            results[test_id] = result
            problems = _valid_result_fields(result)
            actual = (f"{result.get('predicted_class')} {result.get('confidence')} band={result.get('band')} "
                      f"action={result.get('action')}")
            return _row(test_id, name, "status success with complete fields", actual,
                        "PASS" if not problems else "FAIL", "; ".join(problems))
        rows.append(_safe(test_id, name, "status success", case))
        def explain_case(test_id=test_id, name=name) -> Row:
            result = results.get(test_id, {})
            n = len(result.get("supporting_terms", []))
            terms = ", ".join(t["term"] for t in result.get("supporting_terms", [])[:5])
            return _row(f"{test_id}-explain", f"{name}: explanation present",
                        "at least one supporting term", f"{n} supporting terms: {terms}",
                        "PASS" if n >= 1 else "FAIL",
                        "" if n >= 1 else "No influential terms returned: the explanation component is not delivering evidence")
        rows.append(_safe(f"{test_id}-explain", name, "supporting terms", explain_case))

    def t02() -> Row:
        result = predict_email("", LEGITIMATE[1], bundle, table)
        ok = result.get("status") == "success" and not _valid_result_fields(result)
        return _row("T02", "missing subject, valid body", "processed (status success)",
                    f"status={result.get('status')} class={result.get('predicted_class')}", "PASS" if ok else "FAIL")
    rows.append(_safe("T02", "missing subject", "success", t02))

    def t03() -> Row:
        outcomes = [predict_email("", "", bundle, table), predict_email("   ", "\n\t ", bundle, table)]
        ok = all(o.get("status") == "rejected" and o.get("message") and "predicted_class" not in o for o in outcomes)
        return _row("T03", "subject and body both empty", "rejected with a message, no prediction",
                    "; ".join(f"status={o.get('status')}" for o in outcomes), "PASS" if ok else "FAIL")
    rows.append(_safe("T03", "both empty", "rejected", t03))

    def t04() -> Row:
        result = predict_email(*HTML_URL, bundle, table)
        ok = result.get("status") == "success" and not _valid_result_fields(result)
        return _row("T04", "HTML, script tag and URL text", "processed as inert text, no exception",
                    f"status={result.get('status')} class={result.get('predicted_class')}", "PASS" if ok else "FAIL",
                    "The code never renders HTML, follows links or runs scripts; nothing is fetched.")
    rows.append(_safe("T04", "html/url", "success", t04))

    def t07() -> Row:
        reference = predict_email(*PHISHING, bundle, table)
        code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path\n"
                "from src.inference import load_bundle, predict_email\nfrom src.decision import load_decision_table\n"
                "b = load_bundle(Path(sys.argv[2])); t = load_decision_table(Path(sys.argv[3]))\n"
                "r = predict_email(sys.argv[4], sys.argv[5], b, t)\n"
                "print('JSON>>' + json.dumps({k: r.get(k) for k in "
                "('status','predicted_class','confidence','band','action','supporting_terms')}))")
        done = subprocess.run([sys.executable, "-c", code, str(ROOT), str(bundle_path), str(table_path), *PHISHING],
                              capture_output=True, text=True, timeout=300)
        line = next((l for l in done.stdout.splitlines() if l.startswith("JSON>>")), None)
        if line is None:
            return _row("T07", "fresh-process reload", "same prediction after reload",
                        f"no output (return code {done.returncode}): {done.stderr[-200:]}", "FAIL")
        fresh = json.loads(line[len("JSON>>"):])
        same = all(fresh[k] == reference.get(k) for k in
                   ("status", "predicted_class", "confidence", "band", "action", "supporting_terms"))
        return _row("T07", "fresh-process reload", "same class, confidence and terms",
                    f"class={fresh['predicted_class']} confidence={fresh['confidence']}", "PASS" if same else "FAIL")
    rows.append(_safe("T07", "reload", "same prediction", t07))

    def t08() -> Row:
        t_low, t_high = float(table["t_low"]), float(table["t_high"])
        problems = []
        if decide(max(t_low - 0.001, 0.0), table) != "low":
            problems.append("below t_low is not 'low'")
        if decide(t_low, table) != "moderate":
            problems.append("t_low itself is not 'moderate'")
        if decide(t_high, table) != "high":
            problems.append("t_high itself is not 'high'")
        for name in ("Legitimate", "Spam", "Phishing"):
            if lookup_action(name, "low", table) != table["actions"][name]["low"]:
                problems.append(f"low-band action wrong for {name}")
            if "review" not in lookup_action(name, "low", table).lower():
                problems.append(f"low-band action for {name} is not a review action")
        return _row("T08", "low-confidence routing and band boundaries", "below t_low routes to Review",
                    f"t_low={t_low} t_high={t_high} actions={ {c: table['actions'][c]['low'] for c in CLASSES} }",
                    "PASS" if not problems else "FAIL", "; ".join(problems) or
                    "Decision rule tested directly; a natural low-confidence message may not exist for this model.")
    rows.append(_safe("T08", "low confidence", "Review", t08))

    def t09() -> Row:
        limit = int(bundle.get("max_input_chars", 20000))
        result = predict_email("Long message", "word " * (limit // 4), bundle, table)
        ok = result.get("status") == "rejected" and "exceeds" in str(result.get("message", "")).lower()
        return _row("T09", "oversized input", f"rejected (limit {limit} characters)",
                    f"status={result.get('status')} message={result.get('message')}", "PASS" if ok else "FAIL")
    rows.append(_safe("T09", "oversized", "rejected", t09))
    return rows


def run_data_tests(bundle: Dict[str, Any], processed_dir: Path, results_dir: Path) -> List[Row]:
    rows: List[Row] = []

    def t05() -> Row:
        path = processed_dir / "records_partitioned.pkl"
        if not path.exists():
            return _row("T05", "duplicate groups never cross partitions", "no group in two partitions",
                        f"missing {path}", "SKIPPED", "Run with --processed-dir pointing at the run's processed folder")
        frame = pd.read_pickle(path)
        frame = frame.loc[frame["partition_id"].isin(["train", "validation", "test"])]
        spread = frame.groupby("duplicate_group_id")["partition_id"].nunique()
        crossing = int((spread > 1).sum())
        classes_ok = all(set(part["class_id"]) == {0, 1, 2} for _, part in frame.groupby("partition_id"))
        ok = crossing == 0 and classes_ok
        return _row("T05", "duplicate groups never cross partitions", "0 crossing groups; all classes in every partition",
                    f"{spread.size} groups, {crossing} crossing; all classes in every partition: {classes_ok}",
                    "PASS" if ok else "FAIL")
    rows.append(_safe("T05", "groups", "0 crossing", t05))

    def t06() -> Row:
        pipeline = bundle["pipeline"]
        steps = list(pipeline.named_steps)
        vocab = set(pipeline.named_steps["tfidf"].get_feature_names_out())
        problems = [] if steps == ["tfidf", "clf"] else [f"unexpected pipeline steps {steps}"]
        leaked = []
        for term in bundle.get("mask_terms", []):
            key = " ".join(str(term).lower().split())
            if key in vocab:
                leaked.append(key)
        note = (f"{len(leaked)} masked term(s) still appear in the vocabulary (they cross the subject:/body: "
                f"markers or were reintroduced): {', '.join(sorted(leaked)[:15])}") if leaked else ""
        return _row("T06", "feature audit", "model input is text only (tfidf then classifier)",
                    f"steps={steps}; vocabulary size={len(vocab)}; masked terms present: {len(leaked)}",
                    "PASS" if not problems else "FAIL", "; ".join(problems) or note)
    rows.append(_safe("T06", "feature audit", "text only", t06))

    def t10() -> Row:
        path = results_dir / "source_tests_summary.csv"
        if not path.exists():
            return _row("T10", "no macro score for a missing class", "macro_f1 empty when a class is absent",
                        f"missing {path}", "SKIPPED")
        summary = pd.read_csv(path)
        bad = []
        for _, row in summary.iterrows():
            present = str(row.get("present_classes", ""))
            if len([p for p in present.split(",") if p.strip()]) < 3 and pd.notna(row.get("macro_f1")):
                bad.append(str(row.get("experiment")))
        return _row("T10", "no macro score for a missing class", "macro_f1 empty for every two-class hold-out",
                    f"{len(summary)} experiments checked; offending: {bad or 'none'}", "PASS" if not bad else "FAIL")
    rows.append(_safe("T10", "macro absent class", "no macro", t10))
    return rows


def run_log_check(log_path: Optional[Path]) -> Row:
    if log_path is None or not log_path.exists():
        return _row("LOG", "prediction log stores no message text", "hashes only",
                    "no log file found", "SKIPPED", "Run the app once, then rerun with --log logs/predictions.csv")
    frame = pd.read_csv(log_path)
    forbidden = [c for c in frame.columns if c.lower() in ("subject", "body", "text", "message", "model_text")]
    leaked = [s for s in (PHISHING[1][:40], LEGITIMATE[1][:40], SPAM[1][:40])
              if frame.astype(str).apply(lambda col: col.str.contains(s, regex=False)).any().any()]
    ok = not forbidden and not leaked
    return _row("LOG", "prediction log stores no message text", "no text columns, no message text in any cell",
                f"{len(frame)} rows; text columns: {forbidden or 'none'}; sample text found: {bool(leaked)}",
                "PASS" if ok else "FAIL")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the software tests against the real final bundle")
    parser.add_argument("--results-dir", required=True, help="e.g. results_rev5")
    parser.add_argument("--bundle", help="default: models/<results folder name>/final_bundle.joblib")
    parser.add_argument("--processed-dir", help="default: data/processed/<results folder name>")
    parser.add_argument("--table", default="configs/decision_table.yaml", help="the decision table the app uses")
    parser.add_argument("--log", help="optional prediction log, e.g. logs/predictions.csv")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_dir = results_dir if results_dir.is_absolute() else ROOT / results_dir
    tag = results_dir.name
    bundle_path = Path(args.bundle) if args.bundle else ROOT / "models" / tag / "final_bundle.joblib"
    processed = Path(args.processed_dir) if args.processed_dir else ROOT / "data" / "processed" / tag
    table_path = Path(args.table) if Path(args.table).is_absolute() else ROOT / args.table
    log_path = (Path(args.log) if Path(args.log).is_absolute() else ROOT / args.log) if args.log else None
    for label, path in (("bundle", bundle_path), ("decision table", table_path)):
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            return 2

    bundle = load_bundle(bundle_path)
    table = load_decision_table(table_path)
    rows = run_app_tests(bundle, table, bundle_path, table_path)
    rows += run_data_tests(bundle, processed, results_dir)
    rows.append(run_log_check(log_path))

    frame = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"software_test_results_{stamp}.csv"
    frame.to_csv(out, index=False, encoding="utf-8", mode="x")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(frame[["test_id", "name", "status", "actual"]].to_string(index=False))
    for _, row in frame[frame["notes"].astype(str).str.len() > 0].iterrows():
        print(f"\n[{row['test_id']}] {row['notes']}")
    counts = frame["status"].value_counts().to_dict()
    print(f"\nSummary: {counts}\nEvidence table written to {out}")
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
