# app/streamlit_app.py
"""Email Threat Classification and Decision-Support Prototype (Streamlit).

The app contains NO copy of the preprocessing or the decision logic. It calls
``src.inference.predict_email`` with the saved final bundle, so the text is prepared exactly as in
training (the options are stored in the bundle) and the decision table is read from one place.

Run from the project root:   streamlit run app/streamlit_app.py
Optional environment variables (defaults in brackets):
  EMAIL_APP_RUN     results folder name of the run to serve         [results_rev5]
  EMAIL_APP_BUNDLE  full path to final_bundle.joblib                [models/<run>/final_bundle.joblib]
  EMAIL_APP_TABLE   decision table                                  [configs/decision_table.yaml]
  EMAIL_APP_LOG     prediction log                                  [logs/app_predictions.csv]
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.decision import load_decision_table  # noqa: E402
from src.inference import load_bundle, predict_email  # noqa: E402

RUN_TAG = os.environ.get("EMAIL_APP_RUN", "results_rev5")
BUNDLE_PATH = Path(os.environ.get("EMAIL_APP_BUNDLE", PROJECT_ROOT / "models" / RUN_TAG / "final_bundle.joblib"))
TABLE_PATH = Path(os.environ.get("EMAIL_APP_TABLE", PROJECT_ROOT / "configs" / "decision_table.yaml"))
LOG_PATH = Path(os.environ.get("EMAIL_APP_LOG", PROJECT_ROOT / "logs" / "app_predictions.csv"))
LOG_TOP_TERMS = False  # True also logs up to five influential vocabulary terms (words from the message)
LOG_COLUMNS = ["timestamp", "input_sha256_16", "predicted_class", "confidence", "confidence_method",
               "risk_band", "advisory_action", "model_version", "rule_version", "top_terms"]
CLASS_COLOURS = {"Legitimate": "green", "Spam": "orange", "Phishing": "red"}
BAND_LABELS = {"low": "🟡 Low", "moderate": "🟠 Moderate", "high": "🟢 High"}


@st.cache_resource
def _bundle(path: str) -> Dict[str, Any]:
    return load_bundle(Path(path))


@st.cache_resource
def _table(path: str) -> Dict[str, Any]:
    return load_decision_table(Path(path))


def log_prediction(subject: str, body: str, result: Dict[str, Any]) -> None:
    """Append one row: a SHA-256 prefix of the raw input, never the message text."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((subject + "\n" + body).encode("utf-8")).hexdigest()[:16]
        terms = ", ".join(t["term"] for t in result["supporting_terms"][:5]) if LOG_TOP_TERMS else ""
        new_file = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(LOG_COLUMNS)
            writer.writerow([datetime.now().isoformat(timespec="seconds"), digest, result["predicted_class"],
                             f"{result['confidence']:.4f}", result["confidence_method"], result["band"],
                             result["action"], result["model_version"], result["rule_version"], terms])
    except Exception:
        pass  # a logging problem must never break the interface


st.set_page_config(page_title="Email Threat Classifier", page_icon="🛡️", layout="wide")
st.title("🛡️ Email Threat Classification and Decision-Support Prototype")
st.caption("Interpretable TF-IDF + Logistic Regression · Proposed application context: Nigerian oil and gas")

if not BUNDLE_PATH.exists():
    st.error(f"Model bundle not found: {BUNDLE_PATH}. Run the pipeline up to the bundle stage first.")
    st.stop()
if not TABLE_PATH.exists():
    st.error(f"Decision table not found: {TABLE_PATH}")
    st.stop()
bundle, table = _bundle(str(BUNDLE_PATH)), _table(str(TABLE_PATH))
st.info(f"**Decision table v{table['rule_version']}** · t_low = {table['t_low']} · t_high = {table['t_high']} "
        f"· model version {bundle['model_version']}")

st.subheader("1. Email Input")
col_a, col_b = st.columns([1, 3])
with col_a:
    subject = st.text_input("Subject", placeholder="e.g. Urgent: Invoice #4521 attached", max_chars=500)
with col_b:
    body = st.text_area("Body", height=220, placeholder="Paste the email body here. HTML, URLs and header "
                        "remnants are treated as inert text: nothing is opened, rendered or fetched.")
empty_input = not subject.strip() and not body.strip()
if empty_input:
    st.warning("⚠️ Please enter text in the subject or body field before analysing.")
clicked = st.button("Analyse Email", type="primary", disabled=empty_input)

if clicked and not empty_input:
    result = predict_email(subject, body, bundle, table)
    if result["status"] != "success":
        st.error(f"⚠️ {result['message']} No prediction was made.")
    else:
        log_prediction(subject.strip(), body.strip(), result)
        st.markdown("---")
        st.subheader("2. Classification Result")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Predicted Class")
            st.markdown(f"<span style='color:{CLASS_COLOURS.get(result['predicted_class'], 'gray')}; "
                        f"font-size:24px; font-weight:bold'>{result['predicted_class']}</span>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown("### Assessed Confidence")
            st.metric(label=f"Probability ({result['confidence_method']} calibration)",
                      value=f"{result['confidence']:.1%}")
        with c3:
            st.markdown("### Risk Band")
            st.markdown(f"**{BAND_LABELS.get(result['band'], result['band'])}**")
        st.info(f"**Recommended Action:** {result['action']}")
        st.caption("Assessed confidence is a calibrated model probability, not a probability of harm. "
                   "Bands are decision rules set from validation evidence.")
        if result["band"] == "low":
            st.warning("⚠️ **Low confidence.** This message is routed for manual analyst review. "
                       "Do not rely on this output as a final decision.")
        st.subheader("3. Influential Terms")
        if result["supporting_terms"]:
            st.caption("Terms that pushed the model toward the predicted class (TF-IDF value × coefficient).")
            st.dataframe(pd.DataFrame(result["supporting_terms"]).rename(
                columns={"term": "Term", "contribution": "Contribution"}), hide_index=True)
        else:
            st.warning("No known vocabulary terms were found in this message, so no evidence can be shown.")
        if result["opposing_terms"]:
            st.caption("Terms that pushed against the predicted class.")
            st.dataframe(pd.DataFrame(result["opposing_terms"]).rename(
                columns={"term": "Term", "contribution": "Contribution"}), hide_index=True)
        st.subheader("4. Class Probability Distribution")
        st.bar_chart(pd.DataFrame({"Probability": result["probabilities"]}))

st.markdown("---")
st.warning(
    "⚠️ **Scope and Limitations**\n\n"
    "This prototype is a **research decision-support tool**, not a production email gateway.\n\n"
    "- Analyses **English subject and body text only**.\n"
    "- Trained on **historical public corpora** (Enron-Spam, SpamAssassin, Nazario); not validated on "
    "Nigerian oil-and-gas email.\n"
    "- **Does not inspect** live URLs, attachments, sender reputation, SPF/DKIM/DMARC, or images.\n"
    "- **Does not quarantine, forward, or delete** messages. All actions are **advisory**.\n"
    "- Final security decisions remain the responsibility of a qualified human analyst and organisational policy."
)
