# src/explain.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def explain_prediction(
    pipeline: Any,
    text: str,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Return the top positive and negative feature contributions for the predicted class,
    plus the top positive contributions for the runner-up class.

    contributions c_t = x_t * coef[k, t] for non-zero features only.
    Returns empty lists safely when the TF-IDF vector is empty.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])

    if X.nnz == 0:
        return {
            "predicted_class": None,
            "confidence": None,
            "supporting": [],
            "opposing": [],
            "runner_up_class": None,
            "runner_up": [],
        }

    proba = clf.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_
    cols = X.nonzero()[1]

    def _contribs_for_class(class_idx: int) -> List[Dict[str, Any]]:
        out = []
        for col in cols:
            term = str(feature_names[col])
            tfidf_val = float(X[0, col])
            coef_val = float(coefs[class_idx, col])
            out.append({
                "term": term,
                "tfidf_value": round(tfidf_val, 6),
                "coefficient": round(coef_val, 6),
                "contribution": round(tfidf_val * coef_val, 6),
            })
        return out

    pred_contribs = _contribs_for_class(pred_class)
    sorted_pos = sorted(pred_contribs, key=lambda r: r["contribution"], reverse=True)
    sorted_neg = sorted(pred_contribs, key=lambda r: r["contribution"])

    supporting = sorted_pos[:top_n]
    opposing = sorted_neg[:top_n]

    # Runner-up class
    ordered = np.argsort(proba)[::-1]
    runner_up_class = int(ordered[1]) if len(ordered) > 1 else None
    runner_up = []
    if runner_up_class is not None:
        runner_contribs = _contribs_for_class(runner_up_class)
        runner_up = sorted(runner_contribs, key=lambda r: r["contribution"], reverse=True)[:top_n]

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "supporting": supporting,
        "opposing": opposing,
        "runner_up_class": runner_up_class,
        "runner_up": runner_up,
    }


def contribution_sum_check(pipeline: Any, text: str, tol: float = 1e-8) -> bool:
    """Verify that sum(x_t * coef[k, t]) + intercept[k] equals the class decision score
    within the given tolerance, for the predicted class k.

    Returns True when the invariant holds, False otherwise.
    Returns False (with a logged warning) when the TF-IDF vector is empty.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])
    if X.nnz == 0:
        logger.warning("contribution_sum_check: empty TF-IDF vector; cannot verify.")
        return False

    proba = clf.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))

    decision = float(clf.decision_function(X)[0][pred_class])

    coefs = clf.coef_
    intercept = float(clf.intercept_[pred_class])
    cols = X.nonzero()[1]

    total = intercept
    for col in cols:
        tfidf_val = float(X[0, col])
        coef_val = float(coefs[pred_class, col])
        total += tfidf_val * coef_val

    diff = abs(total - decision)
    ok = diff < tol
    if not ok:
        logger.warning(
            f"contribution_sum_check FAILED: total={total}, decision={decision}, diff={diff}"
        )
    return ok