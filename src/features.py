# src/features.py
from __future__ import annotations

from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(model_name: str, params: Dict[str, Any], seed: int) -> Pipeline:
    """Build a TF-IDF + classifier pipeline."""
    tfidf_kw = dict(params.get("tfidf", {}))
    clf_kw = dict(params.get("clf", {}))

    if "ngram_range" in tfidf_kw and isinstance(tfidf_kw["ngram_range"], list):
        tfidf_kw["ngram_range"] = tuple(tfidf_kw["ngram_range"])

    tfidf = TfidfVectorizer(
        lowercase=False,
        stop_words=None,
        **tfidf_kw,
    )

    if model_name == "logreg":
        # Removed penalty="l2" as it is the default and triggers a deprecation warning in newer sklearn
        clf = LogisticRegression(
            solver="lbfgs",
            random_state=seed,
            **clf_kw,
        )
    elif model_name == "nb":
        clf = MultinomialNB(**clf_kw)
    elif model_name == "svm":
        clf = LinearSVC(
            random_state=seed,
            dual=True,
            **clf_kw,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])