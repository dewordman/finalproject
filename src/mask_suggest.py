# src/mask_suggest.py
"""Suggest (never apply) candidate ``text.mask_terms`` from the audit output."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

import pandas as pd

from src.text_build import strip_temporal_cues

_DATEISH = re.compile(r"^(?:mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|"
                      r"sep|sept|oct|nov|dec)$", re.I)


def covered_by_temporal_cues(term: str) -> bool:
    """True if strip_temporal_cues already removes this unigram (years, times, dates)."""
    probe = f" {term} "
    return strip_temporal_cues(probe).strip() == "" or bool(_DATEISH.match(term))


def suggest_terms(flagged_csv: Union[str, Path], dominance_threshold: float,
                  out_csv: Union[str, Path]) -> pd.DataFrame:
    flagged = pd.read_csv(flagged_csv, encoding="utf-8")
    rows = []
    for _, row in flagged.iterrows():
        if float(row["top_source_share"]) < dominance_threshold:
            continue
        for word in str(row["term"]).split():
            if covered_by_temporal_cues(word):
                continue
            rows.append({"term": word, "dominant_source": row["top_source"],
                         "share": round(float(row["top_source_share"]), 4),
                         "class": row["class"]})
    candidates = (pd.DataFrame(rows, columns=["term", "dominant_source", "share", "class"])
                  .sort_values(["share", "term"], ascending=[False, True])
                  .drop_duplicates(["term", "class"]).reset_index(drop=True))
    out_csv = Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"{out_csv} already exists")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_csv, index=False, encoding="utf-8", mode="x")
    return candidates


def yaml_list(candidates: pd.DataFrame) -> str:
    terms: List[str] = sorted(set(candidates["term"]))
    return "mask_terms:\n" + "\n".join(f'  - "{term}"' for term in terms)
