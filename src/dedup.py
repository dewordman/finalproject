# src/dedup.py
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from src.config import require
from src.text_build import build_model_text, get_text_options
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            if self.rank[ri] < self.rank[rj]:
                self.parent[ri] = rj
            elif self.rank[ri] > self.rank[rj]:
                self.parent[rj] = ri
            else:
                self.parent[rj] = ri
                self.rank[ri] += 1

    def get_components(self) -> List[List[int]]:
        components: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            components.setdefault(root, []).append(i)
        return list(components.values())


def compute_exact_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_exact_hash(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exact_hash"] = df["model_text"].apply(compute_exact_hash)
    return df


def duplicate_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Settings that must agree between main and external duplicate checks."""
    return {"representation": "hashing_word_tf_v1",
            "mask_terms": list(require(cfg, "text.mask_terms")),
            "mask_dates": require(cfg, "text.mask_dates"),
            "text_options": get_text_options(cfg),
            "max_input_chars": int(require(cfg, "text.max_input_chars")),
            "near_dup_threshold": float(require(cfg, "dedup.near_dup_threshold"))}


def _save_duplicate_reference(df, cfg, vectorizer) -> None:
    """Persist only main text/hash/group metadata; never persist labels here."""
    reference = df.loc[df["model_text"].fillna("").ne(""),
                       ["record_id", "model_text", "duplicate_group_id"]].copy()
    reference["exact_hash"] = reference["model_text"].map(compute_exact_hash)
    reference = reference.drop_duplicates("exact_hash").reset_index(drop=True)
    vectors = vectorizer.transform(reference["model_text"]) if vectorizer is not None else None
    artifact = {"version": 1, "settings": duplicate_settings(cfg),
                "records": reference, "vectorizer": vectorizer, "vectors": vectors}
    path = Path(require(cfg, "paths.processed_dir")) / "duplicate_reference.joblib"
    if path.exists():
        raise FileExistsError(f"Duplicate reference already exists: {path}")
    joblib.dump(artifact, path)


def assign_duplicate_groups(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "partition_id" in df and df["partition_id"].eq("external_temporal").any():
        raise ValueError("External temporal records must not enter main-corpus deduplication")
    df = df.copy()
    max_chars = require(cfg, "text.max_input_chars")
    mask_terms = list(require(cfg, "text.mask_terms"))
    mask_dates = require(cfg, "text.mask_dates")
    text_options = get_text_options(cfg)
    threshold = require(cfg, "dedup.near_dup_threshold")
    review_size = require(cfg, "dedup.review_sample_size")
    seed = int(require(cfg, "seed"))
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    outputs = [processed_dir / "records_grouped.pkl",
               processed_dir / "duplicate_reference.joblib",
               results_dir / "dedup_report.csv", results_dir / "near_dup_review_sample.csv"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Dedup outputs already exist; choose a fresh results basename")
    if not 0 < threshold <= 1:
        raise ValueError("dedup.near_dup_threshold must be in (0, 1]")

    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    # 1. Rebuild model_text with the current mask_terms and mask_dates (parse outputs are preserved)
    ok_mask = df["parse_status"] == "ok"
    df_ok = df[ok_mask].copy()

    def build_and_truncate(row):
        txt = build_model_text(row["subject_raw"], row["body_raw"], mask_terms=mask_terms, mask_dates=mask_dates,
                               options=text_options)
        return txt[:max_chars]

    df_ok["model_text"] = df_ok.apply(build_and_truncate, axis=1)
    df.loc[ok_mask, "model_text"] = df_ok["model_text"]

    # 2. Exact hash
    df_ok = add_exact_hash(df_ok)
    df.loc[ok_mask, "exact_hash"] = df_ok["exact_hash"]

    # 3. Exact-duplicate conflict check
    exact_groups = df_ok.groupby("exact_hash")
    conflict_hashes = [
        h for h, grp in exact_groups if grp["class_label"].nunique() > 1
    ]
    if conflict_hashes:
        conflict_mask = df["exact_hash"].isin(conflict_hashes)
        df.loc[conflict_mask, "parse_status"] = "excluded"
        df.loc[conflict_mask, "exclusion_reason"] = "conflicting_duplicate"

    ok_mask = df["parse_status"] == "ok"
    df_ok = df[ok_mask].copy()

    if len(df_ok) == 0:
        df["duplicate_group_id"] = -1
        report_df = _generate_report(df)
        _save_duplicate_reference(df, cfg, None)
        df.to_pickle(processed_dir / "records_grouped.pkl")
        report_df.to_csv(results_dir / "dedup_report.csv", index=False, encoding="utf-8", mode="x")
        return df, report_df

    # 4. Near-duplicate grouping: fixed hashing and cosine; no corpus fitting.
    texts = df_ok["model_text"].tolist()
    unique_texts = list(dict.fromkeys(texts))
    text_to_unique_idx = {txt: idx for idx, txt in enumerate(unique_texts)}
    orig_to_unique = [text_to_unique_idx[t] for t in texts]

    vectorizer = HashingVectorizer(n_features=2 ** 20, alternate_sign=False, norm="l2")
    X_unique = vectorizer.transform(unique_texts)
    X_norm = normalize(X_unique, norm="l2", copy=False)
    n_unique = X_norm.shape[0]
    uf_unique = UnionFind(n_unique)
    review_edges: List[Tuple[int, int, float]] = []
    chunk_size = 2000

    for i in range(0, n_unique, chunk_size):
        X_chunk = X_norm[i : i + chunk_size]
        sim = X_chunk @ X_norm.T
        sim_coo = sim.tocoo()
        mask = sim_coo.data >= threshold
        rows = sim_coo.row[mask]
        cols = sim_coo.col[mask]
        vals = sim_coo.data[mask]
        for r, c, v in zip(rows, cols, vals):
            actual_r = i + r
            if actual_r != c:
                uf_unique.union(actual_r, c)
                review_edges.append((actual_r, c, float(v)))

    unique_to_orig: Dict[int, List[int]] = {i: [] for i in range(n_unique)}
    for orig_idx, unique_idx in enumerate(orig_to_unique):
        unique_to_orig[unique_idx].append(orig_idx)

    uf_orig = UnionFind(len(df_ok))
    final_review_edges: List[Tuple[str, str, float, str, str]] = []

    for comp in uf_unique.get_components():
        orig_indices = []
        for unique_idx in comp:
            orig_indices.extend(unique_to_orig[unique_idx])
        for i in range(1, len(orig_indices)):
            uf_orig.union(orig_indices[0], orig_indices[i])
        if len(comp) > 1:
            for u1, u2, v in review_edges:
                if uf_unique.find(u1) == uf_unique.find(comp[0]):
                    idx1 = unique_to_orig[u1][0]
                    idx2 = unique_to_orig[u2][0]
                    if idx1 != idx2:
                        final_review_edges.append(
                            (
                                df_ok.iloc[idx1]["record_id"],
                                df_ok.iloc[idx2]["record_id"],
                                v,
                                df_ok.iloc[idx1]["model_text"][:200],
                                df_ok.iloc[idx2]["model_text"][:200],
                            )
                        )
                        break

    # 5. Near-duplicate conflict check
    conflict_df_indices = []
    for comp in uf_orig.get_components():
        comp_df_indices = [df_ok.index[pos] for pos in comp]
        if df.loc[comp_df_indices, "class_label"].nunique() > 1:
            conflict_df_indices.extend(comp_df_indices)
    if conflict_df_indices:
        df.loc[conflict_df_indices, "parse_status"] = "excluded"
        df.loc[conflict_df_indices, "exclusion_reason"] = "conflicting_duplicate"

    # 6. Assign duplicate_group_id (robust index mapping)
    df["duplicate_group_id"] = -1
    ok_indices = df[df["parse_status"] == "ok"].index.tolist()
    if ok_indices:
        idx_to_pos = {idx: pos for pos, idx in enumerate(ok_indices)}
        pos_to_idx = {pos: idx for pos, idx in enumerate(ok_indices)}
        uf_valid = UnionFind(len(ok_indices))
        for comp in uf_orig.get_components():
            valid_pos = [idx_to_pos[df_ok.index[p]] for p in comp if df_ok.index[p] in idx_to_pos]
            for i in range(1, len(valid_pos)):
                uf_valid.union(valid_pos[0], valid_pos[i])
        group_id = 0
        for comp in uf_valid.get_components():
            for pos in comp:
                df.at[pos_to_idx[pos], "duplicate_group_id"] = group_id
            group_id += 1

    # 7. Write review sample — DETERMINISTIC under cfg["seed"]
    if final_review_edges:
        rng = random.Random(seed)
        sample_size = min(review_size, len(final_review_edges))
        sampled = rng.sample(final_review_edges, sample_size)
        sample_df = pd.DataFrame(
            sampled,
            columns=["record_id_1", "record_id_2", "similarity", "text_1", "text_2"],
        )
        sample_df.to_csv(
            results_dir / "near_dup_review_sample.csv", index=False, encoding="utf-8", mode="x"
        )

    # 8. Save outputs
    _save_duplicate_reference(df, cfg, vectorizer)
    df.to_pickle(processed_dir / "records_grouped.pkl")
    report_df = _generate_report(df)
    report_df.to_csv(results_dir / "dedup_report.csv", index=False, encoding="utf-8", mode="x")
    return df, report_df


def _generate_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["source_dataset", "class_label", "parse_status"])
        .size()
        .reset_index(name="count")
    )
