# src/external.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import require
from src.dedup import UnionFind, compute_exact_hash, duplicate_settings
from src.parse import parse_nazario_external
from src.utils import ensure_dir


def _duplicate_mask(external: pd.DataFrame, reference, threshold: float) -> np.ndarray:
    """Remove entire external components connected to any main text or exact hash."""
    hits = np.zeros(len(external), dtype=bool)
    indices = np.flatnonzero(external["model_text"].fillna("").ne("").to_numpy(dtype=bool))
    if not len(indices):
        return hits
    rows = external.iloc[indices]
    hashes = rows["model_text"].map(compute_exact_hash).tolist()
    main_hashes = set(reference["records"]["exact_hash"])
    connected = UnionFind(len(indices))
    anchors = {index for index, digest in enumerate(hashes) if digest in main_hashes}
    hash_owner = {}
    for index, digest in enumerate(hashes):
        if digest in hash_owner:
            connected.union(index, hash_owner[digest])
        else:
            hash_owner[digest] = index
    vectorizer, main_vectors = reference["vectorizer"], reference["vectors"]
    if vectorizer is not None:
        vectors = vectorizer.transform(rows["model_text"])
        for start in range(0, len(indices), 256):
            chunk = vectors[start:start + 256]
            cross = (chunk @ main_vectors.T).tocoo()
            anchors.update(start + int(row) for row in cross.row[cross.data >= threshold])
            internal = (chunk @ vectors.T).tocoo()
            selected = internal.data >= threshold
            for row, column in zip(internal.row[selected], internal.col[selected]):
                connected.union(start + int(row), int(column))
    anchored_roots = {connected.find(index) for index in anchors}
    for position, original_index in enumerate(indices):
        hits[original_index] = connected.find(position) in anchored_roots
    return hits


def run_external(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Use a label-free main reference; write external artifacts separately."""
    processed = Path(require(cfg, "paths.processed_dir"))
    results = Path(require(cfg, "paths.results_dir"))
    output, report_path = processed / "records_external.pkl", results / "external_report.csv"
    if output.exists() or report_path.exists():
        raise FileExistsError("External outputs already exist; choose a fresh results basename")
    reference = joblib.load(processed / "duplicate_reference.joblib")
    if reference["version"] != 1 or reference["settings"] != duplicate_settings(cfg):
        raise ValueError("Main duplicate reference does not match this run's text/dedup settings")
    expected = {"record_id", "model_text", "duplicate_group_id", "exact_hash"}
    if set(reference["records"].columns) != expected:
        raise ValueError("Duplicate reference must contain text/hash/group metadata only")
    threshold = float(require(cfg, "dedup.near_dup_threshold"))
    if not 0 < threshold <= 1:
        raise ValueError("dedup.near_dup_threshold must be in (0, 1]")
    external = parse_nazario_external(cfg)
    assert external["partition_id"].eq("external_temporal").all()
    external["exact_hash"] = external["model_text"].map(compute_exact_hash)
    removed = _duplicate_mask(external, reference, threshold)
    report_rows = []
    for year, rows in external.groupby("era_year", sort=True):
        index = rows.index.to_numpy()
        excluded = int((rows["parse_status"].ne("ok").to_numpy(dtype=bool) & ~removed[index]).sum())
        duplicate_count = int(removed[index].sum())
        report_rows.append({"era_year": int(year), "parsed": len(rows), "excluded": excluded,
            "removed_as_duplicate": duplicate_count, "retained": len(rows) - excluded - duplicate_count})
    retained = external.loc[~removed].copy().reset_index(drop=True)
    # Group IDs stay in an external namespace and never modify the main reference.
    retained["duplicate_group_id"] = -1
    columns = ["era_year", "parsed", "excluded", "removed_as_duplicate", "retained"]
    report = pd.DataFrame(report_rows, columns=columns)
    ensure_dir(processed)
    ensure_dir(results)
    retained.to_pickle(output)
    report.to_csv(report_path, index=False, encoding="utf-8", mode="x")
    return retained
