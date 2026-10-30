# src/split.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

import numpy as np
import pandas as pd

from src.config import require
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

CLASS_MAP = {"Legitimate": 0, "Spam": 1, "Phishing": 2}


def make_partitions(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test partitions using stratified group assignment."""
    if "partition_id" in df and df["partition_id"].eq("external_temporal").any():
        raise ValueError("External temporal records must not enter main-corpus splitting")
    df = df.copy()
    seed = require(cfg, "seed")
    ratios = require(cfg, "split.ratios")
    n_candidates = require(cfg, "split.n_candidates")
    processed_dir = Path(require(cfg, "paths.processed_dir"))
    results_dir = Path(require(cfg, "paths.results_dir"))
    outputs = [processed_dir / "records_partitioned.pkl",
               results_dir / "partition_report.csv", results_dir / "split_manifest.csv"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Split outputs already exist; choose a fresh results basename")
    
    ensure_dir(processed_dir)
    ensure_dir(results_dir)
    
    retained_mask = df["parse_status"] == "ok"
    df_retained = df[retained_mask].copy()
    
    if len(df_retained) == 0:
        logger.warning("No retained rows to partition.")
        df["partition_id"] = ""
        report_df = pd.DataFrame()
        return df, report_df
    
    # Get unique groups with their class and size
    # Use "max" instead of "first" to get a consistent class_id per group
    groups = df_retained.groupby("duplicate_group_id").agg({
        "class_id": "max",
        "record_id": "count"
    }).reset_index()
    groups.columns = ["group_id", "class_id", "size"]
    groups = groups.sort_values("size", ascending=False).reset_index(drop=True)
    
    logger.info(f"Total groups: {len(groups)}")
    logger.info(f"Groups per class: {groups['class_id'].value_counts().to_dict()}")
    
    class_counts = df_retained["class_id"].value_counts().to_dict()
    targets = {}
    for class_id in [0, 1, 2]:
        total = class_counts.get(class_id, 0)
        targets[class_id] = [int(total * r) for r in ratios]
    
    best_candidate_idx = 0
    best_score = float("inf")
    best_assignments = {}
    
    for cand_idx in range(n_candidates):
        rng = np.random.default_rng(seed + cand_idx)
        shuffled_indices = rng.permutation(len(groups))
        
        current_counts = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}
        assignments: Dict[int, int] = {}
        assigned_groups: Set[int] = set()
        
        # First pass: ensure each partition gets at least one group per class
        for class_id in [0, 1, 2]:
            class_groups = groups[groups["class_id"] == class_id].copy()
            class_groups = class_groups.sort_values("size", ascending=False)
            
            for p_idx in range(3):
                for idx_in_class in range(len(class_groups)):
                    group_row = class_groups.iloc[idx_in_class]
                    group_id = group_row["group_id"]
                    
                    if group_id not in assigned_groups:
                        assignments[group_id] = p_idx
                        assigned_groups.add(group_id)
                        current_counts[class_id][p_idx] += group_row["size"]
                        break
        
        if cand_idx == 0:
            logger.info(f"Candidate 0 first-pass assignments: {len(assignments)} groups")
            for class_id in [0, 1, 2]:
                logger.info(f"  Class {class_id} counts: {current_counts[class_id]}")
        
        # Second pass: assign remaining groups based on deficit
        for idx in shuffled_indices:
            group = groups.iloc[idx]
            group_id = group["group_id"]
            
            if group_id in assigned_groups:
                continue
                
            class_id = group["class_id"]
            size = group["size"]
            
            deficits = []
            for p_idx in range(3):
                target = targets[class_id][p_idx]
                actual = current_counts[class_id][p_idx]
                deficit = target - actual
                deficits.append((deficit, p_idx))
            
            deficits.sort(key=lambda x: (-x[0], x[1]))
            chosen_partition = deficits[0][1]
            
            assignments[group_id] = chosen_partition
            assigned_groups.add(group_id)
            current_counts[class_id][chosen_partition] += size
        
        # Score this candidate
        score = 0.0
        for class_id in [0, 1, 2]:
            total = class_counts.get(class_id, 0)
            if total == 0:
                continue
            for p_idx in range(3):
                actual_proportion = current_counts[class_id][p_idx] / total
                target_ratio = ratios[p_idx]
                score += abs(actual_proportion - target_ratio)
        
        if score < best_score or (score == best_score and cand_idx < best_candidate_idx):
            best_score = score
            best_candidate_idx = cand_idx
            best_assignments = assignments.copy()
    
    logger.info(f"Best candidate: {best_candidate_idx}, score: {best_score:.4f}")
    logger.info(f"Total groups assigned: {len(best_assignments)}")
    
    df["partition_id"] = ""
    partition_names = ["train", "validation", "test"]
    
    for group_id, partition_idx in best_assignments.items():
        mask = df["duplicate_group_id"] == group_id
        df.loc[mask, "partition_id"] = partition_names[partition_idx]
    
    # Debug: check what's in each partition
    for partition in partition_names:
        partition_df = df[df["partition_id"] == partition]
        class_dist = partition_df["class_id"].value_counts().to_dict()
        logger.info(f"{partition}: {len(partition_df)} records, classes: {class_dist}")
    
    report_data = []
    for partition in partition_names:
        for source in df_retained["source_dataset"].unique():
            for class_id in [0, 1, 2]:
                class_label = [k for k, v in CLASS_MAP.items() if v == class_id][0]
                count = len(df[(df["partition_id"] == partition) & 
                              (df["source_dataset"] == source) & 
                              (df["class_id"] == class_id)])
                total_class = len(df[(df["partition_id"] != "") & (df["class_id"] == class_id)])
                proportion = count / total_class if total_class > 0 else 0.0
                report_data.append({
                    "partition": partition,
                    "source_dataset": source,
                    "class_label": class_label,
                    "count": count,
                    "proportion": proportion
                })
    
    report_df = pd.DataFrame(report_data)
    report_df.attrs["chosen_candidate_index"] = best_candidate_idx
    report_df.attrs["best_score"] = best_score
    
    realised_ratios = []
    for partition in partition_names:
        count = len(df[df["partition_id"] == partition])
        total = len(df[df["partition_id"] != ""])
        realised_ratios.append(count / total if total > 0 else 0.0)
    report_df.attrs["realised_ratios"] = realised_ratios
    
    df.to_pickle(processed_dir / "records_partitioned.pkl")
    report_df.to_csv(results_dir / "partition_report.csv", index=False, encoding="utf-8", mode="x")
    
    manifest = df[["record_id", "duplicate_group_id", "partition_id"]].copy()
    manifest.to_csv(results_dir / "split_manifest.csv", index=False, encoding="utf-8", mode="x")
    
    logger.info(f"Realised ratios: {realised_ratios}")
    
    return df, report_df


def assert_partition_integrity(df: pd.DataFrame) -> None:
    """Verify partition integrity constraints."""
    retained = df[df["parse_status"] == "ok"]
    
    group_partitions = retained.groupby("duplicate_group_id")["partition_id"].nunique()
    violations = group_partitions[group_partitions > 1]
    if len(violations) > 0:
        raise AssertionError(
            f"Integrity violation: {len(violations)} duplicate groups span multiple partitions. "
            f"Example group IDs: {violations.index[:5].tolist()}"
        )
    
    for partition in ["train", "validation", "test"]:
        partition_df = retained[retained["partition_id"] == partition]
        classes_present = partition_df["class_id"].unique()
        missing = set([0, 1, 2]) - set(classes_present)
        if missing:
            missing_labels = [k for k, v in CLASS_MAP.items() if v in missing]
            raise AssertionError(
                f"Integrity violation: partition '{partition}' is missing classes: {missing_labels}"
            )
    
    empty_partition = retained[retained["partition_id"] == ""]
    if len(empty_partition) > 0:
        raise AssertionError(
            f"Integrity violation: {len(empty_partition)} retained rows have empty partition_id. "
            f"Example record IDs: {empty_partition['record_id'].head(5).tolist()}"
        )
    
    logger.info("Partition integrity check PASSED.")


def assert_no_metadata_in_features(pipeline, df: pd.DataFrame) -> None:
    """Verify that the pipeline's TF-IDF input is only the model_text column."""
    tfidf_step = None
    for name, step in pipeline.steps:
        if name == "tfidf":
            tfidf_step = step
            break
    
    if tfidf_step is None:
        raise AssertionError("Pipeline does not have a 'tfidf' step.")
    
    # Identify text columns in the DataFrame using comprehensive dtype checking
    text_columns = []
    for col in df.columns:
        dtype = df[col].dtype
        # Check for object dtype (traditional string storage) or any string dtype
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            text_columns.append(col)
    
    # Exclude known metadata columns
    metadata_columns = {
        "source_dataset", "source_subset", "original_label", "class_label",
        "partition_id", "exclusion_reason", "subject_raw", "body_raw", "record_id",
        "era_group", "source_file", "input_label", "parse_status", "exact_hash"
    }
    
    # Filter out metadata columns
    text_columns = [col for col in text_columns if col not in metadata_columns]
    
    # Check that only model_text remains
    if len(text_columns) != 1 or text_columns[0] != "model_text":
        raise AssertionError(
            f"DataFrame contains unexpected text columns: {text_columns}. "
            f"Expected only 'model_text'."
        )
    
    logger.info("No metadata in features check PASSED.")
