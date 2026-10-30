"""
extract_thresholds.py (FIXED for PyArrow indexing)
===================================================
Empirically derives decision thresholds (t_low, t_high) from the
validation partition of the masked dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
PARSED_DATA = Path("data/processed/results_rev3/records_parsed.pkl")
OUTPUT_CSV = Path("results_rev3/threshold_sweep.csv")
SEED = 42

MIN_PHISHING_RECALL = 0.89
MAX_LEGIT_FPR = 0.03

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "sublinear_tf": True,
    "max_features": None,
}

LOGREG_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEED,
}

# ── Step 1: Load and filter data ──────────────────────────────────────
print("=" * 60)
print("THRESHOLD EXTRACTION SCRIPT")
print("=" * 60)

print("\n[1/6] Loading parsed data...")
df = pd.read_pickle(PARSED_DATA)
df = df[df["parse_status"] == "ok"].copy()
print(f"      Loaded {len(df)} records with parse_status='ok'")

# ── Step 2: Grouped train/val/test split ──────────────────────────────
print("\n[2/6] Performing grouped split (70/15/15)...")

df["duplicate_group_id"] = df["duplicate_group_id"].astype(int)
df["class_id"] = df["class_id"].astype(int)
df = df.reset_index(drop=True)

# Give each singleton its own group
split_groups = df["duplicate_group_id"].copy()
singleton_mask = split_groups == -1
split_groups[singleton_mask] = -np.arange(2, 2 + singleton_mask.sum())

print(f"      Total unique split-groups: {split_groups.nunique()}")

# First split: 70% train, 30% temp
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(gss1.split(
    X=np.arange(len(df)),
    y=df["class_id"].values,
    groups=split_groups.values
))

df_train = df.iloc[train_idx]
df_temp = df.iloc[temp_idx]

# Second split: 50/50 of temp -> 15% val, 15% test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss2.split(
    X=np.arange(len(df_temp)),
    y=df_temp["class_id"].values,
    groups=split_groups.iloc[temp_idx].values
))

df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

print(f"      Train: {len(df_train)} records")
print(f"      Val:   {len(df_val)} records")
print(f"      Test:  {len(df_test)} records")

# Verify no group leakage
train_groups = set(df_train["duplicate_group_id"][df_train["duplicate_group_id"] != -1])
val_groups = set(df_val["duplicate_group_id"][df_val["duplicate_group_id"] != -1])
test_groups = set(df_test["duplicate_group_id"][df_test["duplicate_group_id"] != -1])
leakage_tv = train_groups & val_groups
leakage_ts = train_groups & test_groups
leakage_vs = val_groups & test_groups
if leakage_tv or leakage_ts or leakage_vs:
    print(f"      WARNING: Group leakage detected!")
else:
    print("      No duplicate-group leakage detected.")

# ── Step 3: Train TF-IDF + Logistic Regression pipeline ───────────────
print("\n[3/6] Training TF-IDF + Logistic Regression pipeline...")

# KEY FIX: Convert to numpy arrays to avoid PyArrow indexing issues
X_train = df_train["model_text"].to_numpy()
y_train = df_train["class_id"].to_numpy()
X_val = df_val["model_text"].to_numpy()
y_val = df_val["class_id"].to_numpy()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
    ("logreg", LogisticRegression(**LOGREG_PARAMS)),
])

pipeline.fit(X_train, y_train)
print("      Pipeline trained successfully.")

# ── Step 4: Apply sigmoid calibration ─────────────────────────────────
print("\n[4/6] Applying sigmoid calibration on training data...")

calibrated = CalibratedClassifierCV(
    estimator=pipeline,
    method="sigmoid",
    cv=5,
)

calibrated.fit(X_train, y_train)
print("      Calibration complete.")

# Get calibrated probabilities on validation set
y_prob = calibrated.predict_proba(X_val)

class_names = ["Legitimate", "Spam", "Phishing"]
print(f"\n      Validation class distribution:")
for i, name in enumerate(class_names):
    count = (y_val == i).sum()
    print(f"        {name}: {count} ({count/len(y_val)*100:.1f}%)")

# ── Step 5: Threshold sweep ───────────────────────────────────────────
print("\n[5/6] Sweeping thresholds...")

thresholds = np.arange(0.30, 0.96, 0.01)
results = []

for t in thresholds:
    row = {"threshold": round(t, 2)}
    for i, name in enumerate(class_names):
        binary_pred = (y_prob[:, i] >= t).astype(int)
        binary_true = (y_val == i).astype(int)

        if binary_true.sum() > 0:
            rec = recall_score(binary_true, binary_pred, zero_division=0)
        else:
            rec = 0.0

        actual_not_i = (binary_true == 0)
        if actual_not_i.sum() > 0:
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fpr = fp / actual_not_i.sum()
        else:
            fpr = 0.0

        row[f"{name}_recall"] = round(rec, 4)
        row[f"{name}_fpr"] = round(fpr, 4)

    results.append(row)

sweep_df = pd.DataFrame(results)

# ── Step 6: Find optimal thresholds ───────────────────────────────────
print("\n[6/6] Finding optimal thresholds...\n")

phishing_ok = sweep_df[sweep_df["Phishing_recall"] >= MIN_PHISHING_RECALL]

if len(phishing_ok) > 0:
    best_phishing_t = phishing_ok["threshold"].max()
    best_phishing_row = phishing_ok[phishing_ok["threshold"] == best_phishing_t].iloc[0]
    print(f"  Phishing Recall >= {MIN_PHISHING_RECALL}:")
    print(f"    Highest qualifying threshold: {best_phishing_t:.2f}")
    print(f"    At this threshold:")
    print(f"      Phishing Recall: {best_phishing_row['Phishing_recall']:.4f}")
    print(f"      Phishing FPR:    {best_phishing_row['Phishing_fpr']:.4f}")
    print(f"      Legit Recall:    {best_phishing_row['Legitimate_recall']:.4f}")
    print(f"      Legit FPR:       {best_phishing_row['Legitimate_fpr']:.4f}")
else:
    print(f"  WARNING: No threshold achieves Phishing Recall >= {MIN_PHISHING_RECALL}")
    best_phishing_t = None

legit_ok = sweep_df[sweep_df["Legitimate_fpr"] <= MAX_LEGIT_FPR]

if len(legit_ok) > 0:
    best_legit_t = legit_ok["threshold"].min()
    best_legit_row = legit_ok[legit_ok["threshold"] == best_legit_t].iloc[0]
    print(f"\n  Legitimate FPR <= {MAX_LEGIT_FPR}:")
    print(f"    Lowest qualifying threshold: {best_legit_t:.2f}")
    print(f"    At this threshold:")
    print(f"      Legit FPR:       {best_legit_row['Legitimate_fpr']:.4f}")
    print(f"      Legit Recall:    {best_legit_row['Legitimate_recall']:.4f}")
    print(f"      Phishing Recall: {best_legit_row['Phishing_recall']:.4f}")
    print(f"      Phishing FPR:    {best_legit_row['Phishing_fpr']:.4f}")
else:
    print(f"\n  WARNING: No threshold achieves Legitimate FPR <= {MAX_LEGIT_FPR}")
    best_legit_t = None

# ── Recommendation ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDED THRESHOLDS FOR decision_table.yaml")
print("=" * 60)

if best_phishing_t is not None and best_legit_t is not None:
    t_high = best_phishing_t
    t_low = round(best_legit_t - 0.15, 2)
    t_low = max(t_low, 0.40)

    print(f"\n  t_low:  {t_low:.2f}")
    print(f"  t_high: {t_high:.2f}")
    print(f"\n  Confidence bands:")
    print(f"    Low:      confidence < {t_low:.2f}  -> Review")
    print(f"    Moderate: {t_low:.2f} <= confidence < {t_high:.2f}  -> Analyst verification")
    print(f"    High:     confidence >= {t_high:.2f}  -> Advisory action")
    print(f"\n  Justification:")
    print(f"    t_high={t_high:.2f} is the highest threshold that still")
    print(f"    achieves Phishing Recall >= {MIN_PHISHING_RECALL}.")
    print(f"    t_low={t_low:.2f} creates a conservative uncertainty zone")
    print(f"    that keeps Legitimate FPR <= {MAX_LEGIT_FPR}.")
elif best_phishing_t is not None:
    t_high = best_phishing_t
    t_low = round(t_high - 0.20, 2)
    print(f"\n  t_low:  {t_low:.2f} (estimated)")
    print(f"  t_high: {t_high:.2f}")
elif best_legit_t is not None:
    t_low = best_legit_t
    t_high = round(t_low + 0.20, 2)
    print(f"\n  t_low:  {t_low:.2f}")
    print(f"  t_high: {t_high:.2f} (estimated)")
else:
    print("\n  Could not determine thresholds automatically.")
    print("  Review the full sweep in threshold_sweep.csv")

# ── Save sweep results ─────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
sweep_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Full sweep saved to: {OUTPUT_CSV}")
print("=" * 60)