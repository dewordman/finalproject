"""
save_coded_error_review.py
===========================
Converts your manually-coded Excel file back into the clean CSV format
required for supervisor submission (without raw message bodies).
"""

import pandas as pd
from pathlib import Path

INPUT_FILE = Path("error_review_for_coding.xlsx")
OUTPUT_FILE = Path("error_review_sheet.csv")

if not INPUT_FILE.exists():
    print(f"❌ Cannot find {INPUT_FILE}")
    print("   Please complete manual coding in Excel first.")
    exit(1)

print("Loading coded Excel file...")
df = pd.read_excel(INPUT_FILE)

# Check that coding is complete
coding_cols = [
    "ambiguous_intent",
    "promotional_wording",
    "impersonation_or_urgency",
    "formatting_artefact",
    "plausible_source_dependence"
]

missing_codes = 0
for col in coding_cols:
    if col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"⚠️  {null_count} rows missing code for '{col}'")
            missing_codes += null_count

if missing_codes > 0:
    response = input(f"\n{missing_codes} cells still need coding. Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Aborted. Please finish coding in Excel first.")
        exit(0)

# Remove raw message bodies (per supervisor's submission rules)
print("\nRemoving raw message bodies for submission...")
drop_cols = ["model_text", "subject_raw", "body_raw"]
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Save
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"\n✅ Saved clean submission file: {OUTPUT_FILE.absolute()}")
print(f"   Total rows: {len(df_clean)}")
print(f"   Columns: {list(df_clean.columns)}")

# Summary statistics
print("\n" + "=" * 60)
print("CODING STATISTICS (for Chapter 5)")
print("=" * 60)
for col in coding_cols:
    if col in df_clean.columns:
        yes_count = (df_clean[col] == 1).sum()
        pct = yes_count / len(df_clean) * 100
        print(f"  {col}: {yes_count} ({pct:.1f}%)")

print("\n📝 You can now use these statistics in your Chapter 5 error analysis section.")