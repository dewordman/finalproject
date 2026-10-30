"""
error_review_generator.py
=========================
Merges error_review_sheet.csv with the actual model_text from parsed data
to create a working Excel file for manual error analysis coding.
Includes sanitization and safe truncation for Excel compatibility.
"""

import pandas as pd
import re
import warnings
from pathlib import Path

# Suppress the Excel cell length warning since we are handling it manually
warnings.filterwarnings("ignore", message="Cell contents too long")

# ── Configuration ──────────────────────────────────────────────────────
for results_dir in ["results_rev4_masked", "results_rev5"]:
    parsed_path = Path(f"data/processed/{results_dir}/records_parsed.pkl")
    if parsed_path.exists():
        PARSED_DATA = parsed_path
        print(f"✅ Found parsed data in: {results_dir}")
        break
else:
    print("❌ Could not find records_parsed.pkl in either results folder.")
    exit(1)

ERROR_SHEET = Path("error_review_sheet.csv")
if not ERROR_SHEET.exists():
    for results_dir in ["results_rev4_masked", "results_rev5"]:
        alt = Path(f"{results_dir}/error_review_sheet.csv")
        if alt.exists():
            ERROR_SHEET = alt
            print(f"✅ Found error review sheet at: {ERROR_SHEET}")
            break
    else:
        print("❌ Could not find error_review_sheet.csv")
        print("   Run the 'error_review' pipeline stage first.")
        exit(1)

OUTPUT_FILE = Path("error_review_for_coding.xlsx")

# ── Helper function to sanitize text for Excel ─────────────────────────
def sanitize_for_excel(text):
    """Removes characters that are illegal in Excel worksheets and truncates."""
    if isinstance(text, str):
        # Remove illegal control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
        # Truncate to 5000 characters to keep Excel fast and readable
        return text[:5000] + ("..." if len(text) > 5000 else "")
    return ""

# ── Step 1: Load data ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING ERROR REVIEW FILE FOR MANUAL CODING")
print("=" * 60)

print("\n[1/4] Loading parsed data...")
df_all = pd.read_pickle(PARSED_DATA)
print(f"      Loaded {len(df_all):,} total records")

print("\n[2/4] Loading error review sheet...")
df_errors = pd.read_csv(ERROR_SHEET)
print(f"      Found {len(df_errors)} error cases to review")

# ── Step 2: Merge to get model_text ───────────────────────────────────
print("\n[3/4] Merging error cases with actual email text...")

cols_to_keep = ["record_id", "model_text"]
if "subject_raw" in df_all.columns:
    cols_to_keep.append("subject_raw")
if "body_raw" in df_all.columns:
    cols_to_keep.append("body_raw")

df_text = df_all[cols_to_keep].copy()

df_merged = pd.merge(
    df_errors,
    df_text,
    on="record_id",
    how="left",
    suffixes=("", "_full")
)

missing = df_merged["model_text"].isna().sum()
if missing > 0:
    print(f"      ⚠️  Warning: {missing} error records could not be matched to parsed data")

# ── Step 3: Sanitize and truncate text for Excel ──────────────────────
print("\n[3.5/4] Sanitizing and truncating text for Excel compatibility...")
text_cols = [col for col in ["model_text", "subject_raw", "body_raw"] if col in df_merged.columns]
for col in text_cols:
    df_merged[col] = df_merged[col].apply(sanitize_for_excel)

# ── Step 4: Save to Excel ─────────────────────────────────────────────
print("\n[4/4] Saving to Excel...")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_merged.to_excel(writer, index=False, sheet_name="Error Review")
    
    worksheet = writer.sheets["Error Review"]
    for idx, col in enumerate(df_merged.columns):
        # Safely calculate max length using .str.len() which handles NaNs gracefully
        lengths = df_merged[col].astype(str).str.len()
        max_length = max(lengths.max(), len(str(col)))
        
        # Cap width at 80 characters for readability
        adjusted_width = min(max_length + 2, 80)
        
        # Convert index to Excel column letter (A, B, C, ..., Z, AA, AB, etc.)
        col_letter = ""
        temp_idx = idx
        while temp_idx >= 0:
            col_letter = chr(65 + (temp_idx % 26)) + col_letter
            temp_idx = (temp_idx // 26) - 1
            
        worksheet.column_dimensions[col_letter].width = adjusted_width

print(f"\n✅ Saved to: {OUTPUT_FILE.absolute()}")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CODING SUMMARY")
print("=" * 60)
print(f"\nTotal errors to code: {len(df_merged)}")
print("\nBreakdown by error type:")
print(df_merged["category"].value_counts().to_string())

print("\n" + "=" * 60)
print("CODING INSTRUCTIONS")
print("=" * 60)
print("""
1. Open 'error_review_for_coding.xlsx' in Excel or Google Sheets
2. For each row, read the 'model_text' column to see the actual email
3. Fill in the 5 boolean columns with 1 (Yes) or 0 (No):
   
   • ambiguous_intent:         1 if email purpose is genuinely unclear
   • promotional_wording:      1 if contains marketing/sales language
   • impersonation_or_urgency: 1 if contains urgent/threatening language
   • formatting_artefact:      1 if has parsing errors or weird formatting
   • plausible_source_dependence: 1 if model relied on dataset-specific jargon
   
4. Add a brief 'coder_note' explaining WHY the model made this mistake
5. Save the file when finished

Next step: After coding, run 'save_coded_error_review.py' to generate
the final error_review_sheet.csv for your supervisor submission.
""")