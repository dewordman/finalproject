import pandas as pd
from pathlib import Path

file_path = Path("data/processed/results_rev4_masked/records_parsed.pkl")
df = pd.read_pickle(file_path)

# Find all rows where 'model_text' contains 'enron' (case-insensitive)
mask = df['model_text'].str.contains('enron', case=False, na=False)
guilty_rows = df[mask]

print(f"🚨 Found {len(guilty_rows)} rows that still contain 'enron'.\n")

if len(guilty_rows) > 0:
    print("👁️ Visual Inspection of the first 5 GUILTY rows:")
    for i, (index, row) in enumerate(guilty_rows.head(5).iterrows()):
        print(f"\n--- Guilty Row {i+1} (Source: {row['source_dataset']}) ---")
        # Print the FULL model_text so we can see exactly how 'enron' is hiding
        print(f"FULL TEXT:\n{row['model_text']}\n")
        
        # Let's also find exactly where 'enron' is in the string
        text_lower = row['model_text'].lower()
        idx = text_lower.find('enron')
        if idx != -1:
            start = max(0, idx - 30)
            end = min(len(text_lower), idx + 30)
            print(f"CONTEXT AROUND 'enron': ...{text_lower[start:end]}...")
else:
    print("✅ No rows contain 'enron'!")