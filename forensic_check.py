import pandas as pd
import os
from pathlib import Path

# The exact file path from your last run
file_path = Path("data/processed/results_rev4_masked/records_parsed.pkl")

if not file_path.exists():
    print(f"❌ FILE NOT FOUND: {file_path}")
    print("The parse stage did not generate this file! Check your console for errors.")
else:
    print(f"✅ File found: {file_path}")
    
    # Get the exact time the file was last modified
    mtime = os.path.getmtime(file_path)
    from datetime import datetime
    print(f"🕒 Last Modified: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = pd.read_pickle(file_path)
    
    # 1. The ultimate count
    enron_count = df['model_text'].str.contains('enron', case=False).sum()
    print(f"\n🔢 'enron' count in model_text: {enron_count}")
    
    # 2. Check if the rows were actually processed
    print("\n📊 Parse Status Distribution:")
    print(df['parse_status'].value_counts())
    
    # 3. Visual inspection: Let's look at the actual text!
    print("\n👁️ Visual Inspection of first 3 'ok' rows:")
    ok_rows = df[df['parse_status'] == 'ok'].head(3)
    for i, (_, row) in enumerate(ok_rows.iterrows()):
        print(f"\n--- Row {i+1} (Source: {row['source_dataset']}) ---")
        # Print the first 250 characters of the model_text
        print(f"Model Text: {row['model_text'][:250]}...")