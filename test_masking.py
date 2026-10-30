import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.text_build import delete_terms, normalise, build_model_text

# Load the config
with open('config_rev3.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# Get the mask terms
terms = cfg.get('text', {}).get('mask_terms', [])
print(f"Loaded {len(terms)} mask terms from config")
print(f"First 5 terms: {terms[:5]}")
print(f"'enron' in terms: {'enron' in terms}")

# Test the delete_terms function directly
test_text = "this is enron and enron.com and www.enron.com"
print(f"\nOriginal text: {test_text}")

result = delete_terms(test_text, terms, mode='delete')
print(f"After delete_terms: {result}")
print(f"'enron' in result: {'enron' in result.lower()}")

# Test the full normalise function
result2 = normalise(test_text, mask_terms=terms, mask_dates=False)
print(f"\nAfter normalise: {result2}")
print(f"'enron' in result2: {'enron' in result2.lower()}")

# Test build_model_text
result3 = build_model_text("test subject with enron", "test body with enron.com", 
                           mask_terms=terms, mask_dates=False)
print(f"\nAfter build_model_text: {result3}")
print(f"'enron' in result3: {'enron' in result3.lower()}")