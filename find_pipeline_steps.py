"""
find_pipeline_steps.py
======================
Recursively searches the model object to find where 'tfidf' and 'clf' are hiding.
"""
import joblib
from pathlib import Path

MODEL_PATH = Path("models/results_rev5/logreg_calibrated.joblib")
print(f"Loading model from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

def inspect_object(obj, indent=0):
    prefix = "  " * indent
    print(f"{prefix}Type: {type(obj).__name__}")
    
    # Check if this object is the Pipeline we are looking for
    if hasattr(obj, 'named_steps'):
        print(f"{prefix}✅ FOUND named_steps: {list(obj.named_steps.keys())}")
        return obj.named_steps
        
    # Look for common wrapper attributes
    for attr_name in ['estimator', 'base_estimator', 'model', 'pipeline', 'steps', 'inner_estimator', 'wrapped_model']:
        attr_val = getattr(obj, attr_name, None)
        if attr_val is not None:
            print(f"{prefix}Checking attribute: '{attr_name}'")
            result = inspect_object(attr_val, indent + 1)
            if result:
                return result
                
    # If it's a list or tuple (like calibrated_classifiers_), check the first element
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        print(f"{prefix}Checking first element of list/tuple...")
        result = inspect_object(obj[0], indent + 1)
        if result:
            return result
            
    return None

print("\n=== Starting Deep Inspection ===")
steps = inspect_object(pipeline)

if steps:
    tfidf = steps.get('tfidf')
    clf = steps.get('clf') or steps.get('logreg')
    print(f"\n✅ SUCCESS! Found pipeline steps.")
    print(f"  TF-IDF type: {type(tfidf).__name__}")
    print(f"  Classifier type: {type(clf).__name__}")
else:
    print("\n❌ FAILED: Could not find 'named_steps' anywhere in the object tree.")
    print("Please share the full output of this script so we can fix the app.")