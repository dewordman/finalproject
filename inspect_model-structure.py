"""
inspect_model_structure.py
==========================
Recursively inspects the model object to find where TF-IDF and classifier are located.
"""

import joblib
from pathlib import Path

MODEL_PATH = Path("models/results_rev5/logreg_calibrated.joblib")
print(f"Loading model from: {MODEL_PATH}\n")
pipeline = joblib.load(MODEL_PATH)

print(f"Model type: {type(pipeline).__name__}")
print(f"Model module: {type(pipeline).__module__}\n")

def inspect_object(obj, path="pipeline", depth=0):
    """Recursively inspect an object and its attributes."""
    indent = "  " * depth
    prefix = f"{path} ({type(obj).__name__})"
    
    print(f"{indent}{prefix}")
    
    # Check if this object has named_steps (it's a Pipeline)
    if hasattr(obj, 'named_steps'):
        print(f"{indent}  ✅ FOUND named_steps: {list(obj.named_steps.keys())}")
        for step_name, step_obj in obj.named_steps.items():
            print(f"{indent}    - {step_name}: {type(step_obj).__name__}")
        return True
    
    # Check for common attributes that might contain the pipeline
    attrs_to_check = ['estimator', 'base_estimator', 'model', 'pipeline', 'clf', 
                      'classifier', 'inner_estimator', 'wrapped_model', 'steps',
                      'calibrated_classifiers_', 'classes_', 'coef_', 'intercept_']
    
    for attr_name in attrs_to_check:
        try:
            attr_val = getattr(obj, attr_name, None)
            if attr_val is not None:
                attr_type = type(attr_val).__name__
                print(f"{indent}  .{attr_name}: {attr_type}")
                
                # If it's a list/tuple, check first element
                if isinstance(attr_val, (list, tuple)) and len(attr_val) > 0:
                    print(f"{indent}    [0]: {type(attr_val[0]).__name__}")
                    # Recursively inspect first element
                    inspect_object(attr_val[0], f"{path}.{attr_name}[0]", depth + 2)
                # If it's a dict, check values
                elif isinstance(attr_val, dict):
                    for key, val in list(attr_val.items())[:3]:  # Limit to first 3
                        print(f"{indent}    ['{key}']: {type(val).__name__}")
                # Otherwise, recurse
                elif not isinstance(attr_val, (str, int, float, bool, type(None))):
                    inspect_object(attr_val, f"{path}.{attr_name}", depth + 1)
        except Exception as e:
            print(f"{indent}  .{attr_name}: ERROR - {e}")
    
    return False

print("=" * 70)
print("RECURSIVE INSPECTION")
print("=" * 70)
inspect_object(pipeline)

print("\n" + "=" * 70)
print("ATTEMPTING TO EXTRACT TF-IDF AND CLASSIFIER")
print("=" * 70)

# Try different navigation paths
attempts = [
    ("pipeline.named_steps", lambda p: p.named_steps if hasattr(p, 'named_steps') else None),
    ("pipeline.estimator.named_steps", lambda p: p.estimator.named_steps if hasattr(p, 'estimator') and hasattr(p.estimator, 'named_steps') else None),
    ("pipeline.estimator.estimator.named_steps", lambda p: p.estimator.estimator.named_steps if hasattr(p, 'estimator') and hasattr(p.estimator, 'estimator') and hasattr(p.estimator.estimator, 'named_steps') else None),
    ("pipeline.calibrated_classifiers_[0].estimator.named_steps", lambda p: p.calibrated_classifiers_[0].estimator.named_steps if hasattr(p, 'calibrated_classifiers_') and len(p.calibrated_classifiers_) > 0 and hasattr(p.calibrated_classifiers_[0].estimator, 'named_steps') else None),
]

for path_name, extractor in attempts:
    try:
        steps = extractor(pipeline)
        if steps:
            print(f"✅ SUCCESS via {path_name}")
            tfidf = steps.get('tfidf')
            clf = steps.get('clf') or steps.get('logreg')
            print(f"   TF-IDF: {type(tfidf).__name__ if tfidf else 'NOT FOUND'}")
            print(f"   Classifier: {type(clf).__name__ if clf else 'NOT FOUND'}")
            if tfidf and clf:
                print(f"\n🎯 WORKING PATH FOUND: {path_name}")
                break
        else:
            print(f"❌ Failed via {path_name}")
    except Exception as e:
        print(f"❌ Error via {path_name}: {e}")