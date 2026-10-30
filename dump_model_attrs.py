import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/results_rev5/logreg_calibrated.joblib")
model = joblib.load(MODEL_PATH)

print("=== INSTANCE VARIABLES ===")
for k, v in model.__dict__.items():
    print(f"{k}: {type(v).__name__}")

print("\n=== SEARCHING FOR SKLEARN PIPELINE ===")
def find_pipeline(obj, path="model"):
    if isinstance(obj, Pipeline):
        print(f"FOUND PIPELINE at {path}")
        print(f"Steps: {list(obj.named_steps.keys())}")
        return True
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if find_pipeline(v, f"{path}.{k}"):
                return True
    return False

if not find_pipeline(model):
    print("Pipeline not found in instance variables. Checking all public attributes...")
    for attr in dir(model):
        if not attr.startswith('_'):
            try:
                val = getattr(model, attr)
                if isinstance(val, Pipeline):
                    print(f"FOUND PIPELINE at attribute '{attr}'")
                    print(f"Steps: {list(val.named_steps.keys())}")
            except Exception:
                pass