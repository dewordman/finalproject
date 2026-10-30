import sys
sys.path.insert(0, '.')
from pathlib import Path
import joblib

models_dir = Path('models/results_rev3')
print(f"Looking in: {models_dir.absolute()}")
print(f"Directory exists: {models_dir.exists()}")

if not models_dir.exists():
    print("\nERROR: Folder not found!")
    sys.exit(1)

files = list(models_dir.glob('*.joblib'))
print(f"\nFiles found ({len(files)}):")
for f in files:
    print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

if not files:
    print("\nNo .joblib files found!")
    sys.exit(1)

# Inspect each file
for f in files:
    print(f"\n{'='*60}")
    print(f"Inspecting: {f.name}")
    print(f"{'='*60}")
    
    try:
        obj = joblib.load(f)
        print(f"Type: {type(obj).__name__}")
        print(f"Has predict_proba: {hasattr(obj, 'predict_proba')}")
        print(f"Has predict: {hasattr(obj, 'predict')}")
        
        if hasattr(obj, 'named_steps'):
            print("\nPipeline steps:")
            for name, step in obj.named_steps.items():
                pp = "✓ predict_proba" if hasattr(step, 'predict_proba') else "✗ no predict_proba"
                print(f"  - {name}: {type(step).__name__} ({pp})")
        
        if hasattr(obj, 'calibrated_classifiers_'):
            print(f"\nCalibratedClassifierCV with {len(obj.calibrated_classifiers_)} sub-classifiers")
        
        # Try a test prediction
        if hasattr(obj, 'predict_proba'):
            try:
                test = obj.predict_proba(["subject: test body: hello world"])
                print(f"\n✓ predict_proba works! Shape: {test.shape}")
                print(f"  Probabilities: {test[0]}")
            except Exception as e:
                print(f"\n✗ predict_proba failed: {e}")
        else:
            print("\n✗ No predict_proba method on this object")
            
    except Exception as e:
        print(f"ERROR loading: {e}")