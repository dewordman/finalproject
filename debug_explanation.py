"""
debug_explanations.py
=====================
Diagnoses why the app shows "No strongly influential terms identified"
"""

import sys
import joblib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load the model
MODEL_PATH = PROJECT_ROOT / "models" / "results_rev5" / "logreg_calibrated.joblib"
print(f"Loading model from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

print(f"\nModel type: {type(pipeline).__name__}")
print(f"Model attributes: {[a for a in dir(pipeline) if not a.startswith('_')]}")

# Test email (from your screenshot)
subject = "Action Required: Your Microsoft 365 Password Expires in 24 Hours"
body = """Dear Employee,
Your corporate Microsoft 365 password will expire in 24 hours. To prevent interruption to your email and cloud access, please re-verify your current password immediately. Click Here to Keep Your Password Active If you do not update your credentials within 24 hours, your account will be temporarily regards,
Internal IT Helpdesk"""

# Build model text (matching app logic)
from src.text_build import build_model_text

# Load mask terms
import yaml
with open(PROJECT_ROOT / "config_rev5.yaml", "r") as f:
    cfg = yaml.safe_load(f)
mask_terms = cfg.get("text", {}).get("mask_terms", [])

print(f"\nMask terms loaded: {len(mask_terms)} terms")

model_text = build_model_text(subject, body, mask_terms)
print(f"\nModel text length: {len(model_text)} characters")
print(f"Model text preview: {model_text[:200]}...")

# Try to find TF-IDF and classifier
def _find_tfidf_and_clf(pipeline):
    print("\n=== Navigating pipeline structure ===")
    
    # Case 1: standard sklearn Pipeline
    if hasattr(pipeline, 'named_steps'):
        print("✓ Has named_steps")
        tfidf = pipeline.named_steps.get('tfidf')
        clf = pipeline.named_steps.get('clf') or pipeline.named_steps.get('logreg')
        print(f"  TF-IDF found: {tfidf is not None}")
        print(f"  Classifier found: {clf is not None}")
        if tfidf and clf:
            return tfidf, clf

    # Case 2: CalibratedClassifierCV
    if hasattr(pipeline, 'calibrated_classifiers_'):
        print("✓ Has calibrated_classifiers_")
        first = pipeline.calibrated_classifiers_[0]
        est = getattr(first, "estimator", None)
        if est and hasattr(est, "named_steps"):
            print("  Inner estimator has named_steps")
            tfidf = est.named_steps.get('tfidf')
            clf = est.named_steps.get('clf') or est.named_steps.get('logreg')
            print(f"  TF-IDF found: {tfidf is not None}")
            print(f"  Classifier found: {clf is not None}")
            if tfidf and clf:
                return tfidf, clf

    # Case 3: custom SigmoidCalibratedClassifier
    print("\nTrying attribute search...")
    for attr in ['estimator', 'base_estimator', 'base_pipeline_', 'pipeline', 'clf', 'model']:
        inner = getattr(pipeline, attr, None)
        if inner is None:
            continue
        print(f"  ✓ Found attribute: {attr} (type: {type(inner).__name__})")
        if hasattr(inner, 'named_steps'):
            print(f"    Has named_steps: {list(inner.named_steps.keys())}")
            tfidf = inner.named_steps.get('tfidf')
            clf = inner.named_steps.get('clf') or inner.named_steps.get('logreg')
            print(f"    TF-IDF found: {tfidf is not None}")
            print(f"    Classifier found: {clf is not None}")
            if tfidf and clf:
                return tfidf, clf

    return None, None

tfidf, clf = _find_tfidf_and_clf(pipeline)

if tfidf is None or clf is None:
    print("\n❌ FAILED: Could not find TF-IDF or classifier in pipeline")
    print(f"  TF-IDF: {tfidf}")
    print(f"  Classifier: {clf}")
    sys.exit(1)

print(f"\n✓ Successfully found TF-IDF and classifier")
print(f"  TF-IDF type: {type(tfidf).__name__}")
print(f"  Classifier type: {type(clf).__name__}")

# Transform the text
print("\n=== Transforming text ===")
X = tfidf.transform([model_text])
print(f"  Feature matrix shape: {X.shape}")
print(f"  Non-zero features: {X.nnz}")

feature_names = tfidf.get_feature_names_out()
x_vals = X.toarray()[0]

print(f"\n  Top 10 features by TF-IDF value:")
top_indices = np.argsort(x_vals)[-10:][::-1]
for idx in top_indices:
    if x_vals[idx] > 0:
        print(f"    {feature_names[idx]}: {x_vals[idx]:.4f}")

# Get predictions
print("\n=== Getting predictions ===")
probs = pipeline.predict_proba([model_text])[0]
pred_idx = int(probs.argmax())

if hasattr(pipeline, "classes_"):
    predicted_class = str(pipeline.classes_[pred_idx])
else:
    predicted_class = ["Legitimate", "Spam", "Phishing"][pred_idx]

confidence = float(probs[pred_idx])
print(f"  Predicted class: {predicted_class}")
print(f"  Confidence: {confidence:.4f}")

# Calculate contributions
print("\n=== Calculating contributions ===")
class_labels = list(clf.classes_)
print(f"  Class labels: {class_labels}")

if predicted_class not in class_labels:
    print(f"  ❌ Predicted class '{predicted_class}' not in class labels")
    print(f"  Trying to match...")
    # Try case-insensitive match
    predicted_lower = predicted_class.lower()
    for i, label in enumerate(class_labels):
        if str(label).lower() == predicted_lower:
            print(f"  ✓ Found match: {label} at index {i}")
            idx = i
            break
    else:
        print(f"  ❌ No match found")
        sys.exit(1)
else:
    idx = class_labels.index(predicted_class)

coefs = clf.coef_[idx] if clf.coef_.ndim > 1 else clf.coef_
print(f"  Coefficient vector shape: {coefs.shape}")
print(f"  Coefficient range: [{coefs.min():.4f}, {coefs.max():.4f}]")

contributions = x_vals * coefs
print(f"  Contributions shape: {contributions.shape}")
print(f"  Contributions range: [{contributions.min():.4f}, {contributions.max():.4f}]")

# Get all non-zero contributions
ranked = sorted(
    [(feature_names[i], float(contributions[i])) for i in range(len(feature_names)) if contributions[i] != 0],
    key=lambda t: abs(t[1]),
    reverse=True,
)

print(f"\n  Total non-zero contributions: {len(ranked)}")
print(f"\n  Top 20 contributions (by absolute value):")
for term, contrib in ranked[:20]:
    direction = "POSITIVE" if contrib > 0 else "NEGATIVE"
    print(f"    {term}: {contrib:.4f} ({direction})")

if len(ranked) == 0:
    print("\n❌ No non-zero contributions found!")
    print("  This means the text was completely masked or has no overlapping vocabulary.")
else:
    print(f"\n✓ Found {len(ranked)} terms with non-zero contributions")