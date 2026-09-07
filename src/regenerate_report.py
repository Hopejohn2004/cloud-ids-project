"""
Regenerate the evaluation report WITHOUT retraining.

Loads the saved best model + leakage-free test split and re-creates:
  - notebooks/confusion_matrix.png
  - notebooks/classification_report.txt  (also printed to console)
  - notebooks/model_comparison.png       (rebuilt from model_metadata.json)

This is the fast path after the model has already been trained once — handy
for regenerating figures when only the report/notebook outputs are needed.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed/"
MODEL_DIR     = "models/"
OUTPUT_PATH   = "notebooks/"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

if not os.path.exists(os.path.join(PROCESSED_DIR, "train_test_split.npz")):
    raise FileNotFoundError(
        f"Missing {PROCESSED_DIR}train_test_split.npz — "
        "run src/preprocess_data.py and src/train_model.py first."
    )
if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
    raise FileNotFoundError("Missing models/best_model.pkl — run src/train_model.py first.")

print("Loading saved model + untouched test split...")
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
data = np.load(os.path.join(PROCESSED_DIR, "train_test_split.npz"))
X_test = data["X_test"]
y_test = data["y_test"]

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = joblib.load(f)
class_names = list(le.classes_)

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

print(f"Model: {metadata.get('model_name')} | classes: {len(class_names)} | "
      f"test samples: {len(y_test):,}")

y_pred = model.predict(X_test)

report_str = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

# ── 1. Classification report ────────────────────────────────────────────────
report_txt = os.path.join(OUTPUT_PATH, "classification_report.txt")
with open(report_txt, "w") as f:
    f.write(f"Classification Report — {metadata.get('model_name')} (untouched test set)\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n\n")
    f.write(report_str)
print(f"\n{'='*60}\nSAVED: {report_txt}\n{'='*60}\n")
print(report_str)

# ── 2. Confusion matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix — {metadata.get('model_name')} (untouched test set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))
plt.close()
print(f"Saved: notebooks/confusion_matrix.png")

# ── 3. Model comparison (rebuilt from saved metadata — no re-training) ──────
all_metrics = metadata.get("all_model_metrics", {})
if all_metrics:
    metrics_df = pd.DataFrame(
        {name: {k: v for k, v in m.items() if k in ("Accuracy", "Precision",
                                                     "Recall", "F1-Score")}
         for name, m in all_metrics.items()}
    ).T
    metrics_df.plot(kind="bar", figsize=(10, 5))
    plt.title("Model Comparison — CIC-IDS2017 (untouched test set)")
    plt.ylabel("Score (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison.png"))
    plt.close()
    print("Saved: notebooks/model_comparison.png")

print("\nReport regeneration complete.")