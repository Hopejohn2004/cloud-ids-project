"""
Model Training — CORRECTED
Trains Decision Tree, Random Forest and XGBoost on the leakage-free
train/test split produced by preprocess_data.py.
Evaluates ONLY on the untouched original test set.
Saves the best model + full metadata (no hard-coded metrics).
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import time
from datetime import datetime, timezone

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed/"
MODEL_DIR     = "models/"
OUTPUT_PATH   = "notebooks/"
RANDOM_STATE  = 42
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading leakage-free train/test split...")
data = np.load(os.path.join(PROCESSED_DIR, "train_test_split.npz"))
X_train_balanced = data["X_train_balanced"]
y_train_balanced = data["y_train_balanced"]
X_test_original  = data["X_test"]
y_test_original  = data["y_test"]

# Real class names, saved by preprocessing — NOT re-encoded here
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
class_names = list(le.classes_)

with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_names = json.load(f)

print(f"Train (balanced): {X_train_balanced.shape}")
print(f"Test  (untouched): {X_test_original.shape}")
print(f"Classes: {len(class_names)} -> {class_names}")

# ── MODELS ───────────────────────────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*45}")
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - start

    # Evaluate ONLY on the untouched original test set
    y_pred = model.predict(X_test_original)

    acc  = accuracy_score(y_test_original, y_pred) * 100
    prec = precision_score(y_test_original, y_pred, average="weighted", zero_division=0) * 100
    rec  = recall_score(y_test_original, y_pred, average="weighted", zero_division=0) * 100
    f1   = f1_score(y_test_original, y_pred, average="weighted", zero_division=0) * 100
    macro_f1 = f1_score(y_test_original, y_pred, average="macro", zero_division=0) * 100

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "Macro F1": macro_f1,
        "TrainTime": train_time, "model": model, "preds": y_pred
    }
    print(f"{name} — Accuracy {acc:.2f}% | Precision {prec:.2f}% | Recall {rec:.2f}% | "
          f"F1 {f1:.2f}% | Macro F1 {macro_f1:.2f}% | Train time {train_time:.1f}s")

# ── COMPARISON CHART ─────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(
    {k: {m: v for m, v in v.items() if m not in ["model", "preds"]} for k, v in results.items()}
).T
print(f"\n{'='*45}\nMODEL COMPARISON (evaluated on untouched test set)\n{'='*45}")
print(metrics_df.to_string())

plt.figure(figsize=(10, 5))
metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 5))
plt.title("Model Comparison — CIC-IDS2017 (untouched test set)")
plt.ylabel("Score (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison.png"))
plt.show()

# ── BEST MODEL ───────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["F1-Score"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\nBest Model: {best_name} — F1 {results[best_name]['F1-Score']:.2f}% (test set)")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

# ── CONFUSION MATRIX — real class names, not integers ────────────────────────
cm = confusion_matrix(y_test_original, best_preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix — {best_name} (untouched test set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))
plt.show()

report_str = classification_report(y_test_original, best_preds, target_names=class_names, zero_division=0)
print(f"\nClassification Report — {best_name}:\n{report_str}")

# ── SAVE MODEL METADATA (Flask/dashboard reads this, nothing hard-coded) ─────
metadata = {
    "model_name": best_name,
    "metrics": {k: v for k, v in results[best_name].items() if k not in ["model", "preds"]},
    "dataset": "CIC-IDS2017",
    "original_dataset_rows": 2830743,
    "training_sample_size": int(X_train_balanced.shape[0] + X_test_original.shape[0]),
    "feature_count": len(feature_names),
    "class_count": len(class_names),
    "class_names": class_names,
    "random_state": RANDOM_STATE,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "all_model_metrics": {
        name: {k: v for k, v in r.items() if k not in ["model", "preds"]}
        for name, r in results.items()
    }
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: models/best_model.pkl, models/model_metadata.json")
print("Training complete. Run app.py to start the API.")
