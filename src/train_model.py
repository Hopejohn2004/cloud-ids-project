"""
Phase 3 — Model Training
Trains Decision Tree, Random Forest and XGBoost on preprocessed data.
Saves the best model.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/preprocessed_data.csv"
MODEL_DIR   = "models/"
OUTPUT_PATH = "notebooks/"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("⏳ Loading preprocessed data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")

X = df.drop(columns=["Label"])
y = df["Label"]

print(f"✅ Features : {X.shape[1]} columns")
print(f"✅ Classes  : {y.nunique()} attack types")

# ── TRAIN/TEST SPLIT ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training samples : {len(X_train):,}")
print(f"✅ Testing samples  : {len(X_test):,}")

# Re-encode labels sequentially for XGBoost
le2 = LabelEncoder()
y_train_enc = le2.fit_transform(y_train)
y_test_enc  = le2.transform(y_test)

# ── MODELS ───────────────────────────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*45}")
    print(f"⏳ Training {name}...")
    model.fit(X_train, y_train_enc)
    print(f"⏳ Evaluating {name}...")
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test_enc, y_pred) * 100
    prec = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100
    rec  = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100
    f1   = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0) * 100
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "model": model, "preds": y_pred}
    print(f"📊 {name} Results:")
    print(f"   Accuracy  : {acc:.2f}%")
    print(f"   Precision : {prec:.2f}%")
    print(f"   Recall    : {rec:.2f}%")
    print(f"   F1-Score  : {f1:.2f}%")

# ── COMPARISON CHART ─────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print("📊 MODEL COMPARISON SUMMARY")
print(f"{'='*45}")
metrics_df = pd.DataFrame({k: {m: v for m, v in v.items() if m not in ["model","preds"]} for k, v in results.items()}).T
print(metrics_df.to_string())

plt.figure(figsize=(10, 5))
metrics_df[["Accuracy","Precision","Recall","F1-Score"]].plot(kind="bar", figsize=(10,5))
plt.title("Model Comparison — CIC-IDS2017")
plt.ylabel("Score (%)")
plt.xticks(rotation=0)
plt.ylim(95, 101)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison.png"))
print(f"✅ Chart saved to notebooks/model_comparison.png")
plt.show()

# ── BEST MODEL ───────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["F1-Score"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\n🏆 Best Model: {best_name} with F1-Score of {results[best_name]['F1-Score']:.2f}%")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
print(f"✅ Best model saved to models/best_model.pkl")

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
print(f"\n⏳ Generating confusion matrix for {best_name}...")
label_classes = np.load("models/label_classes.npy", allow_pickle=True)
cm = confusion_matrix(y_test_enc, best_preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le2.classes_, yticklabels=le2.classes_)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"))
print(f"✅ Confusion matrix saved to notebooks/confusion_matrix.png")
plt.show()

print(f"\n📋 Full Classification Report — {best_name}:")
print(classification_report(y_test_enc, best_preds, zero_division=0))
print("\n🎉 Training complete! Run app.py to start the API.")
