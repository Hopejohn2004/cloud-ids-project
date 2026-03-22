"""
Phase 3 — Data Preprocessing
Cleans, encodes, scales and balances the CIC-IDS2017 dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import glob
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
MODEL_DIR   = "models/"
OUTPUT_PATH = "notebooks/"
SAMPLE_SIZE = 80000
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading data...")
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
dfs = []
for f in csv_files:
    df_tmp = pd.read_csv(f, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")

# ── CLEAN (memory-efficient) ─────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"✅ After cleaning: {len(df):,} rows remaining")

# ── ENCODE LABELS ────────────────────────────────────────────────────────────
label_col = "Label"
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df[label_col])
print("✅ Label encoding complete. Classes found:")
for i, cls in enumerate(le.classes_):
    print(f"   {i} → {cls}")

# ── FEATURES & TARGET ────────────────────────────────────────────────────────
X = df.drop(columns=[label_col, "label_encoded"]).select_dtypes(include=[np.number])
y = df["label_encoded"]

# Remove constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X.drop(columns=constant_cols, inplace=True)
print(f"✅ Features after removing constants: {X.shape[1]} columns")

# ── SAMPLE ───────────────────────────────────────────────────────────────────
print(f"⏳ Sampling {SAMPLE_SIZE:,} rows to fit in memory...")
X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=SAMPLE_SIZE, random_state=42, stratify=y
)
print(f"✅ Sample ready: {X_sample.shape}")

# ── SCALE ────────────────────────────────────────────────────────────────────
print("⏳ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
print("✅ Scaling complete")

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("✅ Scaler saved to models/scaler.pkl")

# ── BALANCE WITH SMOTE ───────────────────────────────────────────────────────
print("⏳ Filtering rare classes and balancing with SMOTE...")
# Remove classes with fewer than 6 samples
from collections import Counter
class_counts = Counter(y_sample)
valid_classes = [cls for cls, count in class_counts.items() if count >= 6]
mask = y_sample.isin(valid_classes)
X_scaled = X_scaled[mask]
y_sample = y_sample[mask]
print(f"✅ Kept {len(valid_classes)} classes with enough samples")

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y_sample)
print(f"✅ Balancing complete!")
print(f"   Before: {len(X_sample):,} samples")
print(f"   After : {len(X_balanced):,} samples")

# ── SAVE ─────────────────────────────────────────────────────────────────────
feature_names = list(X.columns)
df_out = pd.DataFrame(X_balanced, columns=feature_names)
df_out["Label"] = y_balanced
df_out.to_csv(os.path.join(DATA_PATH, "preprocessed_data.csv"), index=False)
print(f"✅ Saved to data/preprocessed_data.csv")

np.save(os.path.join(MODEL_DIR, "label_classes.npy"), le.classes_)
print(f"✅ Label classes saved to models/label_classes.npy")
print("\n🎉 Preprocessing complete! Run train_model.py next.")
