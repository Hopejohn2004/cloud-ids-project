"""
Data Preprocessing — CORRECTED (leakage-free)
Cleans, encodes, splits, scales and balances the CIC-IDS2017 dataset.

Correct order (matches the required pipeline):
Raw -> Clean -> Stratified sample -> Filter rare classes -> Train/Test split
    -> Fit scaler ONLY on train -> Transform train/test
    -> SMOTE ONLY on train -> Save artifacts

The test set is NEVER touched by the scaler's fit() or by SMOTE.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import glob
import os
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH  = "data/raw/"          # put your original CIC-IDS2017 CSVs here
PROCESSED_DIR  = "data/processed/"
MODEL_DIR      = "models/"
SAMPLE_SIZE    = 80000
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
MIN_CLASS_COUNT = 6                    # classes with fewer rows than this are dropped
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── LOAD RAW DATA ─────────────────────────────────────────────────────────
print("Loading raw data...")
csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError(
        f"No CSVs found in {RAW_DATA_PATH}. "
        "Put ONLY the original CIC-IDS2017 files there (never preprocessed_data.csv)."
    )

dfs = []
for f in csv_files:
    df_tmp = pd.read_csv(f, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

# ── CLEAN ────────────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"After cleaning: {len(df):,} rows remaining")

# ── FEATURES & RAW STRING LABELS (encode later, after filtering) ─────────
label_col = "Label"
X_all = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
y_all_raw = df[label_col]   # keep as original string labels for now

constant_cols = [c for c in X_all.columns if X_all[c].nunique() <= 1]
X_all.drop(columns=constant_cols, inplace=True)
print(f"Features after removing constants: {X_all.shape[1]} columns")

# ── STRATIFIED SAMPLE (for memory) ────────────────────────────────────────
print(f"Sampling {SAMPLE_SIZE:,} rows to fit in memory...")
X_sample, _, y_sample_raw, _ = train_test_split(
    X_all, y_all_raw, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=y_all_raw
)
print(f"Sample ready: {X_sample.shape}")

# ── FILTER RARE CLASSES (on the sample, BEFORE encoding/splitting) ───────
class_counts = Counter(y_sample_raw)
dropped_classes = [c for c, n in class_counts.items() if n < MIN_CLASS_COUNT]
valid_classes = [c for c in class_counts if c not in dropped_classes]
if dropped_classes:
    print(f"Dropping rare classes (<{MIN_CLASS_COUNT} samples): "
          f"{[c.replace(chr(0xFFFD), '?') for c in dropped_classes]}")
mask = y_sample_raw.isin(valid_classes)
X_sample = X_sample[mask]
y_sample_raw = y_sample_raw[mask]
print(f"Kept {len(valid_classes)} classes, {len(X_sample):,} samples")

# ── ENCODE LABELS (fit ONCE, after filtering, so classes are sequential) ─
le = LabelEncoder()
y_sample_enc = le.fit_transform(y_sample_raw)
print("Final class mapping:")
for i, cls in enumerate(le.classes_):
    print(f"   {i} -> {cls.replace(chr(0xFFFD), '?')}")

# ── TRAIN/TEST SPLIT (before scaling, before SMOTE) ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_sample_enc
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── SCALE — fit ONLY on train ─────────────────────────────────────────────
print("Fitting scaler on training data only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # test set only ever TRANSFORMED, never fit
print("Scaling complete")

# ── SMOTE — applied ONLY to training data ─────────────────────────────────
print("Balancing training data with SMOTE (test set untouched)...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"Before SMOTE: {len(X_train_scaled):,} training samples")
print(f"After SMOTE : {len(X_train_balanced):,} training samples")
print(f"Test set (untouched): {len(X_test_scaled):,} samples")

# ── SAVE ARTIFACTS ─────────────────────────────────────────────────────────
feature_names = list(X_sample.columns)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

np.savez(
    os.path.join(PROCESSED_DIR, "train_test_split.npz"),
    X_train_balanced=X_train_balanced,
    y_train_balanced=y_train_balanced,
    X_test=X_test_scaled,
    y_test=y_test,
)

print("\nSaved:")
print("  models/scaler.pkl")
print("  models/label_encoder.pkl")
print("  models/feature_names.json")
print("  data/processed/train_test_split.npz  (X_train_balanced, y_train_balanced, X_test, y_test)")
print("\nPreprocessing complete (leakage-free). Run train_model.py next.")
