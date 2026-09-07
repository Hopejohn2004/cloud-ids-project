"""
Extract real attack samples from raw CIC-IDS2017 data for dashboard simulation,
matched by string label against the current model's schema
"""

import pandas as pd
import numpy as np
import joblib
import json
import glob
import os

RAW_DATA_PATH   = "data/raw/"
FEATURE_NAMES_PATH = "models/feature_names.json"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_PATH     = "templates/attack_samples.json"
RANDOM_STATE    = 42

print("Loading current feature schema and class list...")
with open(FEATURE_NAMES_PATH) as f:
    FEATURE_NAMES = json.load(f)

label_encoder = joblib.load(LABEL_ENCODER_PATH)
CURRENT_CLASSES = list(label_encoder.classes_)
# Undo the earlier '-' cleanup just for matching against the raw CSV's
# original (possibly mangled) label text, since raw data still has it.
RAW_LABEL_LOOKUP = {c.replace(" - ", "\ufffd").replace(" - ", " "): c for c in CURRENT_CLASSES}
RAW_LABEL_LOOKUP.update({c: c for c in CURRENT_CLASSES})  # exact matches too

print(f"Current model classes ({len(CURRENT_CLASSES)}): {CURRENT_CLASSES}")

print("Loading raw CSVs...")
csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSVs found in {RAW_DATA_PATH}")

dfs = []
for f in csv_files:
    df_tmp = pd.read_csv(f, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    dfs.append(df_tmp)
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows")

# Clean, same as preprocessing (so sampled values are realistic/valid)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)

df["Label_clean"] = df["Label"].astype(str).str.strip()

import re

samples = {}
missing = []

def normalize(s: str) -> str:
    """Keep only lowercase letters/digits, stripping spaces, hyphens, and any
    encoding-artifact characters (like the mangled U+FFFD in Web Attack labels)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

for clean_class_name in CURRENT_CLASSES:
    candidates = df[df["Label_clean"] == clean_class_name]
    if len(candidates) == 0:
        mangled_variant = clean_class_name.replace(" - ", "\ufffd")
        candidates = df[df["Label_clean"] == mangled_variant]
    if len(candidates) == 0:
        target_norm = normalize(clean_class_name)
        mask = df["Label_clean"].apply(lambda x: normalize(x) == target_norm)
        candidates = df[mask]

    if len(candidates) == 0:
        missing.append(clean_class_name)
        print(f"  MISSING: no raw rows found for {clean_class_name!r}")
        continue

    row = candidates.sample(1, random_state=RANDOM_STATE).iloc[0]

    # Only keep the exact features the model expects, in the exact saved order
    try:
        sample_dict = {f: float(row[f]) for f in FEATURE_NAMES}
    except KeyError as e:
        raise KeyError(
            f"Feature {e} from feature_names.json not found in raw CSV columns. "
            "Raw CSV schema may not match what preprocess_data.py produced."
        )

    samples[clean_class_name] = sample_dict
    print(f"  OK: {clean_class_name}  (matched raw label: {row['Label_clean']!r})")

with open(OUTPUT_PATH, "w") as f:
    json.dump(samples, f, indent=2)

print(f"\nSaved {len(samples)}/{len(CURRENT_CLASSES)} samples to {OUTPUT_PATH}")
if missing:
    print(f"WARNING — could not find raw rows for: {missing}")
    print("Buttons for these classes will show 'undefined' behaviour until fixed.")
else:
    print("All current model classes have a matching sample. Buttons should be accurate now.")
