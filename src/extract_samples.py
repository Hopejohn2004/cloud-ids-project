"""
Extract real attack samples from preprocessed data for dashboard simulation.
Run once: python src/extract_samples.py
"""

import pandas as pd
import numpy as np
import json
import os

DATA_PATH   = "data/preprocessed_data.csv"
CLASSES_PATH = "models/label_classes.npy"
OUTPUT_PATH  = "templates/attack_samples.json"

print("⏳ Loading data...")
df = pd.read_csv(DATA_PATH)
label_classes = np.load(CLASSES_PATH, allow_pickle=True)

# Build label index → name mapping
label_map = {i: str(name) for i, name in enumerate(label_classes)}

samples = {}
for idx, name in label_map.items():
    subset = df[df["Label"] == idx]
    if len(subset) == 0:
        continue
    row = subset.sample(1, random_state=42).drop(columns=["Label"]).iloc[0]
    samples[name] = {k: float(v) for k, v in row.items()}
    print(f"  ✅ {name} (encoded={idx})")

with open(OUTPUT_PATH, "w") as f:
    json.dump(samples, f)

print(f"\n✅ Saved {len(samples)} real attack samples to {OUTPUT_PATH}")
print(f"✅ Classes: {list(samples.keys())}")
