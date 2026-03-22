"""
Phase 2 — Dataset Exploration
Loads all CIC-IDS2017 CSV files and explores the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "notebooks/"
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load all CSV files
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

dfs = []
for f in csv_files:
    df_tmp = pd.read_csv(f, low_memory=False)
    df_tmp.columns = df_tmp.columns.str.strip()
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

print("\n========== DATASET OVERVIEW ==========")
print(f"Total Rows    : {len(df):,}")
print(f"Total Columns : {len(df.columns)}")
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Column Names ---")
print(list(df.columns))

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0] if missing[missing > 0].any() else "No missing values!")

print("\n========== ATTACK LABEL DISTRIBUTION ==========")
print(df[" Label"].value_counts() if " Label" in df.columns else df["Label"].value_counts())

# Plot attack distribution
label_col = " Label" if " Label" in df.columns else "Label"
label_counts = df[label_col].value_counts()

plt.figure(figsize=(14, 6))
sns.barplot(x=label_counts.values, y=label_counts.index, palette="viridis")
plt.title("Attack Label Distribution — CIC-IDS2017", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Attack Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "attack_distribution.png"))
print(f"\n✅ Chart saved to {OUTPUT_PATH}attack_distribution.png")
plt.show()
