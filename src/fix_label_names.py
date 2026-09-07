"""
One-time label cleanup for the raw CIC-IDS2017 CSVs.

The raw dataset's "Web Attack - XSS" / "Web Attack - Brute Force" labels arrive
with a mojibake/encoding artifact (the Unicode replacement character U+FFFD)
where the separator dash should be, e.g. "Web Attack <U+FFFD> XSS".

The ML pipeline already works around this, but running this script ONCE cleanly
rewrites those labels in place so all downstream code can match on the clean
"Web Attack - XSS" / "Web Attack - Brute Force" strings: a consistent schema
between the raw data, label encoder and the dashboard simulation buttons.

Usage:
    python src/fix_label_names.py            # fix in place (recommended once)
    python src/fix_label_names.py --dry-run  # preview only, change nothing
"""

import glob
import os
import re
import sys

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/"
LABEL_COLUMN  = "Label"
# ────────────────────────────────────────────────────────────────────────────

# Any run of the replacement char (and surrounding whitespace) is treated as
# the ' - ' separator that the clean class names use.
FIX_RE = re.compile(r"\s*\ufffd\s*")


def clean_label(value):
    """Normalise one raw label; returns (clean_text, changed_flag)."""
    text = str(value).strip()
    new_text = FIX_RE.sub(" - ", text)
    return new_text, new_text != text


def process_file(path, dry_run):
    """Rewrite Label column of one CSV. Returns a summary dict."""
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    # Raw CIC-IDS2017 headers carry a BOM/leading whitespace (e.g. " Label");
    # normalise them the same way preprocess_data.py does.
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    if LABEL_COLUMN not in df.columns:
        return {"path": path, "rows": 0, "changed": 0, "skipped": "no Label column"}

    changed_rows = 0
    new_labels = df[LABEL_COLUMN].map(clean_label)
    for new, (_, orig) in zip(new_labels, df[LABEL_COLUMN].items()):
        if new[1]:
            changed_rows += 1

    if not dry_run and changed_rows:
        df[LABEL_COLUMN] = [new[0] for new in new_labels]
        df.to_csv(path, index=False)

    return {"path": path, "rows": len(df), "changed": changed_rows}


def main():
    dry_run = "--dry-run" in sys.argv

    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {RAW_DATA_PATH}")
        return 1

    print(f"{'DRY RUN' if dry_run else 'FIXING'} labels in {len(csv_files)} file(s)...")
    total_fixed = 0
    for path in sorted(csv_files):
        result = process_file(path, dry_run)
        if result.get("skipped"):
            print(f"  SKIP {os.path.basename(path):<45} {result['skipped']}")
            continue
        print(f"  {'WOULD FIX' if dry_run else 'fixed':<10} {os.path.basename(path):<45} "
              f"{result['changed']}/{result['rows']} rows")
        total_fixed += result["changed"]

    print(f"\n{total_fixed} label(s) affected "
          f"({'preview only — rerun without --dry-run to apply' if dry_run else '— run complete'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())