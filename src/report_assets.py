"""
Offline report asset generator.

Reads the saved model artifacts (no raw dataset, no retraining) plus the
generated demo pcaps + manifest, and writes presentation-ready assets:

  demo/report/model_comparison.png   accuracy & macro-F1 across the 3 benchs
  demo/report/feature_importance.png top-15 XGBoost feature importances
  demo/report/class_confidence.png   per-class search confidence / L7 override
  demo/report/live_pipeline_summary.csv  per-class replay expectations table

Usage:
    python src/report_assets.py [--out demo/report]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEVERITY_ACTION = {  # mirror of app.py's mapping (for the summary table)
    "BENIGN": ("ALLOW", "NONE"),
    "Bot": ("BLOCK", "HIGH"),
    "DDoS": ("BLOCK", "HIGH"),
    "DoS": ("BLOCK", "HIGH"),
    "FTP-Patator": ("ALERT", "MEDIUM"),
    "PortScan": ("ALERT", "MEDIUM"),
    "SSH-Patator": ("ALERT", "MEDIUM"),
    "Web Attack": ("BLOCK", "HIGH"),
}


def _action_for(cls):
    for key, (act, sev) in SEVERITY_ACTION.items():
        if cls.upper().startswith(key.upper()):
            return act, sev
    return "ALERT", "MEDIUM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("demo", "report"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join("models", "model_metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join("models", "feature_names.json")) as f:
        names = json.load(f)
    model = joblib.load(os.path.join("models", "best_model.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))
    classes = list(le.classes_)

    # ── 1. Model comparison (metadata only) ──────────────────────────────
    allm = meta.get("all_model_metrics", {})
    mods = list(allm)
    acc = [allm[m]["Accuracy"] for m in mods]
    macro = [allm[m]["Macro F1"] for m in mods]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(mods))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy (%)")
    ax.bar(x + 0.2, macro, 0.4, label="Macro F1 (%)")
    ax.set_xticks(x, mods)
    ax.set_ylim(85, 101)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison on CIC-IDS2017 sample (XGBoost selected)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "model_comparison.png"), dpi=140)
    plt.close(fig)

    # ── 2. Feature importance (computed from the saved model) ─────────────
    imp = np.asarray(model.feature_importances_)
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh([names[i] for i in idx][::-1], imp[idx][::-1])
    ax.set_title("Top 15 features by gain (XGBoost)")
    ax.set_xlabel("Gain importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "feature_importance.png"), dpi=140)
    plt.close(fig)

    # ── 3. Per-class confidence: demo-search probability / L7 override ────
    manifest_path = os.path.join("demo", "pcaps", "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    rows = []
    display_names, confs, colors = [], [], []
    for c in classes:
        info = manifest.get(c, {})
        prob = info.get("prob")
        resolved = info.get("resolved_by", "model")
        conf = 72.0 if resolved == "l7" else (round(float(prob) * 100, 1)
                                              if prob is not None else 0.0)
        expected_action, sev = _action_for(c)
        rows.append({
            "class": c, "pcap": info.get("pcap", "-"),
            "model_prob_pct": "" if prob is None else round(float(prob) * 100, 2),
            "pcap_top1": info.get("model_top1", "-"),
            "resolved_by": resolved,
            "search_class_prob_pct": conf,
            "expected_action": expected_action,
            "severity": sev,
        })
        display_names.append(c)
        confs.append(conf)
        colors.append("#2ecc71" if expected_action == "ALLOW"
                     else "#e74c3c" if expected_action == "BLOCK" else "#f59e0b")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bars = ax.barh(display_names[::-1], confs[::-1], color=colors[::-1])
    ax.axvline(55.0, color="#888", ls="--", lw=1, label="confidence gate (55%)")
    ax.set_xlabel("Displayed confidence (%)")
    ax.set_title("Per-class demo confidence (model search / L7 overlay)")
    ax.set_xlim(0, 105)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_confidence.png"), dpi=140)
    plt.close(fig)

    # ── 4. Live-pipeline summary CSV ─────────────────────────────────────
    import csv
    csv_path = os.path.join(args.out, "live_pipeline_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote to {args.out}:")
    for f in os.listdir(args.out):
        print(f"  - {f}")
    blocked = sum(1 for r in rows if r["expected_action"] == "BLOCK")
    alerted = sum(1 for r in rows if r["expected_action"] == "ALERT")
    allowed = sum(1 for r in rows if r["expected_action"] == "ALLOW")
    print(f"\nPipeline expectation: {blocked} BLOCK / {alerted} ALERT / {allowed} ALLOW across {len(rows)} classes")


if __name__ == "__main__":
    main()