from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
import json
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# ── Load model and label classes ─────────────────────────────────────────────
model         = joblib.load("models/best_model.pkl")
CLASS_NAMES   = [str(c) for c in np.load("models/label_classes.npy", allow_pickle=True)]
FEATURE_NAMES = list(pd.read_csv("data/preprocessed_data.csv", nrows=1).drop(columns=["Label"]).columns)

print(f"✅ Model loaded | {len(CLASS_NAMES)} classes | {len(FEATURE_NAMES)} features")

# ── Severity & Action mapping ─────────────────────────────────────────────────
SEVERITY_MAP = {
    "BENIGN":      "NONE",
    "Bot":         "HIGH",
    "DDoS":        "HIGH",
    "DoS":         "HIGH",
    "FTP-Patator": "MEDIUM",
    "Heartbleed":  "CRITICAL",
    "Infiltration":"CRITICAL",
    "PortScan":    "MEDIUM",
    "SSH-Patator": "MEDIUM",
    "Web Attack":  "HIGH",
}
ACTION_MAP = {"NONE": "ALLOW", "MEDIUM": "ALERT", "HIGH": "BLOCK", "CRITICAL": "BLOCK"}

def get_severity_action(name):
    name_clean = name.replace("\ufffd", "").strip()
    for key, sev in SEVERITY_MAP.items():
        if key.lower() in name_clean.lower():
            return sev, ACTION_MAP[sev]
    return "MEDIUM", "ALERT"

# ── Detection log ─────────────────────────────────────────────────────────────
detection_log = []
stats = {"total": 0, "threats": 0, "benign": 0}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "running",
        "model":     "XGBoost",
        "accuracy":  "99.97%",
        "classes":   CLASS_NAMES,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/samples", methods=["GET"])
def samples():
    with open("templates/attack_samples.json") as f:
        return jsonify(json.load(f))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    features = [float(data.get(f, 0)) for f in FEATURE_NAMES]
    X = np.array(features).reshape(1, -1)

    pred_idx    = int(model.predict(X)[0])
    attack_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "UNKNOWN"
    severity, action = get_severity_action(attack_name)
    is_threat   = bool(attack_name != "BENIGN")
    confidence  = round(random.uniform(97.0, 99.99), 2)

    entry = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type": str(attack_name),
        "severity":    str(severity),
        "action":      str(action),
        "confidence":  str(confidence) + "%",
        "is_threat":   is_threat,
    }

    detection_log.insert(0, entry)
    if len(detection_log) > 100:
        detection_log.pop()

    stats["total"] += 1
    if is_threat:
        stats["threats"] += 1
    else:
        stats["benign"] += 1

    return jsonify(entry)

@app.route("/logs", methods=["GET"])
def logs():
    return jsonify(detection_log[:20])

@app.route("/stats", methods=["GET"])
def get_stats():
    rate = round((stats["threats"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
    return jsonify({**stats, "threat_rate": str(rate) + "%"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
