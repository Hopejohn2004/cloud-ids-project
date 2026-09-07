from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)

# ── Load model artifacts (no raw dataset dependency) ─────────────────────────
MODEL_DIR = "models/"

model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    MODEL_METADATA = json.load(f)

CLASS_NAMES = list(label_encoder.classes_)

print(f"Model loaded: {MODEL_METADATA.get('model_name', 'unknown')} | "
      f"{len(CLASS_NAMES)} classes | {len(FEATURE_NAMES)} features")

# ── Severity & recommended-action mapping ─────────────────────────────────────
SEVERITY_MAP = {
    "BENIGN":       "NONE",
    "Bot":          "HIGH",
    "DDoS":         "HIGH",
    "DoS":          "HIGH",
    "FTP-Patator":  "MEDIUM",
    "Heartbleed":   "CRITICAL",
    "Infiltration": "CRITICAL",
    "PortScan":     "MEDIUM",
    "SSH-Patator":  "MEDIUM",
    "Web Attack":   "HIGH",
}
ACTION_MAP = {"NONE": "ALLOW", "MEDIUM": "ALERT", "HIGH": "BLOCK", "CRITICAL": "BLOCK"}


def get_severity_action(name):
    name_clean = name.replace("\ufffd", "-").strip()
    for key, sev in SEVERITY_MAP.items():
        if key.lower() in name_clean.lower():
            return sev, ACTION_MAP[sev]
    return "MEDIUM", "ALERT"


def validate_predict_input(data):
    """
    Returns (is_valid, error_message_or_None).
    - Every feature in FEATURE_NAMES must be present.
    - Every value must be numeric, not NaN, not infinite.
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object of feature_name: value pairs."

    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return False, f"Missing required features: {missing}"

    bad_values = []
    for f in FEATURE_NAMES:
        val = data[f]
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            bad_values.append(f)
            continue
        if np.isnan(val_f) or np.isinf(val_f):
            bad_values.append(f)

    if bad_values:
        return False, f"Non-numeric, NaN, or infinite values for features: {bad_values}"

    return True, None


# ── Detection log ─────────────────────────────────────────────────────────────
detection_log = []
stats = {"total": 0, "threats": 0, "benign": 0}

# ── Simulated response engine ─────────────────────────────────────────────────
# Turns the recommended action into an "executed" mitigation (simulated only —
# e.g. BLOCK adds the source IP to an in-memory deny list). See src/response_engine.py.
try:
    from src.response_engine import ResponseEngine
    response_engine = ResponseEngine()
except Exception as e:   # never crash startup over the response layer
    print(f"[warn] Response engine unavailable: {e}")
    response_engine = None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    metrics = MODEL_METADATA.get("metrics", {})
    return jsonify({
        "status":               "running",
        "model_loaded":         True,
        "model_name":           MODEL_METADATA.get("model_name", "unknown"),
        "dataset":              MODEL_METADATA.get("dataset", "unknown"),
        "original_dataset_rows": MODEL_METADATA.get("original_dataset_rows"),
        "training_sample_size": MODEL_METADATA.get("training_sample_size"),
        "feature_count":        len(FEATURE_NAMES),
        "class_count":          len(CLASS_NAMES),
        "classes":              CLASS_NAMES,
        "metrics": {
            "accuracy":  round(metrics.get("Accuracy", 0), 2),
            "precision": round(metrics.get("Precision", 0), 2),
            "recall":    round(metrics.get("Recall", 0), 2),
            "f1_score":  round(metrics.get("F1-Score", 0), 2),
            "macro_f1":  round(metrics.get("Macro F1", 0), 2),
        },
        "trained_at":  MODEL_METADATA.get("trained_at"),
        "monitoring_mode": "SIMULATION",  # honest: not connected to live traffic capture yet
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Full metadata for all trained models, for a dashboard 'model intelligence' panel."""
    return jsonify(MODEL_METADATA)


@app.route("/samples", methods=["GET"])
def samples():
    with open("templates/attack_samples.json") as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Optional network-flow context fields (not used for prediction, just logging)
    meta_fields = {
        "source_ip":      data.get("source_ip"),
        "destination_ip": data.get("destination_ip"),
        "source_port":    data.get("source_port"),
        "destination_port": data.get("destination_port"),
    }

    is_valid, error_msg = validate_predict_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # Build DataFrame in the EXACT saved feature order, then apply the SAME
    # preprocessing used at training time (scaler only — never re-fit here).
    row = pd.DataFrame([[float(data[f]) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(row)

    probabilities = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx]) * 100

    attack_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "UNKNOWN"
    severity, recommended_action = get_severity_action(attack_name)
    is_threat = bool(attack_name != "BENIGN")

    # Execute the (simulated) response — BLOCK/ALERT/ALLOW via the response engine.
    if response_engine is not None:
        response = response_engine.handle(
            attack_name, severity, recommended_action,
            source_ip=meta_fields["source_ip"],
            confidence=f"{round(confidence, 2)}%",
        )
        executed_action = response.get("action", recommended_action)
        response_detail = response
    else:
        executed_action = recommended_action
        response_detail = {"executed": False, "detail": "Response engine unavailable."}

    entry = {
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type":         str(attack_name),
        "severity":            str(severity),
        # "action" is what was actually (simulatedly) executed now that the
        # response engine is wired in. recommended_action is kept as the
        # advisory alias used by any API consumers.
        "action":              str(executed_action),
        "recommended_action":  str(recommended_action),
        "response_executed":   bool(response_detail.get("executed", False)),
        "response_detail":     response_detail,
        # dashboards interpolate this directly (CONF: ${d.confidence}) with no
        # extra "%" added on their end, so the % must be baked in here, matching
        # the old behaviour's format even though the number itself is now real.
        "confidence":          f"{round(confidence, 2)}%",
        "is_threat":           is_threat,
        "source_ip":           meta_fields["source_ip"],
        "destination_ip":      meta_fields["destination_ip"],
        "source_port":         meta_fields["source_port"],
        "destination_port":    meta_fields["destination_port"],
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


@app.route("/distribution", methods=["GET"])
def distribution():
    """Attack-type distribution derived from the backend log, so the dashboard
    doesn't need to keep its own in-browser counts that vanish on refresh."""
    counts = {}
    for entry in detection_log:
        name = entry["attack_type"]
        counts[name] = counts.get(name, 0) + 1
    return jsonify(counts)


@app.route("/responses", methods=["GET"])
def responses():
    """State of the simulated response engine: blocked IPs and recent actions."""
    if response_engine is None:
        return jsonify({"available": False}), 503
    return jsonify({"available": True, **response_engine.snapshot()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
