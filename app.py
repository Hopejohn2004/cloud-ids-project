from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import hashlib
import threading
import time

from src.storage import SQLiteStore
from src.response_engine import ResponseEngine

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

# ── Configuration ────────────────────────────────────────────────────────────
# Below this confidence the model is not trusted enough to commit to a single
# attack class; the verdict is labelled UNCERTAIN instead of the argmax class.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 55.0))

# Simple per-client rate limit for /predict (0 disables). Prevents the free
# demo being hammered as an unbounded compute endpoint.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", 60))

# Optional shared secret for live capture-agent posts to /predict. When set,
# requests MUST present the matching X-Sensor-Token header (401 otherwise).
# Leave empty to allow open access (dashboard demo mode). Sensors pass the
# same secret via --token / the IDS_SENSOR_TOKEN environment variable.
SENSOR_TOKEN = os.environ.get("IDS_SENSOR_TOKEN", "")

# Days to keep detections/actions before pruning (0 = keep forever).
IDS_RETENTION_DAYS = int(os.environ.get("IDS_RETENTION_DAYS", 0))

# ── Persistent state (SQLite, survives restarts) ─────────────────────────────
storage = SQLiteStore()
if IDS_RETENTION_DAYS > 0:
    pruned = storage.prune(IDS_RETENTION_DAYS)
    print(f"Retention: pruned {pruned} rows older than {IDS_RETENTION_DAYS} days")

# ── Simulated response engine per client ────────────────────────────────────
# Detection log, stats and response state live in SQLite; each client (browser
# device, or the shared bucket for header-less API callers) gets its own
# engine instance so devices never see each other's feed. Engine instances
# are just thin handles over the shared store, so this cache stays tiny.
MAX_CLIENTS = 200
ENGINE_CACHE = {}


def client_id():
    return request.headers.get("X-Client-Id", "shared") or "shared"


def client_state():
    """Return (creating if needed) the response-engine handle for this client."""
    cid = client_id()
    storage.register_client(cid)
    engine = ENGINE_CACHE.get(cid)
    if engine is None:
        if len(ENGINE_CACHE) >= MAX_CLIENTS:      # evict oldest bucket, keep cache bounded
            ENGINE_CACHE.pop(next(iter(ENGINE_CACHE)))
        engine = ResponseEngine(store=storage, client_id=cid)
        ENGINE_CACHE[cid] = engine
    return engine


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


# ── Minimal per-key rate limiter (in-memory token window) ────────────────────
class RateLimiter:
    """Sliding-window limiter. allow(key) returns True if a request may proceed."""

    def __init__(self, per_minute):
        self.period = per_minute
        self.window = 60.0
        self._hits = {}
        self._lock = threading.Lock()

    def allow(self, key):
        if self.period <= 0:
            return True
        now = time.time()
        with self._lock:
            if len(self._hits) > 10000:                      # prune stale keys
                cutoff = now - self.window
                self._hits = {k: [t for t in v if t >= cutoff] for k, v in self._hits.items() if v}
            window = self._hits.setdefault(key, [])
            while window and now - window[0] > self.window:
                window.pop(0)
            if len(window) >= self.period:
                return False
            window.append(now)
            return True


rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# ── Synthetic simulator source IP ────────────────────────────────────────────
# The dashboard's demo triggers don't send real packet fields, so for simulated
# traffic we mint a stable-looking attacker IP per client so the response
# engine's blocked-source deny list actually fills up. API callers that DO send
# a source_ip keep their own.
def synthetic_source_ip(cid):
    n = storage.stats(cid)["total"]
    digest = hashlib.sha1(f"{cid}:{n}".encode("utf-8")).hexdigest()
    return f"10.{int(digest[0:2], 16) % 256}.{int(digest[2:4], 16) % 256}.{n % 254 + 1}"


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
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "persistence": "sqlite",
        "sensor_auth": bool(SENSOR_TOKEN),
        "raw_sensors": len([c for c in storage.list_clients()
                            if c["client_id"] != "shared"]),
        "last_activity": storage.last_client_seen(),
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
    if SENSOR_TOKEN and request.headers.get("X-Sensor-Token") != SENSOR_TOKEN:
        return jsonify({"error": "Unauthorized sensor. Set IDS_SENSOR_TOKEN."}), 401

    if not rate_limiter.allow(client_id() + ":" + (request.remote_addr or "")):
        return jsonify({"error": "Rate limit exceeded. Try again shortly."}), 429

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

    # Confidence gate: below the threshold we don't trust the argmax commit,
    # so the verdict is "UNCERTAIN" instead of a specific attack class.
    if confidence < CONFIDENCE_THRESHOLD:
        attack_name = "UNCERTAIN"
        severity = "LOW"
        recommended_action = "ALERT"
    else:
        attack_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "UNKNOWN"
        severity, recommended_action = get_severity_action(attack_name)

    is_threat = bool(attack_name != "BENIGN")

    cid = client_id()
    if meta_fields["source_ip"] is None:
        meta_fields["source_ip"] = synthetic_source_ip(cid)

    # Execute the (simulated) response — BLOCK/ALERT/ALLOW via the response engine.
    engine = client_state()
    response = engine.handle(
        attack_name, severity, recommended_action,
        source_ip=meta_fields["source_ip"],
        confidence=f"{round(confidence, 2)}%",
    )
    executed_action = response.get("action", recommended_action)

    entry = {
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type":         str(attack_name),
        "severity":            str(severity),
        # "action" is what was actually (simulatedly) executed. recommended_action
        # is kept as the advisory alias used by any API consumers.
        "action":              str(executed_action),
        "recommended_action":  str(recommended_action),
        "response_executed":   bool(response.get("executed", False)),
        "response_detail":     response,
        # dashboards interpolate this directly (CONF: ${d.confidence}) with no
        # extra "%" added on their end, so the % must be baked in here.
        "confidence":          f"{round(confidence, 2)}%",
        "is_threat":           is_threat,
        "source_ip":           meta_fields["source_ip"],
        "destination_ip":      meta_fields["destination_ip"],
        "source_port":         meta_fields["source_port"],
        "destination_port":    meta_fields["destination_port"],
    }

    storage.add_detection(cid, entry)
    return jsonify(entry)


@app.route("/logs", methods=["GET"])
def logs():
    return jsonify(storage.recent_detections(client_id(), 20))


@app.route("/stats", methods=["GET"])
def get_stats():
    stats = storage.stats(client_id())
    rate = round((stats["threats"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
    return jsonify({**stats, "threat_rate": str(rate) + "%"})


@app.route("/distribution", methods=["GET"])
def distribution():
    """Attack-type distribution persisted in SQLite, survives refreshes/restarts."""
    return jsonify(storage.distribution(client_id()))


@app.route("/responses", methods=["GET"])
def responses():
    """State of the simulated response engine: blocked IPs and recent actions."""
    engine = client_state()
    return jsonify({"available": True, **engine.snapshot()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)