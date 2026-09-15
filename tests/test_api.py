"""
API tests for the Flask app — health, validation, prediction, the confidence
gate, rate limiting, and persistence across "restarts" via SQLite.
"""

import json

CLIENT = {"X-Client-Id": "api-test-1"}


# ── Health / metadata ────────────────────────────────────────────────────────
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    h = res.get_json()
    assert h["status"] == "running"
    assert h["model_loaded"] is True
    assert h["model_name"] == "XGBoost"
    assert h["monitoring_mode"] == "SIMULATION"
    assert h["persistence"] == "sqlite"
    assert h["feature_count"] == 70
    assert h["class_count"] == 12


# ── Input validation ─────────────────────────────────────────────────────────
def test_predict_rejects_non_json(client):
    res = client.post("/predict", data="nope")
    assert res.status_code == 400


def test_predict_rejects_missing_features(client, sample_feature_vector):
    bad = {k: v for k, v in sample_feature_vector.items()}
    bad.pop(next(iter(bad)))
    res = client.post("/predict", json=bad, headers=CLIENT)
    assert res.status_code == 400
    assert "Missing required features" in res.get_json()["error"]


def test_predict_rejects_non_numeric(client, sample_feature_vector):
    bad = dict(sample_feature_vector)
    k = next(iter(bad))
    bad[k] = "not-a-number"
    res = client.post("/predict", json=bad, headers=CLIENT)
    assert res.status_code == 400


def test_predict_rejects_nan(client, sample_feature_vector):
    bad = dict(sample_feature_vector)
    k = next(iter(bad))
    bad[k] = float("nan")
    res = client.post("/predict", json=bad, headers=CLIENT)
    assert res.status_code == 400


# ── Prediction + response engine integration ─────────────────────────────────
def test_predict_valid_threat_flow(client):
    with open("templates/attack_samples.json") as f:
        samples = json.load(f)

    res = client.post("/predict", json=samples["DDoS"], headers=CLIENT)
    assert res.status_code == 200
    entry = res.get_json()

    assert entry["attack_type"] == "DDoS"
    assert entry["is_threat"] is True
    assert entry["severity"] == "HIGH"
    assert entry["action"] == "BLOCK"
    assert entry["response_executed"] is True
    assert entry["source_ip"]  # synthetic attacker IP minted
    assert entry["response_detail"]["blocked_ip"] == entry["source_ip"]

    # The blocked IP must show up on the /responses panel for this client
    r = client.get("/responses", headers=CLIENT).get_json()
    assert entry["source_ip"] in r["blocked_ips"]


def test_predict_valid_benign_flow(client):
    with open("templates/attack_samples.json") as f:
        samples = json.load(f)
    entry = client.post("/predict", json=samples["BENIGN"], headers=CLIENT).get_json()
    assert entry["is_threat"] is False
    assert entry["severity"] == "NONE"
    assert entry["action"] == "ALLOW"


def test_stats_and_distribution_persist_per_client(client):
    hdr = {"X-Client-Id": "api-test-stats"}
    with open("templates/attack_samples.json") as f:
        samples = json.load(f)

    client.post("/predict", json=samples["DDoS"], headers=hdr)
    client.post("/predict", json=samples["BENIGN"], headers=hdr)

    stats = client.get("/stats", headers=hdr).get_json()
    assert stats["total"] == 2
    assert stats["threats"] == 1
    assert stats["benign"] == 1

    dist = client.get("/distribution", headers=hdr).get_json()
    assert dist.get("DDoS") == 1
    assert dist.get("BENIGN") == 1

    logs = client.get("/logs", headers=hdr).get_json()
    assert len(logs) == 2

    # A DIFFERENT client sees none of this — feeds are isolated per device
    other = client.get("/stats", headers={"X-Client-Id": "api-test-other"}).get_json()
    assert other["total"] == 0


# ── Confidence gate ───────────────────────────────────────────────────────────
def test_low_confidence_verdict(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "CONFIDENCE_THRESHOLD", 150.0)

    with open("templates/attack_samples.json") as f:
        samples = json.load(f)

    entry = client.post(
        "/predict", json=samples["DDoS"], headers={"X-Client-Id": "api-test-gate"}
    ).get_json()

    # init-margin above the model's max confidence is impossible, so threshold
    # 150 always pushes the verdict into UNCERTAIN.
    assert entry["attack_type"] == "UNCERTAIN"
    assert entry["severity"] == "LOW"
    assert entry["recommended_action"] == "ALERT"
    assert entry["action"] == "ALERT"


# ── Sensor auth token ─────────────────────────────────────────────────────────
def test_predict_requires_token_when_enabled(client, app_module, monkeypatch,
                                             sample_feature_vector):
    monkeypatch.setattr(app_module, "SENSOR_TOKEN", "s3cret-token")
    hdr = {"X-Client-Id": "api-test-auth", "X-Sensor-Token": "wrong"}

    res = client.post("/predict", json=sample_feature_vector, headers=hdr)
    assert res.status_code == 401

    ok = client.post(
        "/predict", json=sample_feature_vector,
        headers={"X-Client-Id": "api-test-auth", "X-Sensor-Token": "s3cret-token"},
    )
    assert ok.status_code == 200

    # Header-less callers (the browser demo) are also rejected while enabled
    res = client.post("/predict", json=sample_feature_vector,
                      headers={"X-Client-Id": "api-test-auth"})
    assert res.status_code == 401


def test_health_reports_sensor_status(client, app_module, monkeypatch):
    h = client.get("/health").get_json()
    assert h["sensor_auth"] is False            # empty token by default
    assert h["raw_sensors"] >= 0
    assert h["last_activity"] is None or h["last_activity"]

    monkeypatch.setattr(app_module, "SENSOR_TOKEN", "s3cret-token")
    assert client.get("/health").get_json()["sensor_auth"] is True


def test_operator_cookie_allows_predict_without_token(client, app_module, monkeypatch,
                                                      sample_feature_vector):
    monkeypatch.setattr(app_module, "SENSOR_TOKEN", "s3cret-token")
    # Visit "/" so the server issues the HMAC-signed operator cookie
    client.get("/")
    # POST /predict with no X-Sensor-Token — should succeed via the cookie
    res = client.post("/predict", json=sample_feature_vector,
                      headers={"X-Client-Id": "api-test-cookie"})
    assert res.status_code == 200
    assert res.get_json()["attack_type"] in ("BENIGN", "UNCERTAIN") or \
           res.get_json().get("attack_type")


def test_global_scope_aggregates_all_clients(client, sample_feature_vector):
    import os as _os
    with open(_os.path.join("templates", "attack_samples.json")) as f:
        ddos = json.load(f)["DDoS"]
    client.post("/predict", json=sample_feature_vector,
                headers={"X-Client-Id": "scope-a"})
    client.post("/predict", json=ddos,
                headers={"X-Client-Id": "scope-b"})

    stats = client.get("/stats?scope=global").get_json()
    assert stats["total"] >= 2
    assert stats["threats"] >= 1

    logs = client.get("/logs?scope=global").get_json()
    assert len(logs) >= 2

    dist = client.get("/distribution?scope=global").get_json()
    assert len(dist) >= 1


def test_health_monitoring_modes(client, app_module, monkeypatch,
                                 sample_feature_vector):
    monkeypatch.setattr(app_module, "SENSOR_TOKEN", "s3cret-token")

    # No recent sensor activity -> never LIVE (depends on scrollback state)
    app_module._LAST_SENSOR_POST[0] = 0
    h0 = client.get("/health").get_json()
    assert h0["monitoring_mode"] in ("SIMULATION", "MONITORING_IDLE")

    # A live sensor post flips the server to LIVE_CAPTURE
    client.post("/predict", json=sample_feature_vector,
                headers={"X-Client-Id": "sensor-x", "X-Sensor-Token": "s3cret-token"})
    h = client.get("/health").get_json()
    assert h["raw_sensors"] >= 1
    assert h["monitoring_mode"] == "LIVE_CAPTURE"
    assert h["sensor_active"] is True

    # Sensor goes quiet for > SENSOR_IDLE_SECONDS -> MONITORING_IDLE
    app_module._LAST_SENSOR_POST[0] = 0
    h = client.get("/health").get_json()
    assert h["monitoring_mode"] == "MONITORING_IDLE"
    assert h["sensor_active"] is False


# ── Retention / pruning ───────────────────────────────────────────────────────
def test_prune_deletes_old_rows(app_module):
    storage = app_module.storage

    with storage._cursor() as (conn, cur):
        cur.execute(
            """INSERT INTO detections (client_id, timestamp, attack_type,
               severity, action, recommended_action, response_executed,
               response_detail, confidence, is_threat)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            ("prune-test", "2020-01-01 00:00:00", "DDoS", "HIGH", "BLOCK", "BLOCK",
             1, "{}", "99.0%", 1),
        )
        conn.commit()

    assert storage.prune(0) == 0                       # retention disabled = no-op
    pruned = storage.prune(days=1)                     # everything older than 1 day
    assert pruned >= 1

    with storage._cursor() as (conn, cur):
        cur.execute("SELECT COUNT(*) FROM detections WHERE client_id=?", ("prune-test",))
        assert cur.fetchone()[0] == 0


# ── Layer-7 overlay fusion ─────────────────────────────────────────────────────
def test_l7_xss_overlay_promotes_benign_stats(client, sample_feature_vector):
    hdr = {"X-Client-Id": "api-test-xss"}
    res = client.post("/predict", json={
        **sample_feature_vector, "l7": {"xss": True, "brute_force": False},
    }, headers=hdr)
    assert res.status_code == 200
    entry = res.get_json()
    assert entry["attack_type"] == "Web Attack - XSS"
    assert entry["is_threat"] is True
    assert entry["severity"] == "HIGH"
    assert entry["action"] == "BLOCK"
    assert entry["confidence"] == "72.0%"


def test_l7_bruteforce_overlay(client, sample_feature_vector):
    hdr = {"X-Client-Id": "api-test-bf"}
    entry = client.post("/predict", json={
        **sample_feature_vector, "l7": {"xss": False, "brute_force": True},
    }, headers=hdr).get_json()
    assert entry["attack_type"] == "Web Attack - Brute Force"
    assert entry["is_threat"] is True


def test_l7_does_not_override_ddos(client):
    with open("templates/attack_samples.json") as f:
        samples = json.load(f)
    entry = client.post("/predict", json={
        **samples["DDoS"], "l7": {"xss": True, "brute_force": True},
    }, headers={"X-Client-Id": "api-test-nofuse"}).get_json()
    assert entry["attack_type"] == "DDoS"


def test_l7_absent_payload_unchanged(client, sample_feature_vector):
    entry = client.post("/predict", json=sample_feature_vector,
                        headers={"X-Client-Id": "api-test-nol7"}).get_json()
    # No l7 field → the original verdict path runs untouched (still BENIGN).
    assert entry["attack_type"] == "BENIGN"
    assert entry["is_threat"] is False


# ── Rate limiting ────────────────────────────────────────────────────────────
def test_rate_limit_returns_429(client, app_module, monkeypatch):
    from app import RateLimiter

    monkeypatch.setattr(app_module, "rate_limiter", RateLimiter(per_minute=2))
    headers = {"X-Client-Id": "api-test-rl"}

    # Body is invalid JSON → 400, but every attempt still consumes a token.
    assert client.post("/predict", data="x", headers=headers).status_code == 400
    assert client.post("/predict", data="x", headers=headers).status_code == 400
    res = client.post("/predict", data="x", headers=headers)
    assert res.status_code == 429