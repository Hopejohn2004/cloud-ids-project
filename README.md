# 🛡️ ML-Based Cloud Intrusion Detection & Threat Response System

A machine learning-powered network intrusion detection system built as a Final Year Project. Trained on a stratified 80,000-record sample drawn from the CIC-IDS2017 dataset (2,830,743 raw records), achieving **99.79% weighted accuracy** using XGBoost.

> 🌐 **Live demo:** https://cloud-ids-c88k.onrender.com

---

## 🎯 Features

- **99.79% Accuracy** — XGBoost model, evaluated on an untouched 16,000-record test set
- **Simulated Real-Time Detection** — Flask API classifies submitted traffic samples instantly
- **Live Capture Pipeline** — `capture_agent.py` replays pcaps or sniffs live interfaces, extracts CIC-style flow features (70) from packets, and posts them to `/predict` in real time
- **12 Attack Classes** — DDoS, DoS variants, PortScan, Bot, Brute Force, and more
- **Automated Response Engine** — severity-based BLOCK/ALERT/ALLOW decisions are actually *executed* (blocked source IPs, alert/action log — see `/responses`)
- **Persistent State (SQLite)** — detection logs, stats, and the blocked-IP deny list survive restarts (no longer in-memory only)
- **Confidence Gate** — predictions below a confidence threshold are flagged `UNCERTAIN` instead of committing blindly
- **Rate Limiting** — per-client throttling on `/predict` so the free demo isn't abused as a compute endpoint
- **Live Dashboard** — Real-time detection feed, attack-distribution chart, blocked-sources panel & response ledger, all pulled live from the backend
- **REST API** — `/predict`, `/health`, `/logs`, `/stats`, `/model-info`, `/distribution`, `/responses` endpoints
- **Automated Tests** — `pytest` suite covering the response engine, all API endpoints, the flow-feature extractor, and model compatibility (38 tests)

---

## 🔍 Detected Attack Types

| Attack Type | Severity | Recommended Action |
|-------------|----------|---------------------|
| DDoS | HIGH | BLOCK |
| DoS Hulk | HIGH | BLOCK |
| DoS GoldenEye | HIGH | BLOCK |
| DoS slowloris | HIGH | BLOCK |
| DoS Slowhttptest | HIGH | BLOCK |
| Bot | HIGH | BLOCK |
| Web Attack - Brute Force | HIGH | BLOCK |
| Web Attack - XSS | HIGH | BLOCK |
| PortScan | MEDIUM | ALERT |
| FTP-Patator | MEDIUM | ALERT |
| SSH-Patator | MEDIUM | ALERT |
| BENIGN | NONE | ALLOW |

*Note: Heartbleed, Infiltration, and Web Attack SQL Injection were present in the original CIC-IDS2017 dataset but excluded from the final model — see Dataset section below.*

---

## 📊 ML Model Performance

Evaluated on a held-out, untouched 16,000-record test set:

| Model | Accuracy | Weighted F1-Score |
|-------|----------|--------------------|
| Decision Tree | 99.72% | 99.72% |
| Random Forest | 99.63% | 99.68% |
| **XGBoost** ⭐ | **99.79%** | **99.80%** |

XGBoost was selected as the production model based on superior weighted performance. Full per-class results, including macro-averaged F1-score, are documented in the project report (Chapter Four).

---

## ⚠️ Limitations (Honest Disclosure)

- **Simulation mode remains the default** — the dashboard classifies sample feature vectors; live analysis requires running `capture_agent.py` against a reachable server (local or temporary public endpoint)
- **CIC fidelity** — the packet extractor (`flow_features.py`) reproduces the CIC-IDS2017 feature semantics the model was trained on (payload-based lengths, microsecond IAT/burst timing, the dataset's near-zero flag-count artifact). Organic traffic that differs from the benchmark attack shapes may be misclassified, and single-flow statistics cannot separate near-identical web attacks (e.g. XSS vs Brute Force) on their own. `src/synth_attacks.py` generates per-class packet traces that reproduce each attack's benchmark signature for reliable demos
- **Response engine is simulated, not real enforcement** — BLOCK actions maintain a deny list (persisted to SQLite) and log firewall-style commands (`/responses`); no actual network traffic is dropped. It demonstrates the response layer a production IDS would hand off to a firewall/EDR.
- **State persists across redeploys** — logs, stats, and the blocked-IP deny list are stored in a local SQLite file (`ids_state.db`); on Render this lives on a persistent disk mounted at `/data`, so history survives redeploys and instance recycling
- **Minority class performance** — classes with very few test examples (Bot, Web Attack Brute Force, Web Attack XSS) show lower precision/recall than majority classes

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| ML Framework | XGBoost, Scikit-learn |
| Backend | Python, Flask, Flask-CORS, Gunicorn |
| Frontend | HTML, CSS, JavaScript |
| Dataset | CIC-IDS2017 (2.83M raw rows; 80,000-row stratified sample used for training) |
| Data Processing | Pandas, NumPy, imbalanced-learn (SMOTE) |

---

## 📁 Project Structure

```
cloud-ids-project/
├── app.py                      ← Flask API + routes
├── Procfile                    ← Render deployment start command
├── render.yaml                 ← Render Blueprint (deployment config)
├── runtime.txt                 ← Python version pin (3.12)
├── requirements.txt            ← Python dependencies (pinned)
├── src/
│   ├── preprocess_data.py      ← Leakage-free cleaning, scaling, SMOTE
│   ├── train_model.py          ← Model training & evaluation
│   ├── extract_samples.py      ← Real attack samples for dashboard simulation
│   ├── response_engine.py      ← Simulated BLOCK/ALERT/ALLOW response layer
│   ├── storage.py              ← SQLite persistence for logs/stats/deny list
│   ├── flow_features.py        ← CIC-style 70-feature extractor (matches the model)
│   ├── flow_builder.py         ← Bidirectional flow accumulation + idle expiry
│   ├── capture_agent.py        ← pcap/live capture agent feeding /predict
│   ├── synth_attacks.py        ← Per-class traffic synthesizer for live demos
│   ├── fix_label_names.py      ← One-time label encoding cleanup (raw CSVs)
│   └── regenerate_report.py    ← Regenerates confusion matrix/report without retraining
├── tests/
│   ├── conftest.py             ← Test DB + app fixtures
│   ├── test_response_engine.py ← Response engine unit tests
│   ├── test_api.py             ← API endpoint tests
│   ├── test_flow_features.py   ← Flow-feature extractor + model compatibility
│   └── test_flow_builder.py    ← Flow accumulation & direction logic
├── demo/
│   └── pcaps/                  ← Generated per-class traffic (gitignored; see synth_attacks.py)
├── templates/
│   ├── dashboard.html          ← Real-time dashboard UI
│   └── attack_samples.json     ← Real dataset rows for simulation
├── models/                     ← Saved model artifacts (generated locally)
├── notebooks/                  ← Charts and visualizations
└── data/
    └── raw/                    ← CIC-IDS2017 CSV files (not committed)
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Hopejohn2004/cloud-ids-project.git
cd cloud-ids-project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download CIC-IDS 2017 dataset
# https://www.unb.ca/cic/datasets/ids-2017.html
# Place CSV files in data/raw/

# 5. Run pipeline
python src/preprocess_data.py
python src/train_model.py
python src/extract_samples.py
#   (optional) fix mojibake in raw Web Attack labels, then re-run the pipeline
python src/fix_label_names.py --dry-run
#   (optional) regenerate report/figure outputs without retraining
python src/regenerate_report.py

# 6. Run the app
python app.py
```

Open `http://localhost:5000`

## 🔴 Live Traffic / pcap Mode

`capture_agent.py` turns real packet traffic into the flow vectors the model
expects and posts them to the API:

```powershell
# Offline replay of a capture file
python src\capture_agent.py --pcap capture.pcap --api http://127.0.0.1:5000 --client my-sensor

# Live sniffing on an interface (Windows: use the interface name/ID from scapy)
python src\capture_agent.py --iface "Ethernet" --api http://127.0.0.1:5000 --client my-sensor

# Classify without posting anywhere (feature extraction debug)
python src\capture_agent.py --pcap capture.pcap --dry-run
```

Environment: `IDS_API` (default `http://127.0.0.1:5000`), `IDS_SENSOR_ID`
(default `IDS-SENSOR`), and `IDS_SENSOR_TOKEN` (default empty) override the
`--api`/`--client`/`--token` flags. Set `--token` whenever the server has
`IDS_SENSOR_TOKEN` configured, otherwise `/predict` returns `401`.

How it works: packets for a bidirectional flow are accumulated until the flow
goes quiet (`--idle`, default 60s), then a 70-feature CIC-style vector is
derived and sent to `/predict` with the real source/destination IPs, so the
response engine can block actual attacker addresses.

Because the trained model lives near a tight benchmark manifold,
`src/synth_attacks.py` searches the flow knob-space to reproduce each attack
class's signature as replayable pcaps:

```powershell
python src\synth_attacks.py --out demo\pcaps --trials 8000
```

This regenerates a `demo/pcaps/` folder with one pcap per class plus a
`manifest.json`. Replaying them through a running server exercises the entire
packet → flow → feature → model → response chain (e.g. DDoS → `BLOCK`,
BENIGN → `ALLOW`).

---

## 🧪 Running Tests

```bash
# From the project root (venv activated)
python -m pytest tests/ -v
```

Tests use a temporary SQLite database so they never touch the real app state. No network access or raw dataset required (the model files are loaded directly for the compatibility checks).

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port (Render sets this automatically) |
| `IDS_DB_PATH` | `<project root>/ids_state.db` | SQLite database file for logs, stats & the blocked-IP deny list |
| `CONFIDENCE_THRESHOLD` | `55.0` | Minimum prediction confidence (%) — below this the verdict becomes `UNCERTAIN` |
| `RATE_LIMIT_PER_MIN` | `60` | Max `/predict` requests per client per minute (0 disables) |
| `IDS_SENSOR_TOKEN` | *(empty)* | Shared secret for live sensors. When set, `/predict` requires the matching `X-Sensor-Token` header (401 otherwise). Sensors pass it via `--token` or the same env var |
| `IDS_RETENTION_DAYS` | `0` | Delete detections/actions older than this many days at startup (0 = keep forever) |

Open `http://localhost:5000`

---

## ☁️ Cloud Deployment (Render)

A Render Blueprint config (`render.yaml`) is included. To deploy:

1. Push this repo to GitHub
2. Go to https://dashboard.render.com/blueprints → **New + → Blueprint**
3. Connect the `cloud-ids-project` repository
4. Render auto-detects `render.yaml` → click **Apply**

The app reads the `$PORT` environment variable and runs under Gunicorn, so no code changes are needed between local and cloud. Model artifacts (`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) are committed so a fresh clone runs without the raw dataset.

`render.yaml` also attaches a **persistent disk** mounted at `/data` with `IDS_DB_PATH=/data/ids_state.db`, so detection history survives redeploys/recycling on Render's free tier instead of being wiped with the ephemeral filesystem.

The dashboard shows a **LIVE CAPTURE · _N_ SENSORS** badge (green) once one or more capture agents have reported, and falls back to an amber **SIMULATION MODE** badge otherwise. The badge and sensor count come from polling `/health` every 5s.

---

## 📚 Dataset

**CIC-IDS 2017** (Canadian Institute for Cybersecurity)
- **Raw size:** 2,830,743 rows × 79 features
- **Used for training:** 80,000-row stratified sample (12 classes, 70 features after cleaning)
- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html

---

## 👤 Author

**Hopejohn2004** — Cloud Intrusion Detection Final Year Project
