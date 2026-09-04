# 🛡️ ML-Based Cloud Intrusion Detection & Threat Response System

A machine learning-powered network intrusion detection system built as a Final Year Project. Trained on a stratified 80,000-record sample drawn from the CIC-IDS2017 dataset (2,830,743 raw records), achieving **99.79% weighted accuracy** using XGBoost.

---

## 🎯 Features

- **99.79% Accuracy** — XGBoost model, evaluated on an untouched 16,000-record test set
- **Simulated Real-Time Detection** — Flask API classifies submitted traffic samples instantly
- **12 Attack Classes** — DDoS, DoS variants, PortScan, Bot, Brute Force, and more
- **Recommended Response** — Severity-based ALERT/BLOCK/ALLOW recommendation (advisory only — no automated blocking is executed; see Limitations)
- **Live Dashboard** — Real-time detection feed, attack distribution chart, and model metrics pulled live from the backend
- **REST API** — `/predict`, `/health`, `/logs`, `/stats`, `/model-info`, `/distribution` endpoints

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

- **Not deployed to a live cloud environment** at time of writing — tested locally only
- **Simulation-based, not live traffic** — the dashboard classifies sample feature vectors, not packets captured from a real network in real time
- **No automated blocking** — the system recommends a response (`recommended_action`) and explicitly reports `response_executed: false`; no traffic is actually blocked
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
├── requirements.txt            ← Python dependencies
├── src/
│   ├── preprocess_data.py      ← Leakage-free cleaning, scaling, SMOTE
│   ├── train_model.py          ← Model training & evaluation
│   ├── extract_samples.py      ← Real attack samples for dashboard simulation
│   ├── fix_label_names.py      ← One-time label encoding cleanup
│   └── regenerate_report.py    ← Regenerates confusion matrix/report
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

# 6. Run the app
python app.py
```

Open `http://localhost:5000`

---

## 📚 Dataset

**CIC-IDS 2017** (Canadian Institute for Cybersecurity)
- **Raw size:** 2,830,743 rows × 79 features
- **Used for training:** 80,000-row stratified sample (12 classes, 70 features after cleaning)
- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html

---

## 👤 Author

**Hopejohn2004** — Cloud Intrusion Detection Final Year Project
