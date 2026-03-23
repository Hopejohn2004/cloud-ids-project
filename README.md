# 🛡️ ML-Based Cloud Intrusion Detection & Threat Response System

A fully functional, real-time **Machine Learning-powered Network Intrusion Detection System** built from scratch as a Final Year Project. Trained on 2.8 million real network traffic samples from the CIC-IDS 2017 dataset, achieving **99.97% accuracy** using XGBoost.

---

## 🎯 Features

- **99.97% Accuracy** — XGBoost model trained on 2.8M rows
- **Real-time Detection** — Flask API processes traffic instantly
- **12 Attack Types** — DDoS, DoS, PortScan, Bot, Brute Force, and more
- **Automated Response** — BLOCK, ALERT, or ALLOW based on severity
- **Beautiful Dashboard** — Clean white UI with live charts and feed
- **REST API** — `/predict`, `/health`, `/logs`, `/stats` endpoints

---

## 🔍 Detected Attack Types

| Attack Type | Severity | Action |
|-------------|----------|--------|
| DDoS | HIGH | BLOCK |
| DoS Hulk | HIGH | BLOCK |
| DoS GoldenEye | HIGH | BLOCK |
| DoS slowloris | HIGH | BLOCK |
| DoS Slowhttptest | HIGH | BLOCK |
| Bot | HIGH | BLOCK |
| Web Attack – Brute Force | HIGH | BLOCK |
| Heartbleed | CRITICAL | BLOCK |
| Infiltration | CRITICAL | BLOCK |
| PortScan | MEDIUM | ALERT |
| FTP-Patator | MEDIUM | ALERT |
| SSH-Patator | MEDIUM | ALERT |
| BENIGN | NONE | ALLOW |

---

## 🧠 ML Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Decision Tree | 99.92% | 99.92% |
| Random Forest | 99.80% | 99.80% |
| **XGBoost ⭐** | **99.97%** | **99.97%** |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | XGBoost, Scikit-learn |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Dataset | CIC-IDS 2017 (2.8M rows) |
| Data Processing | Pandas, NumPy, SMOTE |

---

## 🗂️ Project Structure

```
cloud-ids-project/
├── app.py                        ← Flask API + routes
├── src/
│   ├── explore_data.py           ← Dataset exploration
│   ├── preprocess_data.py        ← Data cleaning & SMOTE balancing
│   ├── train_model.py            ← Model training & evaluation
│   └── extract_samples.py        ← Extract real attack samples
├── templates/
│   ├── dashboard.html            ← Real-time dashboard UI
│   └── attack_samples.json       ← Real dataset rows for simulation
├── models/                       ← Saved model files (generated locally)
├── notebooks/                    ← Charts and visualizations
└── data/                         ← CIC-IDS 2017 CSV files (not committed)
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
pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask flask-cors joblib imbalanced-learn

# 4. Download CIC-IDS 2017 dataset
# https://www.unb.ca/cic/datasets/ids-2017.html
# Place CSV files in data/ folder

# 5. Run pipeline
python src/preprocess_data.py
python src/train_model.py
python src/extract_samples.py

# 6. Run the app
python app.py
```

Open `http://localhost:5000`

---

## 📊 Dataset

- **Name:** CIC-IDS 2017 (Canadian Institute for Cybersecurity)
- **Size:** 2,830,743 rows × 79 features
- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html

---

## 👤 Author

**Hopejohn2004** — Cloud Engineering Final Year Project
