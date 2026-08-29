# ⚡ Federated Learning IDS — Smart Energy IoT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flower](https://img.shields.io/badge/Flower-FL-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

**A privacy-preserving Intrusion Detection System for Smart Energy IoT environments
using Federated Learning + CNN + BiLSTM + Attention**

[Features](#-features) • [Architecture](#-model-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Dashboard](#-dashboard)

</div>

---

## 📌 Project Overview

This project implements a **Federated Learning-based Intrusion Detection System (IDS)**
for Smart Energy IoT environments. Instead of centralizing sensitive network traffic data,
each IoT client (Smart Home, EV Charging Station, Grid Sensor, etc.) trains a local model
and shares only **model weights** — never raw data — with a central aggregation server.

### 🎯 Key Highlights

- ✅ **94.65% test accuracy** on combined CICIoT2023 + EdgeIIoTset datasets
- ✅ **Privacy-preserving** — raw data never leaves client devices
- ✅ **Hybrid deep learning** — CNN + BiLSTM + Attention mechanism
- ✅ **5 federated clients** simulating real smart energy environments
- ✅ **Multi-class detection** — DDoS, DoS, Normal, and Other attacks
- ✅ **Interactive dashboard** with 11 pages including live detection and XAI

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🔄 Federated Learning | FedAvg across 5 heterogeneous IoT clients |
| 🧠 Hybrid Model | Parallel CNN + BiLSTM + Attention branches |
| 📊 Multi-class IDS | Detects DDoS, DoS, Normal, Other attack types |
| 🔍 Live Detection | Upload CSV → get real-time attack predictions |
| 🔴 Real-time Monitor | Animated threat level gauge with live logs |
| 🧠 XAI | Attention weight heatmaps and feature importance |
| 🆚 Baseline Comparison | vs CNN only, LSTM only, CNN+BiLSTM |
| 🛡️ Attack Encyclopedia | Visual guide to all attack types |
| 🗺️ Federation Map | Visual FL network topology |
| 📈 Training History | Accuracy/loss curves with LR schedule |

---

## 📁 Project Structure

```
fl_ids_energy/
│
├── data/
│   ├── raw/
│   │   ├── CICIoT2023/          # CICIoT2023 CSV files
│   │   └── EdgeIIoTset/         # EdgeIIoTset CSV file
│   └── processed/               # Preprocessed .npy files
│
├── preprocessing/
│   ├── preprocess.py            # Data cleaning, encoding, scaling
│   ├── windowing.py             # Sliding window sequence generation
│   └── check_columns.py        # Dataset column diagnostic tool
│
├── model/
│   ├── __init__.py
│   └── architecture.py          # CNN + BiLSTM + Attention model
│
├── federated/
│   ├── __init__.py
│   ├── client.py                # Flower NumPyClient implementation
│   └── server.py                # FedAvg server + simulation
│
├── evaluation/
│   └── evaluate.py              # Full metrics + confusion matrix + ROC
│
├── dashboard/
│   └── app.py                   # Streamlit 11-page dashboard
│
├── saved_models/                # Trained model + test data
├── train_direct.py              # Direct training (no FL overhead)
├── requirements.txt
└── README.md
```

---

## 🏗️ Model Architecture

```
Input (20 × 46)
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
 CNN Branch          BiLSTM Branch              │
 Conv1D(64)         BiLSTM(64×2)               │
 BatchNorm          return_seq=True             │
 MaxPool            Dropout(0.3)               │
 Dropout(0.3)       BiLSTM(64×2)               │
 Conv1D(128)        return_seq=True             │
 GlobalAvgPool          │                      │
    │               Attention Layer             │
    │               Dense(1, tanh)              │
    │               Softmax(axis=1)             │
    │               Multiply + Sum              │
    │                     │                     │
    └──────── Concatenate (256) ────────────────┘
                     │
               Dense(256, relu)
               BatchNorm + Dropout
               Dense(128, relu)
               Dropout
                     │
               Softmax(4 classes)
```

### Why This Architecture?

| Component | Purpose |
|---|---|
| **CNN Branch** | Extracts local spatial patterns in traffic features |
| **BiLSTM Branch** | Models temporal dependencies in both directions |
| **Attention Layer** | Focuses on the most suspicious time steps |
| **Parallel merge** | Avoids information bottleneck between branches |

---

## 📦 Datasets

| Dataset | Source | Rows | Features |
|---|---|---|---|
| CICIoT2023 | University of New Brunswick | 1,352,000 | 46 |
| EdgeIIoTset | IEEE DataPort | 69,387 | 46 |

### Download Links

- **CICIoT2023**: https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
- **EdgeIIoTset**: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fl-ids-energy.git
cd fl-ids-energy
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

You can automatically download the datasets programmatically directly from Kaggle:
```bash
python download_datasets.py
```
*Note: The script will check if you are already authenticated with Kaggle. If not, it will prompt you interactively in the terminal for your Kaggle Username and API Key (you can get these from your Kaggle Account profile).*

Alternatively, you can manually place your downloaded dataset files at:
```
data/raw/CICIoT2023/wataiData/csv/CICIoT2023/   ← all .csv files here
data/raw/EdgeIIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv
```

---

## ▶️ Usage

### Step 1 — Preprocess data

```bash
python preprocessing/preprocess.py
```

This cleans, encodes, and normalizes both datasets and saves them to `data/processed/`.

### Step 2 — Generate sequences

```bash
python preprocessing/windowing.py
```

Creates sliding window sequences (window=20, stride=5) from tabular data.

### Step 3 — Train the model

```bash
python -u train_direct.py
```

Trains the CNN + BiLSTM + Attention model directly. Expected time: 30-60 min on CPU.

### Step 4 — Evaluate

```bash
python -u evaluation/evaluate.py
```

Generates full classification report, confusion matrix, and ROC curves.

### Step 5 — Launch Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

---

## 📊 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **94.65%** |
| Macro F1 | 68.40% |
| Macro Precision | 68.97% |
| Macro Recall | 67.90% |
| Best Epoch | 5 / 20 |
| Test Samples | 10,774 |

### Per-class Results

| Class | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|
| DDoS | 81.0% | 74.9% | **77.8%** | ✅ Good |
| DoS | 0.0% | 0.0% | **0.0%** | ⚠️ Too few samples |
| Normal | 100.0% | 100.0% | **100.0%** | 🌟 Perfect |
| Other | 94.9% | 96.8% | **95.8%** | ✅ Excellent |

> **Note:** DoS achieves 0% F1 because it had only 147 training sequences vs 33,360 for Other.
> Increasing `ROWS_PER_CIC_FILE` in `preprocess.py` from 8,000 to 20,000 will significantly
> improve DoS detection.

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, metrics summary, pipeline |
| 🔴 Real-time Monitor | Live threat level gauge + detection log |
| 🔍 Live Detection | Upload CSV → predict attack class + confidence |
| 📊 Evaluation Results | Full metrics, ROC curves, confusion matrix |
| 📈 Training History | Accuracy/loss/LR curves over epochs |
| 🆚 Baseline Comparison | vs CNN, LSTM, CNN+BiLSTM architectures |
| 🧠 Model Explainability | Attention heatmaps + feature importance |
| 🗺️ Federation Map | FL network topology + round-by-round accuracy |
| 🛡️ Attack Encyclopedia | Visual guide to DDoS, DoS, Botnet etc. |
| 🏗️ Architecture | Layer stack + hyperparameter table |
| ℹ️ About | Project details + tech stack |

---

## 🔄 Federated Learning Setup

| Parameter | Value |
|---|---|
| FL Framework | Flower (flwr) |
| Strategy | FedAvg |
| Number of Clients | 5 |
| FL Rounds | 10 |
| Local Epochs | 3 per round |
| Batch Size | 64 |
| Data Distribution | Non-IID (by attack class) |

### Simulated Clients

| Client | Environment | Role |
|---|---|---|
| Client 1 | Smart Home | Residential IoT traffic |
| Client 2 | EV Charging Station | Vehicle charging sessions |
| Client 3 | Grid Sensor | Power grid telemetry |
| Client 4 | Solar/Wind Controller | Renewable energy control |
| Client 5 | Industrial Energy System | SCADA/industrial traffic |

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| TensorFlow | 2.x | Deep learning |
| Flower (flwr) | Latest | Federated learning |
| NumPy | Latest | Array operations |
| Pandas | Latest | Data manipulation |
| Scikit-learn | Latest | Preprocessing + metrics |
| Imbalanced-learn | Latest | SMOTE (optional) |
| Matplotlib | Latest | Visualization |
| Seaborn | Latest | Heatmaps |
| Streamlit | Latest | Dashboard |

---

## 🔮 Future Work

- [ ] Add **Differential Privacy** (Gaussian noise on weights)
- [ ] Implement **Byzantine-Robust Aggregation** (Krum / Median)
- [ ] Add **Blockchain audit ledger** for weight verification
- [ ] Simulate **model poisoning attack** and defense
- [ ] Expand to **11 attack classes** with more data sampling
- [ ] Deploy dashboard to **Streamlit Cloud** or **AWS**
- [ ] Add **MQTT live traffic ingestion** for real IoT devices

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- **CICIoT2023 Dataset** — Canadian Institute for Cybersecurity, University of New Brunswick
- **EdgeIIoTset Dataset** — Dr. Mohamed Amine Ferrag, IEEE DataPort
- **Flower Framework** — flower.dev — Federated Learning made easy
- **TensorFlow Team** — Deep learning framework

---

<div align="center">
  Made with ❤️ as a Final Year Major Project<br>
  <b>Federated Learning IDS — Smart Energy IoT</b>
</div>
