<div align="center">

<img src="https://img.shields.io/badge/AeroSentinel-AI-5dc4ff?style=for-the-badge&logo=airplayvideo&logoColor=white" alt="AeroSentinel AI"/>

# ✈️ AeroSentinel AI
### Aircraft Predictive Maintenance Intelligence Platform

> **Production-grade end-to-end ML + GenAI system for turbofan engine fleet health monitoring, Remaining Useful Life (RUL) prediction, and AI-driven maintenance diagnosis — built on NASA CMAPSS sensor telemetry.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM%20%2B%20Attention-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Mission%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Inference-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-blue)](https://faiss.ai)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1%2070B-F55036?logo=groq&logoColor=white)](https://groq.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Dashboard-Streamlit%20Cloud-5dc4ff?style=for-the-badge)](https://your-username-aerosentinel.streamlit.app)
[![API Docs](https://img.shields.io/badge/📡%20API%20Docs-Swagger%20UI-009688?style=for-the-badge)](https://your-username-aerosentinel.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/⭐%20Star%20this%20Repo-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/your-username/aerosentinel-ai)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [What Was Built](#-what-was-built)
- [Technical Stack](#-technical-stack)
- [Model Performance](#-model-performance)
- [Dashboard Workspaces](#-dashboard-workspaces-10-pages)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)

---

## 🔍 Overview

**AeroSentinel AI** is a full-stack aerospace intelligence platform that ingests raw multi-variate turbofan engine sensor data from the **NASA CMAPSS dataset**, trains a deep learning RUL predictor, and surfaces fleet-wide health intelligence through a 10-workspace mission-control dashboard — augmented by a Retrieval-Augmented Generation (RAG) diagnosis assistant powered by **LLaMA 3.1 70B on Groq**.

The system replicates what a real-world Predictive Health Management (PHM) platform does: continuous sensor ingestion → anomaly detection → failure horizon forecasting → AI-assisted maintenance recommendation.

**Key results on NASA CMAPSS FD001:**
| Metric | Score |
|---|---|
| RMSE (RUL prediction) | **15.4 cycles** |
| MAE | **11.7 cycles** |
| NASA PHM Score | **402.7** |
| Test engines | 100 |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AeroSentinel AI — System Architecture            │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │   NASA CMAPSS Data   │  ← 4 datasets (FD001–FD004)
  │  Raw Sensor Telemetry│    21 sensors × N engines × T cycles
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Preprocessing      │  ← MinMaxScaler, sliding window (30 cycles),
  │   Pipeline           │    RUL capping at 125, feature selection
  └──────────┬───────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────────────────────────────┐
│  Flat        │  │     Sequential (3D) Features          │
│  Features   │  │     shape: (N, 30, 14)                 │
└──────┬──────┘  └────────────────┬─────────────────────┘
       │                          │
       ▼                          ▼
┌────────────────┐    ┌───────────────────────────────────┐
│  XGBoost       │    │   Stacked Bidirectional LSTM       │
│  Random Forest │    │   ┌─────────────────────────────┐ │
│  (Baselines)   │    │   │  Input (14 features)         │ │
└────────────────┘    │   │  → BiLSTM Layer 1 (128 units)│ │
                      │   │  → Dropout (0.3)             │ │
                      │   │  → BiLSTM Layer 2 (64 units) │ │
                      │   │  → Attention Pooling         │ │
                      │   │  → FC(64) → ReLU → FC(1)    │ │
                      │   └─────────────────────────────┘ │
                      │        RUL Prediction (cycles)     │
                      └───────────────────┬───────────────┘
                                          │
             ┌────────────────────────────┼──────────────────────┐
             │                            │                       │
             ▼                            ▼                       ▼
  ┌─────────────────┐       ┌─────────────────────┐   ┌──────────────────┐
  │  FastAPI REST   │       │  GenAI RAG Pipeline  │   │  Streamlit       │
  │  API Server     │       │                      │   │  Dashboard       │
  │                 │       │  Maintenance Docs     │   │  (10 Workspaces) │
  │  /predict/rul   │       │  → FAISS VectorStore  │   │                  │
  │  /diagnose      │       │  → HuggingFace Embed  │   │  Fleet Command   │
  │  /health        │       │  → LangChain RAG      │   │  Digital Twin    │
  │                 │       │  → Groq LLaMA 3.1 70B │   │  Sensor Intel    │
  └─────────────────┘       └─────────────────────┘   │  Risk Center     │
                                                        │  AI Copilot      │
                                                        │  Exec Reports    │
                                                        └──────────────────┘
```

---

## 🔨 What Was Built

### 1. Data Engineering Pipeline
- Parsed and processed **4 NASA CMAPSS datasets** (FD001–FD004) covering single and multi-operating condition scenarios
- Implemented **sliding window sequence extraction** (window size = 30 cycles) for temporal modelling
- Applied **MinMaxScaler** normalization per dataset with persisted scalers for inference
- Applied **RUL capping at 125 cycles** (piece-wise linear degradation assumption) matching PHM literature
- Generated flat feature matrices for ensemble baselines and 3D tensors `(N, 30, 14)` for LSTM

### 2. Deep Learning Model — Bidirectional LSTM with Attention
- Designed a **stacked 2-layer Bidirectional LSTM** that reads sensor sequences both forward and backward
- Implemented a custom **soft attention mechanism** that learns to weight which time steps in the degradation trajectory matter most for RUL estimation
- Trained with **Adam optimizer (lr=1e-3)**, batch size 256, over 60 epochs with 15% validation split
- Achieved **RMSE = 15.4 cycles** and **MAE = 11.7 cycles** on NASA CMAPSS FD001 test set

### 3. ML Ensemble Baselines
- Trained **XGBoost Regressor** and **Random Forest Regressor** on flattened feature windows as comparison benchmarks
- Persisted models as `.json` (XGBoost) and `.pkl` (Random Forest) for fast reload and serving

### 4. GenAI RAG Diagnosis System
- Built a **FAISS vector store** from aerospace maintenance documentation using `sentence-transformers/all-MiniLM-L6-v2` embeddings (runs fully locally — no API cost)
- Designed a **LangChain RetrievalQA chain** that fuses retrieved maintenance knowledge with live LSTM RUL predictions
- Integrated **Groq's free-tier LLaMA 3.1 70B** as the LLM backbone (14,400 requests/day free, no credit card)
- The chatbot receives engine context (RUL, status, action) + user question → returns grounded, source-attributed maintenance diagnosis

### 5. FastAPI REST Inference Server
- Built a production-ready **FastAPI application** with 3 endpoints: `/health`, `/predict/rul`, `/diagnose`
- Implemented **singleton model loading** (lazy init, cached in memory) to avoid reload overhead per request
- Added **CORS middleware** for cross-origin dashboard integration
- Auto-generated **Swagger UI** at `/docs` for testing and portfolio demonstration
- Pydantic v2 request/response models with field validation and descriptions

### 6. Streamlit Mission-Control Dashboard (10 Workspaces)
- Built a **10-page multi-workspace Streamlit application** with 3 switchable dark themes (Aero Command, Nova Control, Orbit Noir)
- **Dashboard** — Real-time system status: LSTM online/offline, VectorStore loaded, LLM ready, mission state
- **Fleet Command** — Per-engine RUL predictions with colour-coded health tiers (Healthy / Caution / Warning / Critical)
- **Digital Twin** — Per-engine degradation timeline simulation with maintenance window forecasting
- **Sensor Intelligence** — Statistical drift index, outlier ratio, sensor stability ranking across all 21 sensors
- **Risk Center** — Fleet-wide risk heatmap, failure probability distribution, maintenance urgency prioritisation
- **RAG Intelligence** — Knowledge base Q&A with FAISS retrieval and source attribution
- **AI Copilot** — Interactive chat interface backed by the LangChain + Groq RAG chain
- **Executive Reports** — Auto-generated fleet summary with CSV / Excel / PDF export suite
- **Operations Center** — Maintenance scheduling, work order tracking, dispatch status
- **Settings** — Theme switcher, model tag selector, system configuration

### 7. Project Infrastructure
- Modular codebase with typed function signatures, dataclasses, and separation of concerns
- `python-dotenv` based secrets management — no hardcoded credentials
- `requirements.txt` pinned dependency versions for reproducible environments
- Evaluation script with RMSE, MAE, and NASA PHM Scoring Function

---

## 🛠 Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| Deep Learning | PyTorch 2.3, BiLSTM + Attention | RUL sequence regression |
| ML Baselines | XGBoost 2.0, Scikit-learn RF | Benchmark comparison |
| Data Processing | Pandas 2.2, NumPy 1.26, SciPy | Feature engineering, preprocessing |
| GenAI / LLM | LangChain 0.2, Groq LLaMA 3.1 70B | RAG diagnosis assistant |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | Local vector embeddings |
| Vector Store | FAISS CPU 1.8 | Local semantic retrieval |
| Dashboard | Streamlit 1.35, Plotly 5.22 | Interactive mission-control UI |
| API Server | FastAPI 0.111, Uvicorn 0.30 | REST inference endpoints |
| Serialisation | Pydantic v2 | Request/response validation |
| Dataset | NASA CMAPSS (FD001–FD004) | Turbofan degradation telemetry |

---

## 📊 Model Performance

```
Dataset  │  RMSE   │   MAE   │  NASA Score  │  Engines
─────────┼─────────┼─────────┼──────────────┼──────────
FD001    │  15.40  │  11.66  │    402.66    │   100
```

> NASA PHM Score penalises late predictions (over-confidence) more heavily than early ones, reflecting the real-world asymmetric cost of missing a failure vs. scheduling unnecessary maintenance.

---

## 🖥 Dashboard Workspaces (10 Pages)

```
✈️  AeroSentinel AI
├── 📡  Dashboard          → System health: LSTM / VectorStore / LLM status
├── 🛩  Fleet Command      → Engine-by-engine RUL table with health tier labels
├── 🔬  Digital Twin       → Per-engine degradation curve + maintenance horizon
├── 📈  Sensor Intelligence → Drift index, outlier ratio, sensor stability ranking
├── ⚠️  Risk Center        → Fleet risk heatmap + failure probability distribution
├── 🧠  RAG Intelligence   → Knowledge base search with FAISS source attribution
├── 🤖  AI Copilot         → Interactive LangChain + Groq maintenance chatbot
├── 📋  Executive Reports  → Fleet summary + CSV / Excel / PDF export
├── 🔧  Operations Center  → Maintenance scheduling + work order tracking
└── ⚙️  Settings           → Theme switcher + model config
```

---

## 📁 Project Structure

```
aerosentinel-ai/
│
├── data/
│   ├── raw/                        # NASA CMAPSS raw text files (FD001–FD004)
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/                  # Scaled numpy arrays + persisted scalers
│       ├── X_train_FD001.npy
│       ├── X_test_FD001.npy
│       ├── y_train_FD001.npy
│       ├── y_test_FD001.npy
│       └── scaler_FD001.pkl
│
├── models/
│   ├── preprocess.py               # Sliding window + RUL capping pipeline
│   ├── train_model.py              # AeroLSTM architecture + XGBoost/RF training
│   ├── evaluate.py                 # RMSE, MAE, NASA PHM scoring
│   ├── saved/
│   │   ├── lstm_FD001.pt           # Trained BiLSTM checkpoint
│   │   ├── xgb_FD001.json          # XGBoost model
│   │   ├── rf_FD001.pkl            # Random Forest model
│   │   └── metrics_FD001.csv       # Evaluation results
│   └── plots/
│       └── evaluation_FD001.png    # Predicted vs actual RUL plot
│
├── genai/
│   ├── build_vectorstore.py        # FAISS index construction from maintenance docs
│   ├── chatbot.py                  # LangChain RAG chain + Groq LLM integration
│   └── vectorstore/
│       ├── index.faiss             # FAISS binary index
│       └── index.pkl               # Document metadata store
│
├── dashboard/
│   └── streamlt_app.py             # 10-workspace Streamlit mission-control app
│
├── deployment/
│   └── api_server.py               # FastAPI REST server (3 endpoints)
│
├── .env                            # API keys (not committed to git)
├── requirements.txt                # Pinned dependency versions
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### 1. Clone & Install
```bash
git clone https://github.com/your-username/aerosentinel-ai.git
cd aerosentinel-ai
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Edit .env and add your key
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Full Pipeline
```bash
# Step 1 — Preprocess data
python models/preprocess.py

# Step 2 — Train models (LSTM + XGBoost + RF)
python models/train_model.py

# Step 3 — Build GenAI vector store
python genai/build_vectorstore.py

# Step 4 — Launch dashboard
streamlit run dashboard/streamlt_app.py

# Step 5 (optional) — Start REST API
uvicorn deployment.api_server:app --reload --port 8000
```

---

## 🚀 Deployment

### Option A — Streamlit Community Cloud (Dashboard) — Free
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set main file: `dashboard/streamlt_app.py`
4. Add secret in Advanced Settings: `GROQ_API_KEY = "your_key"`
5. Deploy → live at `https://your-username-aerosentinel.streamlit.app`

### Option B — Render (FastAPI Backend) — Free
1. Go to [render.com](https://render.com) → New Web Service → connect repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn deployment.api_server:app --host 0.0.0.0 --port $PORT`
4. Add env var: `GROQ_API_KEY`

### Option C — Docker + Railway — Full Stack Free
```bash
# Build and run locally
docker-compose up --build
# Dashboard → http://localhost:8501
# API Docs  → http://localhost:8000/docs
```
Deploy `Dockerfile` to [railway.app](https://railway.app) for a public URL.

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "model": "AeroSense v1.0", "device": "cpu" }
```

### `POST /predict/rul`
```json
// Request
{ "engine_id": 42 }

// Response
{
  "engine_id": 42,
  "predicted_rul": 67.3,
  "status": "Caution",
  "action": "Schedule preventive maintenance within 2 weeks",
  "days_remaining": 33.7
}
```

### `POST /diagnose`
```json
// Request
{ "engine_id": 42, "question": "Which sensors show the most degradation?" }

// Response
{
  "engine_id": 42,
  "predicted_rul": 67.3,
  "status": "Caution",
  "answer": "Based on retrieved maintenance knowledge and RUL=67.3 cycles..."
}
```

Full interactive docs at `/docs` (Swagger UI).

---

## ⚙️ Configuration

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for LLaMA 3.1 70B | ✅ Yes |
| `HF_TOKEN` | HuggingFace token (optional — local embeddings used by default) | ❌ No |

---

## 📌 Built For

This project was built to demonstrate applied data science and AI engineering capabilities relevant to **aerospace predictive maintenance** roles — specifically covering:

- End-to-end ML pipeline design and deployment
- Deep learning (LSTM) for time-series regression on real sensor telemetry
- GenAI integration (RAG, LLM, vector stores) using production-grade frameworks
- Full-stack deployment with REST API and interactive dashboard
- Python ecosystem: PyTorch, Scikit-learn, Pandas, NumPy, LangChain, FastAPI, Streamlit

---

---

<div align="center">

**Built with PyTorch · LangChain · Groq · Streamlit · FastAPI**

⭐ Star this repo if you found it useful

</div>