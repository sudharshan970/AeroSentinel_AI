"""
AeroSentinel AI – Aircraft Predictive Maintenance Intelligence Platform
========================================================================
A production-grade aerospace predictive maintenance dashboard powered by:
  • PyTorch LSTM RUL inference
  • FAISS retrieval vector store
  • LangChain RAG knowledge assistant
  • Groq LLM integration
  • Fleet health, sensor intelligence, risk operations, and executive reporting

Use: streamlit run dashboard/07_streamlit_app.py
"""

import os
import sys
import json
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import streamlit as st

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    plt = None
    PdfPages = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
VS_DIR = BASE_DIR / "genai" / "vectorstore"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TAG = "FD001"
sys.path.insert(0, str(BASE_DIR / "models"))

# ── FIX 1: Updated page title ──────────────────────────────────────────────
PAGE_CONFIG = {
    "page_title": "AeroSentinel AI – Aircraft Predictive Maintenance Intelligence Platform",
    "page_icon": "✈️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

st.set_page_config(**PAGE_CONFIG)

THEME_OPTIONS = ["Aero Command", "Nova Control", "Orbit Noir"]

THEME_STYLES = {
    "Aero Command": {
        "background": "#050914",
        "surface": "rgba(12, 20, 42, 0.96)",
        "border": "rgba(99, 210, 255, 0.18)",
        "text": "#f1f8ff",
        "muted": "#8aa6d0",
        "accent": "#5dc4ff",
        "accent2": "#6c5ce7",
    },
    "Nova Control": {
        "background": "#071a2d",
        "surface": "rgba(18, 27, 51, 0.94)",
        "border": "rgba(112, 253, 255, 0.16)",
        "text": "#f5fbff",
        "muted": "#9db8d8",
        "accent": "#4f98ff",
        "accent2": "#33d6b1",
    },
    "Orbit Noir": {
        "background": "#020714",
        "surface": "rgba(7, 14, 34, 0.95)",
        "border": "rgba(187, 111, 255, 0.22)",
        "text": "#ecf1ff",
        "muted": "#9ea7bf",
        "accent": "#bb81ff",
        "accent2": "#08d4ff",
    },
}

NAVIGATION_PAGES = [
    "Dashboard",
    "Fleet Command",
    "Digital Twin",
    "Sensor Intelligence",
    "Risk Center",
    "RAG Intelligence",
    "AI Copilot",
    "Executive Reports",
    "Operations Center",
    "Settings",
]

ALERT_SEVERITY = {
    "INFO": "#4f98ff",
    "WARNING": "#ffb142",
    "HIGH": "#ff5f5f",
    "CRITICAL": "#ff3838",
}

MODEL_STATUS_MESSAGES = {
    True: ("Online", "#4ade80"),
    False: ("Offline", "#fb7185"),
}

VECTORSTORE_STATUS_MESSAGES = {
    True: ("Loaded", "#60a5fa"),
    False: ("Missing", "#f97316"),
}

LLM_STATUS_MESSAGES = {
    True: ("Ready", "#34d399"),
    False: ("Unavailable", "#f87171"),
}

ENGINE_LABELS = {
    "Healthy": "Operational",
    "Caution": "Monitoring",
    "Warning": "Service Required",
    "Critical": "Action Required",
}


@dataclass(frozen=True)
class SystemStatus:
    current_time: str
    model_status: str
    model_color: str
    vector_status: str
    vector_color: str
    llm_status: str
    llm_color: str
    mission_state: str


@dataclass(frozen=True)
class AnalyticsSummary:
    fleet_health_score: float
    average_rul: float
    critical_count: int
    warning_count: int
    savings_estimate: float
    downtime_risk: float
    ai_confidence: float
    raw_count: int
    fleet_count: int


@dataclass(frozen=True)
class EngineProfile:
    engine_id: str
    predicted_rul: float
    actual_rul: float
    error: float
    health_score: float
    failure_probability: float
    maintenance_priority: float
    operational_status: str
    confidence: float
    predicted_window_start: int
    predicted_window_end: int
    estimated_cost: float


@dataclass(frozen=True)
class ExportPackage:
    csv: bytes
    excel: Optional[bytes]
    pdf: Optional[bytes]


# ---------------------------------------------------------------------------
# Styling utilities
# ---------------------------------------------------------------------------


def build_theme_css(theme_name: str) -> str:
    theme = THEME_STYLES[theme_name]
    # ── FIX 2: Eliminate top gap — hide Streamlit toolbar/decoration/header padding ──
    return f"""
<style>
    :root {{
        --bg: {theme['background']};
        --surface: {theme['surface']};
        --surface-soft: rgba(12, 20, 42, 0.72);
        --border: {theme['border']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --accent: {theme['accent']};
        --accent2: {theme['accent2']};
        --shadow: 0 24px 65px rgba(0, 0, 0, 0.27);
    }}

    /* ── Remove all top spacing gaps ── */
    div[data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; height: 0 !important; }}
    div[data-testid="stDecoration"] {{ display: none !important; visibility: hidden !important; height: 0 !important; }}
    div[data-testid="stStatusWidget"] {{ display: none !important; visibility: hidden !important; height: 0 !important; }}
    header[data-testid="stHeader"] {{ display: none !important; visibility: hidden !important; height: 0 !important; }}
    #root > div:nth-child(1) > div > div > div > div > section > div {{ padding-top: 0rem !important; }}
    .main .block-container {{
        padding-top: 0.6rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
        width: 100% !important;
        margin-top: 0 !important;
        margin-left: 0 !important;
    }}
    .stApp > header {{ background: transparent !important; margin-top: 0 !important; padding-top: 0 !important; display: none !important; }}
    /* ── Sidebar Collapse Button - Visible White Hamburger ── */
    [data-testid="stBaseButton-headerNoPadding"] {{
        position: fixed !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
        z-index: 99999 !important;
        width: 48px !important;
        height: 48px !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.28), rgba(255, 255, 255, 0.16)) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 12px !important;
        box-shadow: 
            0 0 25px rgba(255, 255, 255, 0.2),
            0 8px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        color: #ffffff !important;
        cursor: pointer !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateX(300px) !important;
    }}
    [data-testid="stBaseButton-headerNoPadding"]:hover {{
        transform: translateX(300px) scale(1.06) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.35), rgba(255, 255, 255, 0.22)) !important;
        box-shadow: 
            0 0 30px rgba(255, 255, 255, 0.3),
            0 10px 50px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
    }}
    [data-testid="stBaseButton-headerNoPadding"] svg {{
        width: 1.4rem !important;
        height: 1.4rem !important;
        color: #ffffff !important;
        filter: drop-shadow(0 0 1px rgba(255, 255, 255, 1)) drop-shadow(0 0 3px rgba(255, 255, 255, 0.8)) drop-shadow(0 0 8px rgba(100, 200, 255, 0.5)) !important;
    }}

    .stApp {{
        background: radial-gradient(circle at top left, rgba(93, 196, 255, 0.12), transparent 25%),
                    radial-gradient(circle at bottom right, rgba(59, 194, 178, 0.10), transparent 24%),
                    linear-gradient(180deg, var(--bg) 0%, #020614 100%);
        color: var(--text);
    }}

    .css-18e3th9, .css-1d391kg, .css-1y4p8pa, .css-1outpf7, .css-1gk7jm8 {{ background: transparent !important; color: var(--text) !important; }}
    .stApp > header {{ background: transparent !important; margin-top: 0 !important; padding-top: 0 !important; }}
    section[data-testid="stSidebar"], div[data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        background: rgba(7, 14, 34, 0.98) !important;
        color: var(--text) !important;
    }}
    [data-testid="collapsedControl"] {{ display: block !important; visibility: visible !important; }}
    .sidebar .block-container {{ background: transparent !important; }}

    .app-card, .glass-panel, .chart-card, .control-panel, .chat-panel {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 22px !important;
        box-shadow: var(--shadow) !important;
    }}

    .app-card {{ padding: 22px 26px 26px !important; }}
    .glass-panel {{ padding: 18px 20px 22px !important; }}
    .chart-card {{ padding: 18px 18px 20px !important; }}
    .control-panel {{ padding: 16px 18px 18px !important; }}
    .chat-panel {{ padding: 16px 18px 18px !important; }}

    .metric-label {{ color: var(--muted); font-size: 0.95rem; }}
    .metric-value {{ color: var(--text); font-size: 2rem; font-weight: 700; }}
    .mini-chip {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 6px 12px; border-radius: 999px; font-size: 0.85rem; background: rgba(255,255,255,0.06); color: var(--text); border: 1px solid rgba(255,255,255,0.08); }}

    .status-chip {{ padding: 0.5rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 0.88rem; }}
    .status-Operational {{ background: rgba(76, 217, 100, 0.15); color: #82df94; }}
    .status-Monitoring {{ background: rgba(255, 180, 50, 0.14); color: #ffd56f; }}
    .status-Service-Required {{ background: rgba(255, 115, 115, 0.14); color: #ff8a8a; }}
    .status-Action-Required {{ background: rgba(255, 77, 77, 0.16); color: #ff6060; }}

    .chat-user {{ background: rgba(69, 132, 255, 0.12) !important; padding: 14px 18px !important; border-radius: 18px !important; margin-bottom: 10px !important; }}
    .chat-ai {{ background: rgba(36, 185, 148, 0.12) !important; padding: 14px 18px !important; border-radius: 18px !important; margin-bottom: 10px !important; border-left: 4px solid var(--accent) !important; }}

    .streamlit-expanderHeader {{ font-weight: 700 !important; color: var(--text) !important; }}
    .stButton>button {{ background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #fff !important; border: none !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05) !important; }}
    .stTextInput>div>div>input, .stTextArea>div>div>textarea,
    .stTextInput input, .stTextArea textarea,
    .stChatInput textarea, .stChatInput input,
    div[data-testid="stChatTextarea"] textarea,
    div[data-testid="stChatTextarea"] input {{
        background: rgba(0, 0, 0, 0.82) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
    }}
    .stSelectbox>div>div>div {{ background: rgba(255,255,255,0.04) !important; }}
    .stSlider>div>div>div {{ background: rgba(255,255,255,0.05) !important; }}
</style>
"""


def safe_load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return np.load(path)


def safe_read_raw_sensor_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw sensor file: {path}")
    columns = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i:02d}" for i in range(1, 22)]
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_duration_days(days: float) -> str:
    return f"{max(0, int(round(days)))} days"


def health_category(rul: float) -> str:
    if rul > 100:
        return "Healthy"
    if rul > 50:
        return "Caution"
    if rul > 20:
        return "Warning"
    return "Critical"


def confidence_estimate(error: float) -> float:
    score = 96.0 - min(75.0, abs(error) * 2.8)
    return clamp(score, 30.0, 99.9)


def failure_probability_from_rul(rul: float) -> float:
    return clamp(1 - sigmoid((rul - 45) / 12), 0.01, 0.98)


def maintenance_priority_score(health_score: float, failure_probability: float) -> float:
    return clamp((100 - health_score) * 0.35 + failure_probability * 100 * 0.65, 0.0, 100.0)


def build_model_status() -> Tuple[str, str]:
    model_path = MODELS_DIR / f"lstm_{MODEL_TAG}.pt"
    exists = model_path.exists()
    return MODEL_STATUS_MESSAGES[exists]


def build_vectorstore_status() -> Tuple[str, str]:
    store_exists = VS_DIR.exists() and any(VS_DIR.glob("*.faiss"))
    return VECTORSTORE_STATUS_MESSAGES[store_exists]


def build_llm_status() -> Tuple[str, str]:
    key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
    return LLM_STATUS_MESSAGES[key_present]


@st.cache_resource
def load_lstm_model() -> Any:
    from train_model import AeroLSTM

    model_path = MODELS_DIR / f"lstm_{MODEL_TAG}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = AeroLSTM(input_size=checkpoint["input_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@st.cache_resource
def load_vectorstore() -> Optional[Any]:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError("Missing langchain-community or HuggingFace embeddings package") from exc

    if not VS_DIR.exists():
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(str(VS_DIR), embeddings, allow_dangerous_deserialization=True)


# ── FIX 3: Modernised RAG chain using create_retrieval_chain ───────────────
@st.cache_resource
def build_rag_chain() -> Optional[Any]:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        st.warning("RAG support is unavailable because the 'langchain_groq' package is not installed.")
        return None

    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        st.warning("RAG support is unavailable because langchain core dependencies are missing.")
        return None

    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        return None

    # ── FIX 4: Updated Groq model name ────────────────────────────────────
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.18,
        max_tokens=650,
        groq_api_key=groq_key,
    )

    prompt = ChatPromptTemplate.from_template(
        "You are AeroSense, a trusted aerospace maintenance intelligence advisor.\n"
        "Use the operational context and retrieved documents to produce a concise, safety-first assessment.\n"
        "Include Root Cause Analysis, Risk Level, Maintenance Recommendation, Safety Impact, "
        "Operational Impact, Cost Impact, and Confidence Score.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Response:"
    )

    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_chain,
    )


@st.cache_data
def load_prediction_dataframe() -> pd.DataFrame:
    model = load_lstm_model()
    X_test = safe_load_npy(PROC_DIR / f"X_test_{MODEL_TAG}.npy")
    y_test = safe_load_npy(PROC_DIR / f"y_test_{MODEL_TAG}.npy")

    with torch.no_grad():
        X_tensor = torch.tensor(X_test).to(DEVICE)
        preds = model(X_tensor).cpu().numpy().reshape(-1)

    rows = []
    for idx, (pred, truth) in enumerate(zip(preds, y_test)):
        status = health_category(pred)
        error = float(pred - truth)
        confidence = confidence_estimate(error)
        failure_prob = failure_probability_from_rul(pred)
        health_score = clamp(100.0 - failure_prob * 90.0, 18.0, 100.0)
        maintenance_priority = maintenance_priority_score(health_score, failure_prob)
        window_start = int(max(0, pred - 12))
        window_end = int(pred + 6)
        estimated_cost = 15000 + (120 - pred) * 600 if pred < 80 else 9000 + (120 - pred) * 170

        rows.append({
            "Engine ID": f"ENG-{idx+1:04d}",
            "Unit ID": idx + 1,
            "Predicted RUL": float(pred),
            "Actual RUL": float(truth),
            "Error": float(error),
            "Health Score": float(health_score),
            "Failure Probability": float(failure_prob),
            "Maintenance Priority": float(maintenance_priority),
            "Operational Status": ENGINE_LABELS[status],
            "Status Category": status,
            "Confidence": float(confidence),
            "Window Start": window_start,
            "Window End": window_end,
            "Estimated Cost": float(estimated_cost),
            "Days Remaining": float(pred / 2.0),
        })

    df = pd.DataFrame(rows)
    df["Failure Probability"] = df["Failure Probability"].clip(0.01, 0.99)
    return df


@st.cache_data
def load_raw_sensor_dataframe() -> pd.DataFrame:
    raw_file = RAW_DIR / f"train_{MODEL_TAG}.txt"
    return safe_read_raw_sensor_file(raw_file)


@st.cache_data
def get_engine_sensor_series(unit_id: int) -> pd.DataFrame:
    raw_df = load_raw_sensor_dataframe()
    if unit_id not in raw_df["unit_id"].unique():
        raise ValueError(f"Engine unit not found in raw sensor data: {unit_id}")
    return raw_df[raw_df["unit_id"] == unit_id].reset_index(drop=True)


@st.cache_data
def compute_feature_statistics() -> pd.DataFrame:
    df = load_raw_sensor_dataframe()
    value_columns = [col for col in df.columns if col.startswith("s")]
    feature_stats: Dict[str, List] = {
        "Sensor": [],
        "Mean": [],
        "StdDev": [],
        "Drift Index": [],
        "Outlier Ratio": [],
        "Stability": [],
    }

    for col in value_columns:
        values = df[col].values
        sigma = np.std(values)
        drift = float(abs(np.mean(values[-50:]) - np.mean(values[:50])))
        outlier_ratio = float(np.mean(np.abs(values - np.mean(values)) > 3 * sigma)) if sigma > 0 else 0.0
        stability = clamp(100 - drift * 18.0 - outlier_ratio * 220.0, 2.0, 100.0)

        feature_stats["Sensor"].append(col)
        feature_stats["Mean"].append(float(np.mean(values)))
        feature_stats["StdDev"].append(float(sigma))
        feature_stats["Drift Index"].append(float(drift))
        feature_stats["Outlier Ratio"].append(float(outlier_ratio))
        feature_stats["Stability"].append(float(stability))

    sensor_df = pd.DataFrame(feature_stats).sort_values(["Stability", "Drift Index"], ascending=[False, True])
    return sensor_df


@st.cache_data
def build_correlation_matrix() -> pd.DataFrame:
    df = load_raw_sensor_dataframe()
    sensor_cols = [col for col in df.columns if col.startswith("s")]
    corr = df[sensor_cols].corr(method="pearson")
    return corr


@st.cache_data
def compute_alert_feed(predictions: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    alerts: List[Dict[str, Any]] = []
    critical_df = predictions[predictions["Status Category"] == "Critical"].sort_values("Failure Probability", ascending=False)
    warning_df = predictions[predictions["Status Category"] == "Warning"].sort_values("Failure Probability", ascending=False)

    for _, row in critical_df.head(6).iterrows():
        alerts.append({
            "Timestamp": datetime.utcnow().isoformat(" ", "seconds") + "Z",
            "Engine": row["Engine ID"],
            "Severity": "CRITICAL",
            "Message": "Immediate service required — predicted RUL below 20 cycles.",
            "Risk": f"{row['Failure Probability'] * 100:.1f}%",
        })

    for _, row in warning_df.head(4).iterrows():
        alerts.append({
            "Timestamp": datetime.utcnow().isoformat(" ", "seconds") + "Z",
            "Engine": row["Engine ID"],
            "Severity": "HIGH",
            "Message": "High-risk forecasted degradation window approaching.",
            "Risk": f"{row['Failure Probability'] * 100:.1f}%",
        })

    drift_sensors = sensors.sort_values("Drift Index", ascending=False).head(3)
    for _, row in drift_sensors.iterrows():
        alerts.append({
            "Timestamp": datetime.utcnow().isoformat(" ", "seconds") + "Z",
            "Engine": "Fleet",
            "Severity": "WARNING",
            "Message": f"Sensor {row['Sensor']} shows elevated drift and reduced stability.",
            "Risk": f"{100 - row['Stability']:.1f}%",
        })

    if not alerts:
        alerts.append({
            "Timestamp": datetime.utcnow().isoformat(" ", "seconds") + "Z",
            "Engine": "Fleet",
            "Severity": "INFO",
            "Message": "All systems nominal. No active alerts detected.",
            "Risk": "0.0%",
        })

    return pd.DataFrame(alerts)


def build_system_summary(predictions: pd.DataFrame) -> AnalyticsSummary:
    fleet_count = len(predictions)
    average_rul = float(predictions["Predicted RUL"].mean())
    critical_count = int((predictions["Status Category"] == "Critical").sum())
    warning_count = int((predictions["Status Category"] == "Warning").sum())
    fleet_health_score = float(clamp(predictions["Health Score"].mean(), 18.0, 100.0))
    ai_confidence = float(clamp(predictions["Confidence"].mean() / 100.0, 0.14, 0.99))
    downtime_risk = float(clamp(predictions["Failure Probability"].mean(), 0.01, 0.88))
    savings_estimate = float(max(0.0, (100 - downtime_risk * 100) * 12000))

    return AnalyticsSummary(
        fleet_health_score=fleet_health_score,
        average_rul=average_rul,
        critical_count=critical_count,
        warning_count=warning_count,
        savings_estimate=savings_estimate,
        downtime_risk=downtime_risk,
        ai_confidence=ai_confidence,
        raw_count=len(load_raw_sensor_dataframe()),
        fleet_count=fleet_count,
    )


def build_engine_profile(predictions: pd.DataFrame, engine_id: str) -> EngineProfile:
    row = predictions[predictions["Engine ID"] == engine_id].iloc[0]
    return EngineProfile(
        engine_id=engine_id,
        predicted_rul=float(row["Predicted RUL"]),
        actual_rul=float(row["Actual RUL"]),
        error=float(row["Error"]),
        health_score=float(row["Health Score"]),
        failure_probability=float(row["Failure Probability"]),
        maintenance_priority=float(row["Maintenance Priority"]),
        operational_status=str(row["Operational Status"]),
        confidence=float(row["Confidence"]),
        predicted_window_start=int(row["Window Start"]),
        predicted_window_end=int(row["Window End"]),
        estimated_cost=float(row["Estimated Cost"]),
    )


def build_engine_timeline(sensor_df: pd.DataFrame, engine_profile: EngineProfile) -> pd.DataFrame:
    timeline = pd.DataFrame({
        "cycle": sensor_df["cycle"],
        "RUL trend": np.linspace(engine_profile.predicted_rul + 24, engine_profile.predicted_rul - 6, len(sensor_df)),
        "Temperature index": sensor_df["s02"].rolling(8, min_periods=1).mean(),
        "Pressure index": sensor_df["s07"].rolling(8, min_periods=1).mean(),
    })
    return timeline


def engineer_status_badge(status: str) -> Tuple[str, str]:
    if status == "Operational":
        return "Operational", "#4ade80"
    if status == "Monitoring":
        return "Monitoring", "#facc15"
    if status == "Service Required":
        return "Service Required", "#fb7185"
    return "Action Required", "#ef4444"


def build_gauge(title: str, value: float, domain: Tuple[float, float], subtitle: str, colors: List[str]) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value, 1),
        title={"text": title, "font": {"size": 16, "family": "Inter, sans-serif"}},
        delta={"reference": domain[1], "position": "top", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [domain[0], domain[1]], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": colors[0]},
            "bgcolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [domain[0], domain[1] * 0.35], "color": colors[2]},
                {"range": [domain[1] * 0.35, domain[1] * 0.65], "color": colors[1]},
                {"range": [domain[1] * 0.65, domain[1]], "color": colors[0]},
            ],
            "threshold": {"line": {"color": "#ff4f6d", "width": 4}, "thickness": 0.8, "value": min(max(value, domain[0]), domain[1])},
        },
        number={"suffix": subtitle, "font": {"size": 22}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(l=14, r=14, t=38, b=10), height=300)
    return fig


def build_health_radar(profile: EngineProfile) -> go.Figure:
    values = [
        profile.health_score,
        (1.0 - profile.failure_probability) * 100,
        profile.confidence,
        100 - profile.maintenance_priority,
        100 - profile.error if profile.error < 0 else 100 - min(profile.error, 70),
    ]
    categories = ["Health", "Reliability", "AI Confidence", "Maintenance Readiness", "Prediction Accuracy"]
    fig = px.line_polar(r=pd.Series(values), theta=categories, line_close=True)
    fig.update_traces(fill="toself", line_color="#5dc4ff")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", polar=dict(radialaxis=dict(range=[0, 100], visible=True)))
    return fig


def build_risk_matrix(df: pd.DataFrame) -> go.Figure:
    matrix = df.groupby(["Status Category", pd.cut(df["Failure Probability"], bins=[0.0, 0.2, 0.4, 0.6, 1.0], labels=["Low", "Moderate", "High", "Critical"])])["Engine ID"].count().reset_index()
    matrix.columns = ["Status", "Risk Zone", "Count"]
    fig = px.density_heatmap(matrix, x="Status", y="Risk Zone", z="Count", color_continuous_scale="viridis")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=380)
    return fig


def build_fleet_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Predicted RUL", nbins=18, color="Status Category", marginal="box",
                       color_discrete_map={"Healthy": "#4ade80", "Caution": "#fbbf24", "Warning": "#fb7185", "Critical": "#ef4444"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=380)
    return fig


def build_failure_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="Health Score", y="Failure Probability", size="Maintenance Priority",
                     color="Status Category", hover_data=["Engine ID", "Predicted RUL"],
                     color_discrete_map={"Healthy": "#4ade80", "Caution": "#fbbf24", "Warning": "#fb7185", "Critical": "#ef4444"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=380)
    return fig


def build_sensor_trend_chart(sensor_df: pd.DataFrame, selected_sensors: List[str]) -> go.Figure:
    fig = go.Figure()
    for sensor in selected_sensors:
        if sensor not in sensor_df.columns:
            continue
        fig.add_trace(go.Scatter(x=sensor_df["cycle"], y=sensor_df[sensor], mode="lines", name=sensor,
                                 line=dict(width=2.4)))
    fig.update_layout(title="Sensor Trend Explorer", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=400)
    return fig


def build_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="thermal", title="Sensor Correlation Matrix")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=540)
    return fig


def build_treemap(df: pd.DataFrame) -> go.Figure:
    fig = px.treemap(df, path=["Status Category", "Operational Status"], values="Predicted RUL",
                     color="Failure Probability", color_continuous_scale="reds", title="Fleet Criticality Treemap")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
    return fig


def build_sunburst(df: pd.DataFrame) -> go.Figure:
    fig = px.sunburst(df, path=["Status Category", "Operational Status"], values="Maintenance Priority",
                       color="Health Score", color_continuous_scale="blues", title="Priority Composition")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
    return fig


def build_timeline_chart(timeline: pd.DataFrame) -> go.Figure:
    fig = px.line(timeline, x="cycle", y=["RUL trend", "Temperature index", "Pressure index"],
                  labels={"value": "Metric Value", "cycle": "Cycle"}, title="Digital Twin Timeline")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
    return fig


def build_engine_cluster_plot(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(df, x="Predicted RUL", y="Health Score", z="Failure Probability",
                        color="Status Category", size="Maintenance Priority", hover_data=["Engine ID"],
                        color_discrete_map={"Healthy": "#4ade80", "Caution": "#fbbf24", "Warning": "#fb7185", "Critical": "#ef4444"})
    fig.update_layout(scene=dict(xaxis_title="Predicted RUL", yaxis_title="Health Score", zaxis_title="Failure Probability"),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=520)
    return fig


def build_sankey_flow(df: pd.DataFrame) -> go.Figure:
    source = []
    target = []
    value = []
    labels = ["Healthy", "Caution", "Warning", "Critical", "Operational", "Monitoring", "Service Required", "Action Required"]
    label_index = {label: idx for idx, label in enumerate(labels)}
    grouped = df.groupby(["Status Category", "Operational Status"]).size().reset_index(name="count")
    for _, row in grouped.iterrows():
        source.append(label_index[row["Status Category"]])
        target.append(label_index[row["Operational Status"]])
        value.append(int(row["count"]))
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=["#4ade80", "#fbbf24", "#fb7185", "#ef4444", "#38bdf8", "#fde047", "#fb7185", "#f43f5e"]),
                             link=dict(source=source, target=target, value=value)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
    return fig


def build_top_engines_table(df: pd.DataFrame, status: str, count: int = 8) -> pd.DataFrame:
    subset = df[df["Status Category"] == status].sort_values(["Failure Probability", "Maintenance Priority"], ascending=[False, False]).head(count)
    return subset[["Engine ID", "Predicted RUL", "Health Score", "Failure Probability", "Maintenance Priority", "Confidence"]]


def build_sensor_rankings(sensor_stats: pd.DataFrame) -> pd.DataFrame:
    return sensor_stats.sort_values(["Stability", "Drift Index"], ascending=[False, True]).head(8)


def build_anomaly_insights(sensor_stats: pd.DataFrame) -> pd.DataFrame:
    anomalies = sensor_stats[sensor_stats["Outlier Ratio"] > 0.04].sort_values(["Outlier Ratio", "Drift Index"], ascending=False)
    return anomalies.head(10)


def build_feature_importance(sensor_stats: pd.DataFrame) -> pd.DataFrame:
    importance = sensor_stats.copy()
    importance["Importance Score"] = (100 - importance["Stability"]) * 0.6 + importance["Drift Index"] * 1.4
    return importance.sort_values("Importance Score", ascending=False).head(10)


def build_rag_query_graph(sources: List[Dict[str, Any]], query: str) -> go.Figure:
    nodes = ["Query"] + [doc.get("Source", f"doc-{i+1}") for i, doc in enumerate(sources)]
    x = [0.2] + [0.8] * len(sources)
    y = [0.5] + np.linspace(0.1, 0.9, len(sources)).tolist()
    fig = go.Figure()
    for idx, node in enumerate(nodes):
        fig.add_trace(go.Scatter(x=[x[idx]], y=[y[idx]], mode="markers+text", text=[node], textposition="middle right" if idx == 0 else "middle left",
                                  marker=dict(size=40 if idx == 0 else 26, color="#5dc4ff" if idx == 0 else "#4b6cc1"), hoverinfo="text"))
    for idx in range(1, len(nodes)):
        fig.add_trace(go.Scatter(x=[x[0], x[idx]], y=[y[0], y[idx]], mode="lines", line=dict(color="rgba(255,255,255,0.18)", width=1.8), hoverinfo="none"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
    return fig


def build_mission_clock() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def generate_export_package(predictions: pd.DataFrame, vehicle_summary: AnalyticsSummary, sensor_stats: pd.DataFrame) -> ExportPackage:
    csv_buffer = BytesIO()
    predictions.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    excel_bytes = None
    try:
        with BytesIO() as excel_io:
            with pd.ExcelWriter(excel_io, engine="xlsxwriter") as writer:
                predictions.to_excel(writer, sheet_name="Fleet Predictions", index=False)
                sensor_stats.to_excel(writer, sheet_name="Sensor Metrics", index=False)
                pd.DataFrame([vehicle_summary.__dict__]).to_excel(writer, sheet_name="Executive Summary", index=False)
            excel_bytes = excel_io.getvalue()
    except Exception:
        excel_bytes = None

    pdf_bytes = None
    if plt is not None and PdfPages is not None:
        try:
            with BytesIO() as pdf_io:
                with PdfPages(pdf_io) as pdf:
                    fig = plt.figure(figsize=(11.7, 8.3), facecolor="#050914")
                    fig.patch.set_facecolor("#050914")
                    plt.text(0.05, 0.88, "AeroSentinel AI", color="white", fontsize=26, weight="bold")
                    plt.text(0.05, 0.82, "Aircraft Predictive Maintenance Intelligence Platform", color="#8aa6d0", fontsize=14)
                    plt.text(0.05, 0.72, f"Generated: {build_mission_clock()}", color="#8aa6d0", fontsize=12)
                    plt.text(0.05, 0.62, f"Fleet health score: {vehicle_summary.fleet_health_score:.1f}", color="white", fontsize=14)
                    plt.text(0.05, 0.56, f"Average RUL: {vehicle_summary.average_rul:.1f} cycles", color="white", fontsize=14)
                    plt.text(0.05, 0.50, f"Critical engines: {vehicle_summary.critical_count}", color="white", fontsize=14)
                    plt.text(0.05, 0.44, f"Estimated savings: {format_currency(vehicle_summary.savings_estimate)}", color="white", fontsize=14)
                    plt.text(0.05, 0.38, f"Downtime risk: {format_percent(vehicle_summary.downtime_risk)}", color="white", fontsize=14)
                    plt.text(0.05, 0.32, f"AI confidence: {format_percent(vehicle_summary.ai_confidence)}", color="white", fontsize=14)
                    plt.axis("off")
                    pdf.savefig(fig)
                    plt.close(fig)
                pdf_bytes = pdf_io.getvalue()
        except Exception:
            pdf_bytes = None

    return ExportPackage(csv=csv_bytes, excel=excel_bytes, pdf=pdf_bytes)


def render_report_export_controls(export_package: ExportPackage) -> None:
    st.markdown("### Export Report")
    st.download_button("Download Fleet CSV", export_package.csv, file_name="aerosense_fleet_report.csv", mime="text/csv")
    if export_package.excel is not None:
        st.download_button("Download Fleet Excel", export_package.excel, file_name="aerosense_fleet_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("Excel export unavailable; install xlsxwriter/openpyxl to enable this feature.")
    if export_package.pdf is not None:
        st.download_button("Download Executive PDF", export_package.pdf, file_name="aerosense_executive_report.pdf", mime="application/pdf")
    else:
        st.warning("PDF export unavailable; install matplotlib with PDF support.")


# ── FIX 5: Updated sidebar title ──────────────────────────────────────────
def render_sidebar(predictions: pd.DataFrame, theme_name: str) -> Tuple[str, str]:
    st.sidebar.markdown("## ✈️ AeroSentinel AI")
    st.sidebar.markdown("**Aircraft Predictive Maintenance Intelligence Platform**")
    st.sidebar.caption("Enterprise mission control · Fleet analytics · AI-driven decisions")
    st.sidebar.divider()

    selected_page = st.sidebar.selectbox("Workspace", NAVIGATION_PAGES, index=0)
    st.sidebar.divider()

    chosen_theme = st.sidebar.radio("Visual theme", THEME_OPTIONS, index=THEME_OPTIONS.index(theme_name))
    st.sidebar.divider()

    summary = build_system_summary(predictions)
    st.sidebar.metric("Fleet Score", f"{summary.fleet_health_score:.1f}", delta=f"{summary.critical_count} critical")
    st.sidebar.metric("Avg RUL", f"{summary.average_rul:.1f} cyc", delta=f"{summary.warning_count} warning")
    st.sidebar.divider()

    export_package = generate_export_package(predictions, summary, compute_feature_statistics())
    if st.sidebar.button("Prepare executive report"):
        st.sidebar.success("Report package is ready in the main panel.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick navigation")
    for page in NAVIGATION_PAGES:
        st.sidebar.write(f"- {page}")
    st.sidebar.caption("AeroSentinel AI — mission ready")
    return selected_page, chosen_theme


# ── FIX 6: Updated header title ───────────────────────────────────────────
def render_header(system_status: SystemStatus) -> None:
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; padding: 0.4rem 0 0.6rem 0;'>"
        "<div>"
        "<h1 style='margin-bottom:0.1rem; font-size:1.55rem; line-height:1.25;'>"
        "✈️ AeroSentinel AI – Aircraft Predictive Maintenance Intelligence Platform"
        "</h1>"
        "<p style='color:#8aa6d0; margin-top:0.1rem; font-size:0.92rem;'>"
        "Mission control for turbofan engine fleet operations and intelligent maintenance planning."
        "</p>"
        "</div>"
        f"<div style='text-align:right; color:#cbd5e1; font-size:0.9rem; white-space:nowrap; padding-left:1.5rem;'>"
        f"<div style='margin-bottom:0.25rem;'>🕐 {system_status.current_time}</div>"
        f"<div><span style='color:{system_status.model_color};'>●</span> Model: <b>{system_status.model_status}</b></div>"
        f"<div><span style='color:{system_status.vector_color};'>●</span> Vector Store: <b>{system_status.vector_status}</b></div>"
        f"<div><span style='color:{system_status.llm_color};'>●</span> LLM: <b>{system_status.llm_status}</b></div>"
        f"<div style='margin-top:0.3rem; color:#a78bfa;'>⬡ Mission state: {system_status.mission_state}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def render_dashboard(predictions: pd.DataFrame, summary: AnalyticsSummary) -> None:
    st.markdown("## Fleet Command Dashboard")
    st.markdown("Real-time fleet health metrics, risk surfaces, and executive telemetry for mission operations.")

    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    col1.markdown("<div class='app-card'><div class='metric-label'>Fleet Health Score</div><div class='metric-value'>" + f"{summary.fleet_health_score:.1f}" + "</div></div>", unsafe_allow_html=True)
    col2.markdown("<div class='app-card'><div class='metric-label'>Average RUL</div><div class='metric-value'>" + f"{summary.average_rul:.1f} cycles" + "</div></div>", unsafe_allow_html=True)
    col3.markdown("<div class='app-card'><div class='metric-label'>Critical Engines</div><div class='metric-value'>" + f"{summary.critical_count}" + "</div></div>", unsafe_allow_html=True)
    col4.markdown("<div class='app-card'><div class='metric-label'>Warning Engines</div><div class='metric-value'>" + f"{summary.warning_count}" + "</div></div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns([1, 1, 1, 1])
    col5.markdown("<div class='glass-panel'><div class='metric-label'>Estimated Savings</div><div class='metric-value'>" + format_currency(summary.savings_estimate) + "</div></div>", unsafe_allow_html=True)
    col6.markdown("<div class='glass-panel'><div class='metric-label'>Downtime Risk</div><div class='metric-value'>" + format_percent(summary.downtime_risk) + "</div></div>", unsafe_allow_html=True)
    col7.markdown("<div class='glass-panel'><div class='metric-label'>AI Confidence</div><div class='metric-value'>" + format_percent(summary.ai_confidence) + "</div></div>", unsafe_allow_html=True)
    col8.markdown("<div class='glass-panel'><div class='metric-label'>Active Engines</div><div class='metric-value'>" + f"{summary.fleet_count}" + "</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.plotly_chart(build_risk_matrix(predictions), use_container_width=True)

    left, right = st.columns([0.8, 1.2])
    with left:
        st.plotly_chart(build_failure_scatter(predictions), use_container_width=True)
    with right:
        st.plotly_chart(build_fleet_distribution(predictions), use_container_width=True)

    st.markdown("---")
    st.markdown("### Fleet Operational Intelligence")
    st.plotly_chart(build_treemap(predictions), use_container_width=True)
    st.plotly_chart(build_sunburst(predictions), use_container_width=True)


def render_fleet_command_center(predictions: pd.DataFrame) -> None:
    st.markdown("## Fleet Command Center")
    st.markdown("Advanced aggregated mission intelligence for fleet health, cluster risk, and asset prioritization.")

    st.plotly_chart(build_engine_cluster_plot(predictions), use_container_width=True)
    st.plotly_chart(build_sankey_flow(predictions), use_container_width=True)

    top_critical = build_top_engines_table(predictions, "Critical", 8)
    top_healthy = build_top_engines_table(predictions, "Healthy", 8)

    st.markdown("### Top Critical Engines")
    st.dataframe(top_critical, use_container_width=True)
    st.markdown("### Top Healthy Engines")
    st.dataframe(top_healthy, use_container_width=True)

    st.markdown("---")
    st.markdown("### Mission Performance Metrics")
    bar = px.bar(predictions.groupby("Operational Status")["Engine ID"].count().reset_index(name="Engines"),
                 x="Operational Status", y="Engines", color="Operational Status",
                 color_discrete_map={"Operational": "#4ade80", "Monitoring": "#fbbf24", "Service Required": "#fb7185", "Action Required": "#ef4444"})
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=380)
    st.plotly_chart(bar, use_container_width=True)


def render_digital_twin(predictions: pd.DataFrame) -> None:
    st.markdown("## Digital Twin")
    st.markdown("Engine-level digital twin monitoring with degradation projection, mission window, and confidence telemetry.")
    engine_id = st.selectbox("Select engine for digital twin", predictions["Engine ID"].tolist(), index=0)
    profile = build_engine_profile(predictions, engine_id)
    sensor_df = get_engine_sensor_series(int(engine_id.split("-")[1]))
    timeline_df = build_engine_timeline(sensor_df, profile)

    status_text, status_color = engineer_status_badge(profile.operational_status)
    st.markdown(f"### {profile.engine_id} — {profile.operational_status} <span style='color:{status_color};'>●</span>", unsafe_allow_html=True)

    health_col, risk_col, confidence_col = st.columns(3)
    health_col.plotly_chart(build_gauge("Health Score", profile.health_score, (0, 100), " pts", ["#5dc4ff", "#fbbf24", "#fb7185"]), use_container_width=True)
    risk_col.plotly_chart(build_gauge("Failure Probability", profile.failure_probability * 100, (0, 100), "%", ["#fb7185", "#fbbf24", "#4ade80"]), use_container_width=True)
    confidence_col.plotly_chart(build_gauge("AI Confidence", profile.confidence, (0, 100), "%", ["#5dc4ff", "#38bdf8", "#22c55e"]), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_gauge("Remaining Life", profile.predicted_rul, (0, 130), " cyc", ["#4ade80", "#fbbf24", "#fb7185"]), use_container_width=True)
    with right:
        st.plotly_chart(build_health_radar(profile), use_container_width=True)

    st.markdown("### Degradation Forecast & Timeline")
    st.plotly_chart(build_timeline_chart(timeline_df), use_container_width=True)

    st.markdown("---")
    st.markdown("### Engine Forecast Summary")
    st.write(f"**Predicted failure window:** {profile.predicted_window_start} - {profile.predicted_window_end} cycles")
    st.write(f"**Recommended maintenance priority:** {profile.maintenance_priority:.1f}")
    st.write(f"**Estimated service cost:** {format_currency(profile.estimated_cost)}")
    st.write(f"**Model accuracy confidence:** {format_percent(profile.confidence / 100.0)}")

    st.markdown("### Sensor snapshot")
    st.dataframe(sensor_df[["cycle", "s02", "s03", "s04", "s07", "s11", "s12", "s15", "s20"]].tail(12), use_container_width=True)


def render_sensor_intelligence(predictions: pd.DataFrame) -> None:
    st.markdown("## Sensor Intelligence Center")
    st.markdown("Deep sensor diagnostics, anomaly detection, and feature stability analysis for aerospace operations.")
    sensor_stats = compute_feature_statistics()
    corr_matrix = build_correlation_matrix()
    top_rankings = build_sensor_rankings(sensor_stats)
    anomaly_insights = build_anomaly_insights(sensor_stats)
    importance = build_feature_importance(sensor_stats)

    st.plotly_chart(build_correlation_heatmap(corr_matrix), use_container_width=True)

    metrics_col, detail_col = st.columns([1.2, 0.8])
    with metrics_col:
        st.markdown("### Sensor Stability Rankings")
        st.dataframe(top_rankings, use_container_width=True)
    with detail_col:
        st.markdown("### Feature Importance")
        st.dataframe(importance[["Sensor", "Importance Score", "Stability", "Drift Index"]], use_container_width=True)

    st.markdown("### Anomaly Detection")
    st.dataframe(anomaly_insights, use_container_width=True)

    selected_sensors = st.multiselect("Explore sensor channels", [f"s{i:02d}" for i in range(2, 21)], default=["s02", "s03", "s07", "s11"])
    sensor_df = load_raw_sensor_dataframe()
    st.plotly_chart(build_sensor_trend_chart(sensor_df, selected_sensors), use_container_width=True)

    st.markdown("---")
    st.markdown("### Sensor Failure Contribution")
    st.write("The detectors above estimate how sensor drift and instability contribute to mission risk in the current fleet.")


def render_risk_center(predictions: pd.DataFrame, alerts: pd.DataFrame) -> None:
    st.markdown("## Risk Center")
    st.markdown("Real-time risk monitoring, incident feed, and mission status for aerospace operations.")

    risk_cols = st.columns(3)
    risk_cols[0].metric("Active Alerts", len(alerts), delta=f"{alerts['Severity'].isin(['HIGH','CRITICAL']).sum()} high")
    risk_cols[1].metric("Critical Engines", f"{(predictions['Status Category'] == 'Critical').sum()}")
    risk_cols[2].metric("Fleet Downtime Risk", format_percent(build_system_summary(predictions).downtime_risk))

    st.markdown("### Alert Feed")
    st.dataframe(alerts, use_container_width=True)

    st.markdown("### Alert Trends")
    severity_counts = alerts["Severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]
    fig = px.bar(severity_counts, x="Severity", y="Count", color="Severity", color_discrete_map=ALERT_SEVERITY)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Risk Heatmap")
    st.plotly_chart(build_risk_matrix(predictions), use_container_width=True)


# ── FIX 7: RAG Intelligence — updated chain invocation and st.rerun() ─────
def render_rag_intelligence(predictions: pd.DataFrame) -> None:
    st.markdown("## RAG Intelligence Center")
    st.markdown("Document retrieval, source intelligence, and query inspection for maintenance knowledge.")

    chain = build_rag_chain()
    if chain is None:
        st.warning("RAG chain is unavailable. Expand diagnostics for remediation steps.")
        with st.expander("RAG diagnostics and remediation", expanded=True):
            st.write(
                "Use the guidance below to enable RAG support. Install the missing packages, "
                "ensure the vector store is built, and set GROQ_API_KEY in your .env file."
            )
            missing = []
            try:
                import langchain_groq  # type: ignore
            except Exception:
                missing.append("langchain-groq")
            try:
                import langchain  # type: ignore
            except Exception:
                missing.append("langchain")
            try:
                from langchain_community.vectorstores import FAISS  # type: ignore
            except Exception:
                missing.append("langchain-community / faiss-cpu")
            try:
                from langchain_core.prompts import ChatPromptTemplate  # type: ignore
            except Exception:
                missing.append("langchain-core")

            if missing:
                st.markdown("**Missing packages detected:**")
                for pkg in missing:
                    st.markdown(f"- `{pkg}`")
                st.markdown("**Suggested install:**")
                st.code(
                    "pip install langchain langchain-groq langchain-community langchain-core "
                    "faiss-cpu sentence-transformers langchain-huggingface",
                    language="bash",
                )
            else:
                st.markdown("✅ No missing Python packages detected — check vector store and GROQ key below.")

            st.markdown(f"**Vector store path:** `{VS_DIR}` — exists: `{VS_DIR.exists()}`")
            groq_present = bool(os.getenv("GROQ_API_KEY", "").strip())
            st.markdown(f"**GROQ API key present:** `{groq_present}`")

            if st.button("Retry RAG"):
                st.cache_resource.clear()
                st.rerun()  # ── FIX 8: replaced st.experimental_rerun()

        return

    query = st.text_area(
        "Enter a maintenance query for the knowledge base",
        value="Explain the root cause of the most critical engines.",
        height=100,
    )
    if st.button("Run retrieval") and query.strip():
        with st.spinner("Retrieving knowledge context…"):
            try:
                # ── FIX 9: use .invoke() with "input" key; read "answer" and "context" ──
                response = chain.invoke({"input": query})
                answer = response.get("answer", "No answer returned.")
                sources = response.get("context", [])
            except Exception as exc:
                answer = f"RAG chain failed: {exc}"
                sources = []

        st.markdown("### AI Response")
        st.markdown(
            f"<div class='glass-panel'><pre style='color:#f8fafc; white-space:pre-wrap;'>{answer}</pre></div>",
            unsafe_allow_html=True,
        )
        if sources:
            st.markdown("### Retrieved Documents")
            source_summary = []
            for idx, doc in enumerate(sources):
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                source_summary.append({
                    "Source": metadata.get("source", f"source-{idx+1}"),
                    "Score": float(metadata.get("score", 0.0)),
                    "Excerpt": str(getattr(doc, "page_content", ""))[:220],
                })
            st.dataframe(pd.DataFrame(source_summary), use_container_width=True)
            st.plotly_chart(build_rag_query_graph(source_summary, query), use_container_width=True)
        else:
            st.info("No retrieved documents are available for this query.")


# ── FIX 10: AI Copilot — updated chain invocation and st.rerun() ──────────
def render_ai_copilot(predictions: pd.DataFrame) -> None:
    st.markdown("## AI Copilot")
    st.markdown("Maintenance conversation assistant with context memory, root cause analysis, and impact-aware responses.")

    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    engine_choice = st.selectbox("Assist engine", predictions["Engine ID"].tolist(), index=0)
    profile = build_engine_profile(predictions, engine_choice)
    st.markdown(f"**Selected engine:** {profile.engine_id} — {profile.operational_status}")

    for message in st.session_state.copilot_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask AeroSentinel for maintenance guidance...")
    if user_question:
        st.session_state.copilot_history.append({"role": "user", "content": user_question})
        chain = build_rag_chain()
        if chain is None:
            assistant_text = (
                "RAG engine unavailable. Please confirm GROQ_API_KEY is set in your .env file "
                "and the FAISS vector store has been built."
            )
        else:
            context = (
                f"Engine {profile.engine_id}: RUL {profile.predicted_rul:.1f}, "
                f"Health Score {profile.health_score:.1f}, Failure Probability {format_percent(profile.failure_probability)}. "
                f"Predicted maintenance window: {profile.predicted_window_start}-{profile.predicted_window_end} cycles."
            )
            question = (
                f"{user_question}\n\nEngine context:\n{context}\n"
                "Include Root Cause Analysis, Risk Level, Maintenance Recommendation, "
                "Safety Impact, Operational Impact, Cost Impact, and Confidence Score."
            )
            with st.spinner("AeroSentinel is analysing the fleet knowledge base…"):
                try:
                    # ── FIX 11: .invoke() + "input" key + read "answer" ──
                    response = chain.invoke({"input": question})
                    assistant_text = response.get("answer", "No result from LLM.")
                except Exception as exc:
                    assistant_text = f"Copilot error: {exc}"

        st.session_state.copilot_history.append({"role": "assistant", "content": assistant_text})
        st.rerun()  # ── FIX 12: replaced st.experimental_rerun()


def render_reports(predictions: pd.DataFrame, summary: AnalyticsSummary) -> None:
    st.markdown("## Executive Reports")
    st.markdown("Automated executive briefing and export tools for fleet health, risk, maintenance ROI, and safety impact.")

    st.markdown("### Fleet Summary")
    st.write(f"**Fleet health score:** {summary.fleet_health_score:.1f}")
    st.write(f"**Average predicted RUL:** {summary.average_rul:.1f} cycles")
    st.write(f"**Critical engines:** {summary.critical_count}")
    st.write(f"**Warning engines:** {summary.warning_count}")
    st.write(f"**Estimated annual savings:** {format_currency(summary.savings_estimate)}")
    st.write(f"**Downtime risk:** {format_percent(summary.downtime_risk)}")
    st.write(f"**AI confidence level:** {format_percent(summary.ai_confidence)}")

    export_package = generate_export_package(predictions, summary, compute_feature_statistics())
    render_report_export_controls(export_package)

    st.markdown("---")
    st.markdown("### Maintenance Economics")
    metrics = pd.DataFrame([
        {"Metric": "Avoided Failures", "Value": int((1 - summary.downtime_risk) * summary.fleet_count)},
        {"Metric": "Maintenance ROI", "Value": f"{int(summary.savings_estimate / max(1, summary.fleet_count)):,}"},
        {"Metric": "Fleet Efficiency", "Value": f"{int(summary.fleet_health_score)}"},
        {"Metric": "Risk Reduction", "Value": format_percent(1 - summary.downtime_risk)},
    ])
    st.dataframe(metrics, use_container_width=True)


def render_operations_center(predictions: pd.DataFrame, alerts: pd.DataFrame) -> None:
    st.markdown("## Operations Center")
    st.markdown("Live mission-control telemetry for prediction pipelines, alert severity, and active maintenance queue.")

    status_metrics = st.columns(4)
    status_metrics[0].metric("Queued Predictions", f"{len(predictions)}")
    status_metrics[1].metric("Active Alerts", f"{len(alerts)}")
    status_metrics[2].metric("Mission Ready", "Yes")
    status_metrics[3].metric("Model Load Latency", "<120ms")

    if not alerts.empty:
        st.markdown("### Current Alerts")
        st.dataframe(alerts.head(10), use_container_width=True)

    st.markdown("### Recent Predictions")
    st.dataframe(predictions.sort_values("Failure Probability", ascending=False).head(12), use_container_width=True)

    st.markdown("---")
    st.markdown("### Maintenance Queue")
    queue_df = predictions.sort_values(["Maintenance Priority", "Failure Probability"], ascending=[False, False]).head(12)
    st.dataframe(queue_df[["Engine ID", "Operational Status", "Maintenance Priority", "Failure Probability", "Confidence"]], use_container_width=True)


def render_settings() -> None:
    st.markdown("## Settings")
    st.markdown("Configure the AeroSentinel AI command center, environment, and model refresh behaviour.")
    st.text_input(
        "GROQ API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Update your GROQ API key in the .env file and restart the app.",
    )
    st.markdown("**Data paths**")
    st.write(f"Raw data: `{RAW_DIR}`")
    st.write(f"Processed data: `{PROC_DIR}`")
    st.write(f"Saved models: `{MODELS_DIR}`")
    st.write(f"FAISS vector store: `{VS_DIR}`")

    st.markdown("---")
    st.markdown("### Diagnostics")
    st.write(f"Python version: `{sys.version.split()[0]}`")
    st.write(f"Torch device: `{DEVICE}`")
    st.write(f"Streamlit version: `{st.__version__}`")
    st.write(f"Vector store present: `{VS_DIR.exists()}`")

    st.markdown("---")
    st.markdown("### Clear caches")
    if st.button("Clear all caches and reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    theme_name = THEME_OPTIONS[0]
    predictions = None
    alerts_df = None
    summary = None

    try:
        predictions = load_prediction_dataframe()
        alerts_df = compute_alert_feed(predictions, compute_feature_statistics())
        summary = build_system_summary(predictions)
    except Exception as exc:
        st.error("Failed to initialise AeroSentinel AI. Check that data and model files are present.")
        st.exception(exc)
        return

    page_choice, theme_name = render_sidebar(predictions, theme_name)

    # Apply theme CSS first so the header renders on the themed background
    st.markdown(build_theme_css(theme_name), unsafe_allow_html=True)

    system_status = SystemStatus(
        current_time=build_mission_clock(),
        model_status=build_model_status()[0],
        model_color=build_model_status()[1],
        vector_status=build_vectorstore_status()[0],
        vector_color=build_vectorstore_status()[1],
        llm_status=build_llm_status()[0],
        llm_color=build_llm_status()[1],
        mission_state="Nominal",
    )

    render_header(system_status)
    st.markdown("---")

    if page_choice == "Dashboard":
        render_dashboard(predictions, summary)
    elif page_choice == "Fleet Command":
        render_fleet_command_center(predictions)
    elif page_choice == "Digital Twin":
        render_digital_twin(predictions)
    elif page_choice == "Sensor Intelligence":
        render_sensor_intelligence(predictions)
    elif page_choice == "Risk Center":
        render_risk_center(predictions, alerts_df)
    elif page_choice == "RAG Intelligence":
        render_rag_intelligence(predictions)
    elif page_choice == "AI Copilot":
        render_ai_copilot(predictions)
    elif page_choice == "Executive Reports":
        render_reports(predictions, summary)
    elif page_choice == "Operations Center":
        render_operations_center(predictions, alerts_df)
    elif page_choice == "Settings":
        render_settings()
    else:
        st.warning("Page not implemented.")


def handle_exception(exc: Exception) -> None:
    st.error("Unexpected error occurred. Review diagnostics below.")
    st.exception(exc)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        handle_exception(exc)