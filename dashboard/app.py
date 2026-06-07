"""
STEP 7 — Advanced Interactive Streamlit Dashboard
================================================
A corporate-grade dashboard for AeroSense featuring:
  • Fleet health analytics and engine risk scoring
  • Deep-dive sensor trend visualizations
  • Actionable maintenance recommendations
  • GenAI assistant with engine-context awareness

Usage: streamlit run dashboard/07_streamlit_app_advanced.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
VS_DIR = BASE_DIR / "genai" / "vectorstore"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, str(BASE_DIR / "models"))

PAGE_CONFIG = {
    "page_title": "AeroSense — Engine Health Monitor",
    "page_icon": "✈️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

st.set_page_config(**PAGE_CONFIG)

CUSTOM_CSS = """
<style>
  .card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 18px;
    background: linear-gradient(180deg, rgba(16,24,40,0.95), rgba(12,18,30,0.95));
    box-shadow: 0 10px 30px rgba(0,0,0,0.14);
  }
  .metric-title { color: #a1a8c0; font-size: 0.95rem; }
  .metric-value { color: #ffffff; font-size: 2rem; font-weight: 700; }
  .status-pill { padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
  .status-Healthy  { background: rgba(69,193,122,0.16); color: #45c17a; }
  .status-Caution  { background: rgba(247,149,59,0.16); color: #f7953b; }
  .status-Warning  { background: rgba(255,112,67,0.16); color: #ff7043; }
  .status-Critical { background: rgba(239,89,89,0.16); color: #ef5959; }
  .chat-user { background: #1e2432; border-radius: 12px; padding: 14px; margin: 10px 0; }
  .chat-ai   { background: #121827; border-radius: 12px; padding: 14px; margin: 10px 0; border-left: 4px solid #5b8def; }
  .sidebar .stButton>button { width: 100%; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_lstm_model():
    from train_model import AeroLSTM
    tag = "FD001"
    ckpt = torch.load(MODELS_DIR / f"lstm_{tag}.pt", map_location=DEVICE)
    model = AeroLSTM(input_size=ckpt["input_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@st.cache_resource
def load_rag_chain():
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(str(VS_DIR), embeddings, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0.2,
        max_tokens=512,
        groq_api_key=groq_key,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are AeroSense, a trusted jet engine maintenance AI advisor.\n"
            "Use the maintenance context and provide clear, actionable answers.\n\n"
            "MAINTENANCE KNOWLEDGE:\n{context}\n\n"
            "QUESTION: {question}\nANSWER:"
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
    )

@st.cache_data
def load_all_predictions():
    model = load_lstm_model()
    X_test = np.load(PROC_DIR / "X_test_FD001.npy")
    y_test = np.load(PROC_DIR / "y_test_FD001.npy")

    with torch.no_grad():
        X_t = torch.tensor(X_test).to(DEVICE)
        preds = model(X_t).cpu().numpy().clip(0)

    def status(rul):
        if rul > 100:
            return "Healthy", "#45c17a"
        if rul > 50:
            return "Caution", "#f7953b"
        if rul > 20:
            return "Warning", "#ff7043"
        return "Critical", "#ef5959"

    rows = []
    for idx, (pred, true) in enumerate(zip(preds, y_test)):
        label, color = status(pred)
        rows.append({
            "Engine ID": f"ENG-{idx+1:04d}",
            "Predicted RUL": int(round(pred)),
            "Actual RUL": int(round(true)),
            "Error": int(round(pred - true)),
            "Status": label,
            "Color": color,
            "Days Remaining": int(round(pred / 2)),
        })

    return pd.DataFrame(rows)

@st.cache_data
def load_sensor_trends(engine_idx: int):
    import pandas as _pd
    raw_file = BASE_DIR / "data" / "raw" / "train_FD001.txt"
    cols = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i:02d}" for i in range(1, 22)]
    df = _pd.read_csv(raw_file, sep=r"\s+", header=None, names=cols)
    engine_ids = sorted(df["unit_id"].unique())
    engine_idx = min(engine_idx, len(engine_ids) - 1)
    selected_engine = engine_ids[engine_idx]
    return df[df["unit_id"] == selected_engine].reset_index(drop=True)

def build_status_card(title: str, value: str, subtitle: str, color: str):
    return f"""
<div class='card'>
  <div class='metric-title'>{title}</div>
  <div class='metric-value'>{value}</div>
  <div style='color:{color}; margin-top:10px;'>{subtitle}</div>
</div>
"""

def render_sidebar(data_loaded: bool):
    st.sidebar.markdown("## AeroSense Engine Insights")
    st.sidebar.write(
        "Predictive maintenance analytics for turbofan fleets using LSTM scoring, \n"
        "rich interactive fault visualization, and integrated maintenance guidance."
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Deployment readiness**")
    st.sidebar.info(
        "Runs locally with Streamlit and supports GPU inference if available. "
        "Add `GROQ_API_KEY` to `.env` for AI chat capabilities."
    )
    if data_loaded:
        if st.sidebar.button("Export predictions as CSV"):
            df = load_all_predictions()
            st.sidebar.download_button(
                label="Download fleet predictions",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="aerosense_fleet_predictions.csv",
                mime="text/csv",
            )
    st.sidebar.divider()
    st.sidebar.markdown("### Resources")
    st.sidebar.markdown("- `data/raw/` for source datasets\n- `models/saved/` for model snapshots\n- `genai/vectorstore/` for RAG index")

def render_fleet_overview(df: pd.DataFrame):
    st.subheader("Fleet Health Summary")
    counts = df["Status"].value_counts().reindex(["Healthy", "Caution", "Warning", "Critical"], fill_value=0)
    total = len(df)
    score = int((counts["Healthy"] + counts["Caution"] * 0.6 + counts["Warning"] * 0.3) / total * 100)

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    col1.markdown(build_status_card("Fleet Stability Index", f"{score}%", "Higher is better", "#5b8def"), unsafe_allow_html=True)
    col2.markdown(build_status_card("Healthy Engines", f"{counts['Healthy']}", "Ready for mission", "#45c17a"), unsafe_allow_html=True)
    col3.markdown(build_status_card("Warning Engines", f"{counts['Warning']}", "Needs inspection", "#ff7043"), unsafe_allow_html=True)
    col4.markdown(build_status_card("Critical Engines", f"{counts['Critical']}", "Immediate action", "#ef5959"), unsafe_allow_html=True)

    fig = px.scatter(
        df,
        x="Actual RUL",
        y="Predicted RUL",
        color="Status",
        color_discrete_map={
            "Healthy": "#45c17a",
            "Caution": "#f7953b",
            "Warning": "#ff7043",
            "Critical": "#ef5959",
        },
        hover_data=["Engine ID", "Days Remaining"],
        title="Predicted vs Actual RUL",
        labels={"Predicted RUL": "Predicted RUL (cycles)", "Actual RUL": "Actual RUL (cycles)"},
    )
    max_val = max(df["Actual RUL"].max(), df["Predicted RUL"].max()) + 5
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View fleet prediction details"):
        styled_df = df[["Engine ID", "Predicted RUL", "Actual RUL", "Error", "Days Remaining", "Status"]].sort_values("Predicted RUL")
        st.data_editor(styled_df, height=320, use_container_width=True)

def render_engine_deep_dive(df: pd.DataFrame):
    st.subheader("Engine Risk Dashboard")
    engine_list = df["Engine ID"].tolist()
    selected = st.selectbox("Select engine to inspect", engine_list, index=0)
    row = df[df["Engine ID"] == selected].iloc[0]
    engine_idx = engine_list.index(selected)

    status_color = row["Color"]
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(f"### {selected} — {row['Status']}  ")
        st.markdown(f"**Predicted RUL:** {row['Predicted RUL']} cycles  ")
        st.markdown(f"**Actual RUL:** {row['Actual RUL']} cycles  ")
        st.markdown(f"**Estimated Service Window:** ~{row['Days Remaining']} days  ")
        st.markdown(f"<span class='status-pill status-{row['Status']}'>{row['Status']}</span>", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Recommended next step**")
        if row["Status"] == "Critical":
            st.error("Schedule immediate maintenance and isolate the engine from upcoming missions.")
        elif row["Status"] == "Warning":
            st.warning("Perform detailed diagnostics and inspect the top risk sensors.")
        elif row["Status"] == "Caution":
            st.info("Monitor performance and plan service before fatigue increases.")
        else:
            st.success("Engine is stable. Continue regular monitoring.")

    with colB:
        st.plotly_chart(rul_gauge(row["Predicted RUL"], selected), use_container_width=True)

    st.markdown("---")
    trend_df = load_sensor_trends(engine_idx)
    sensor_keys = ["s02", "s03", "s04", "s07", "s11", "s12", "s15", "s20"]
    sensor_labels = {
        "s02": "Fan inlet temp",
        "s03": "LPC outlet temp",
        "s04": "HPC outlet temp",
        "s07": "HPC outlet pressure",
        "s11": "Static pressure",
        "s12": "Fuel-air ratio",
        "s15": "Bleed enthalpy",
        "s20": "Bypass ratio",
    }

    chart_cols = st.columns(2)
    for i, sensor in enumerate(sensor_keys):
        if sensor not in trend_df.columns:
            continue
        fig = go.Figure(go.Scatter(
            x=trend_df["cycle"],
            y=trend_df[sensor],
            mode="lines",
            line=dict(color="#5b8def", width=2),
        ))
        fig.update_layout(
            title=sensor_labels.get(sensor, sensor),
            height=240,
            margin=dict(l=18, r=18, t=36, b=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Cycle",
            yaxis_title="Reading",
        )
        with chart_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("View sensor distribution and raw cycle table"):
        st.data_editor(trend_df.head(120), use_container_width=True, height=300)

def render_ai_chat(df: pd.DataFrame):
    st.subheader("AeroSense GenAI Maintenance Advisor")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.warning(
            "Groq API key not found. Add `GROQ_API_KEY` to `.env` to enable the AI chatbot."
        )
        return

    chat_engine = df["Engine ID"].iloc[0]
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = chat_engine
    st.selectbox("Engine for chat context", df["Engine ID"].tolist(), key="chat_engine")
    selected_engine = st.session_state.chat_engine
    context_row = df[df["Engine ID"] == selected_engine].iloc[0]

    st.info(
        f"{selected_engine} → RUL {context_row['Predicted RUL']} cycles | Status {context_row['Status']} | "
        f"Days remaining ~{context_row['Days Remaining']}"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        container_class = "chat-user" if message["role"] == "user" else "chat-ai"
        icon = "👨‍✈️" if message["role"] == "user" else "🤖"
        st.markdown(f"<div class='{container_class}'>{icon} {message['content']}</div>", unsafe_allow_html=True)

    suggestion_cols = st.columns(2)
    suggestions = [
        "What should the maintenance team inspect first?",
        "Explain the risk profile for this engine.",
        "Which sensors indicate imminent failure?",
        "What is the best repair priority?",
    ]
    for idx, suggestion in enumerate(suggestions):
        if suggestion_cols[idx % 2].button(suggestion, key=f"suggest_{idx}"):
            st.session_state.pending_question = suggestion

    user_input = st.chat_input("Ask AeroSense about this engine and maintenance recommendations…")
    if hasattr(st.session_state, "pending_question"):
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        context_text = (
            f"Engine {selected_engine}: RUL={context_row['Predicted RUL']} cycles, "
            f"Status={context_row['Status']}, Days remaining~{context_row['Days Remaining']}"
        )
        prompt_text = f"Engine context:\n{context_text}\n\nQuestion: {user_input}"

        with st.spinner("Consulting the AeroSense knowledge base…"):
            try:
                chain = load_rag_chain()
                if chain is None:
                    answer = "⚠️ Please add your GROQ_API_KEY to the `.env` file."
                else:
                    response = chain.invoke({"query": prompt_text})
                    answer = response.get("result") if isinstance(response, dict) else str(response)
            except Exception as error:
                answer = f"Error: {error}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()

def main():
    try:
        df = load_all_predictions()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False
        st.error("Model or processed data is missing. Run preprocessing and training first.")

    render_sidebar(data_loaded)
    st.title("AeroSense — Turbofan Engine Health Monitor")
    st.markdown(
        "A premium operational dashboard for fleet health, sensor diagnostics, and AI-enabled maintenance planning."
    )
    st.markdown("---")

    if not data_loaded:
        return

    tabs = st.tabs(["Fleet Overview", "Engine Deep-Dive", "AI Diagnosis Chat"])
    with tabs[0]:
        render_fleet_overview(df)
    with tabs[1]:
        render_engine_deep_dive(df)
    with tabs[2]:
        render_ai_chat(df)

if __name__ == "__main__":
    main()
