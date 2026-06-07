"""
STEP 6 — GenAI Diagnosis Chatbot (LangChain + FREE Groq API)
=============================================================
What this script does:
  1. Loads the FAISS vector store built in Step 5
  2. Loads the trained LSTM model from Step 3
  3. Creates a LangChain RAG chain:
       Sensor Data → LSTM → RUL Prediction
       User Question + RUL + Retrieved Docs → LLaMA 3.1 70B → Diagnosis
  4. Runs an interactive terminal chatbot loop

FREE Token needed: GROQ_API_KEY
  Get it FREE at: https://console.groq.com/keys
  Add to .env file: GROQ_API_KEY=your_key_here

Model used: llama-3.1-70b-versatile (free on Groq, no credit card)

Run: python genai/06_chatbot.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# ── Load environment variables ─────────────────────────────────
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
VS_DIR     = BASE_DIR / "genai" / "vectorstore"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load LSTM Model ────────────────────────────────────────────

def load_lstm():
    """Load trained LSTM for RUL inference."""
    sys.path.insert(0, str(BASE_DIR / "models"))
    from train_model import AeroLSTM

    tag  = "FD001"
    ckpt = torch.load(MODELS_DIR / f"lstm_{tag}.pt", map_location=DEVICE)
    model = AeroLSTM(input_size=ckpt["input_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_rul(model, engine_id: int) -> float:
    """
    Predict RUL for a given engine ID from the test set.
    Returns predicted RUL in cycles.
    """
    X_test = np.load(PROC_DIR / "X_test_FD001.npy")

    # Clamp engine_id to valid range
    engine_id = max(0, min(engine_id, len(X_test) - 1))
    x = torch.tensor(X_test[engine_id:engine_id+1]).to(DEVICE)

    with torch.no_grad():
        rul = model(x).item()

    return max(0.0, rul)


# ── Load Vector Store ──────────────────────────────────────────

def load_vectorstore():
    """Load the FAISS vector store built in Step 5."""
    embeddings = HuggingFaceEmbeddings(
        model_name    = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    return FAISS.load_local(str(VS_DIR), embeddings,
                            allow_dangerous_deserialization=True)


# ── Build LangChain RAG Chatbot ────────────────────────────────

SYSTEM_PROMPT = """You are AeroSense, an expert AI maintenance advisor for
Rolls-Royce turbofan jet engines. You have access to:
  1. Real-time sensor data and ML-predicted Remaining Useful Life (RUL)
  2. Aerospace maintenance knowledge from engineering documents

Your job is to:
  • Interpret sensor anomalies and RUL predictions in plain English
  • Give actionable maintenance recommendations
  • Explain what the data means for flight safety and operational planning
  • Quantify business impact (cost savings, scheduling)

Always be precise, professional, and safety-first.
Never recommend operating an engine with RUL < 20 cycles without senior engineer sign-off.
"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are AeroSense, an expert AI maintenance advisor for Rolls-Royce jet engines.

RELEVANT MAINTENANCE KNOWLEDGE:
{context}

ENGINEER'S QUESTION: {question}

Provide a clear, actionable answer. Include:
- Root cause explanation
- Recommended action with urgency level
- Safety implications if relevant

ANSWER:
""",
)


def build_chain(llm, vectorstore):
    """Build LangChain RAG retrieval chain."""
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3},
    )
    return RetrievalQA.from_chain_type(
        llm            = llm,
        chain_type     = "stuff",
        retriever      = retriever,
        chain_type_kwargs = {"prompt": RAG_PROMPT},
        return_source_documents = False,
    )


def get_engine_context(model, engine_id: int) -> str:
    """Build a sensor context string for a given engine."""
    rul = predict_rul(model, engine_id)

    # Determine alert level
    if rul > 100:
        status = "🟢 GREEN — Healthy"
        action = "Continue normal operations, monitor next scheduled interval"
    elif rul > 50:
        status = "🟡 AMBER — Caution"
        action = "Schedule preventive maintenance within next 2 weeks"
    elif rul > 20:
        status = "🟠 ORANGE — Warning"
        action = "Expedite maintenance, reduce flight load if possible"
    else:
        status = "🔴 RED — Critical"
        action = "Ground engine immediately, do not dispatch"

    return f"""
ENGINE STATUS REPORT
═══════════════════════════════════════
Engine ID        : {engine_id:04d}
Predicted RUL    : {rul:.0f} cycles (~{rul/2:.0f} flight days at 2 cycles/day)
Health Status    : {status}
Recommended Action: {action}
═══════════════════════════════════════
"""


# ── Interactive Chat Loop ──────────────────────────────────────

def chat_loop(model, rag_chain, llm):
    """Run the interactive terminal chatbot."""
    print("\n" + "═" * 60)
    print("  AeroSense — AI Maintenance Diagnosis Chatbot")
    print("  Powered by: LLaMA 3.1 70B (Groq) + LSTM + RAG")
    print("═" * 60)
    print("  Type 'engine <id>' to load an engine (e.g. 'engine 5')")
    print("  Type 'quit' to exit")
    print("═" * 60)

    current_engine_id = 0
    engine_context    = get_engine_context(model, current_engine_id)
    print(engine_context)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("  Goodbye!")
            break

        # Switch engine
        if user_input.lower().startswith("engine"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                current_engine_id = int(parts[1])
                engine_context = get_engine_context(model, current_engine_id)
                print(engine_context)
            else:
                print("  Usage: engine <number>  (e.g. engine 42)")
            continue

        # Augment question with engine context
        augmented_question = f"""
Current engine data:
{engine_context}

Engineer's question: {user_input}
"""
        print("\n  AeroSense: ", end="", flush=True)
        try:
            response = rag_chain.invoke({"query": augmented_question})
            answer   = response["result"] if isinstance(response, dict) else str(response)
            print(answer)
        except Exception as e:
            print(f"  [Error: {e}]")
            print("  Check your GROQ_API_KEY in the .env file.")


# ── Main ───────────────────────────────────────────────────────

def main():
    # Validate API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n  ❌  GROQ_API_KEY not found!")
        print("  Get your FREE key at: https://console.groq.com/keys")
        print("  Then add to .env file: GROQ_API_KEY=your_key_here")
        sys.exit(1)

    print("\n  Loading models…")

    # 1. Load LSTM
    print("  ① Loading LSTM model…", end=" ", flush=True)
    try:
        lstm_model = load_lstm()
        print("✅")
    except FileNotFoundError:
        print("❌  Run Step 3 first: python models/03_train_model.py")
        sys.exit(1)

    # 2. Load vector store
    print("  ② Loading vector store…", end=" ", flush=True)
    try:
        vectorstore = load_vectorstore()
        print("✅")
    except Exception:
        print("❌  Run Step 5 first: python genai/05_build_vectorstore.py")
        sys.exit(1)

    # 3. Init Groq LLM (FREE)
    print("  ③ Connecting to Groq (LLaMA 3.1 70B)…", end=" ", flush=True)
    llm = ChatGroq(
        model_name   = "llama-3.3-70b-versatile",
        temperature  = 0.2,     # Low temp = more factual, less creative
        max_tokens   = 512,
        groq_api_key = groq_api_key,
    )
    print("✅")

    # 4. Build RAG chain
    rag_chain = build_chain(llm, vectorstore)

    # 5. Start chat
    chat_loop(lstm_model, rag_chain, llm)


if __name__ == "__main__":
    main()
