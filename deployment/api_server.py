"""
STEP 8 — FastAPI REST API (Production Deployment)
==================================================
Exposes 3 endpoints:
  GET  /health          — Server health check
  POST /predict/rul     — Predict RUL from sensor sequence
  POST /diagnose        — GenAI diagnosis given engine ID + question

Run: python deployment/08_api_server.py
Test: http://localhost:8000/docs  (auto-generated Swagger UI)

This is what you'd deploy on Azure Container Apps / Azure ML endpoint.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
VS_DIR     = BASE_DIR / "genai" / "vectorstore"
DEVICE     = "cpu"
sys.path.insert(0, str(BASE_DIR / "models"))

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title       = "AeroSense API",
    description = "Jet Engine Predictive Maintenance — RUL Prediction + GenAI Diagnosis",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Request/Response Models ────────────────────────────────────

class RULRequest(BaseModel):
    engine_id: int = Field(..., description="Engine index in the test set (0-based)", ge=0)

class RULResponse(BaseModel):
    engine_id       : int
    predicted_rul   : float   = Field(..., description="Predicted cycles remaining")
    status          : str     = Field(..., description="Health status: Healthy/Caution/Warning/Critical")
    action          : str     = Field(..., description="Recommended maintenance action")
    days_remaining  : float   = Field(..., description="Approximate days at 2 cycles/day")

class DiagnoseRequest(BaseModel):
    engine_id: int   = Field(..., description="Engine index", ge=0)
    question : str   = Field(..., description="Maintenance question in plain English",
                             min_length=5, max_length=500)

class DiagnoseResponse(BaseModel):
    engine_id     : int
    predicted_rul : float
    status        : str
    answer        : str


# ── Model & Chain Singletons ───────────────────────────────────

_lstm_model = None
_rag_chain  = None

def get_lstm():
    global _lstm_model
    if _lstm_model is None:
        from train_model import AeroLSTM
        tag  = "FD001"
        ckpt = torch.load(MODELS_DIR / f"lstm_{tag}.pt", map_location=DEVICE)
        m    = AeroLSTM(input_size=ckpt["input_size"]).to(DEVICE)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _lstm_model = m
    return _lstm_model


def get_rag():
    global _rag_chain
    if _rag_chain is None:
        from langchain_groq import ChatGroq
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise HTTPException(status_code=503,
                                detail="GROQ_API_KEY not set. Add to .env file.")

        emb = HuggingFaceEmbeddings(
            model_name    = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        vs  = FAISS.load_local(str(VS_DIR), emb, allow_dangerous_deserialization=True)
        llm = ChatGroq(model_name="llama-3.1-70b-versatile",
                       temperature=0.2, max_tokens=512, groq_api_key=groq_key)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="You are AeroSense, jet engine AI advisor.\n"
                     "KNOWLEDGE:\n{context}\nQUESTION: {question}\nANSWER:",
        )
        _rag_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
        )
    return _rag_chain


def predict_rul_for(engine_id: int) -> dict:
    model  = get_lstm()
    X_test = np.load(PROC_DIR / "X_test_FD001.npy")
    engine_id = max(0, min(engine_id, len(X_test) - 1))
    x     = torch.tensor(X_test[engine_id:engine_id+1]).to(DEVICE)
    with torch.no_grad():
        rul = float(model(x).item())
    rul = max(0.0, rul)

    if rul > 100:
        status = "Healthy"
        action = "Continue operations, monitor at next scheduled interval"
    elif rul > 50:
        status = "Caution"
        action = "Schedule preventive maintenance within 2 weeks"
    elif rul > 20:
        status = "Warning"
        action = "Expedite maintenance, reduce operational load"
    else:
        status = "Critical"
        action = "Ground engine immediately — do not dispatch"

    return {"rul": round(rul, 1), "status": status, "action": action}


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "AeroSense v1.0", "device": DEVICE}


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(req: RULRequest):
    """Predict Remaining Useful Life for a given engine."""
    try:
        result = predict_rul_for(req.engine_id)
        return RULResponse(
            engine_id      = req.engine_id,
            predicted_rul  = result["rul"],
            status         = result["status"],
            action         = result["action"],
            days_remaining = round(result["rul"] / 2, 1),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="Model not found. Run Steps 2–3 first.")


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    """Get a GenAI diagnosis for an engine + question."""
    result  = predict_rul_for(req.engine_id)
    context = (f"Engine ID {req.engine_id}: "
               f"RUL={result['rul']} cycles, Status={result['status']}, "
               f"Action={result['action']}")
    augmented_q = f"Engine context: {context}\n\nQuestion: {req.question}"

    try:
        chain  = get_rag()
        resp   = chain.invoke({"query": augmented_q})
        answer = resp["result"] if isinstance(resp, dict) else str(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DiagnoseResponse(
        engine_id     = req.engine_id,
        predicted_rul = result["rul"],
        status        = result["status"],
        answer        = answer,
    )


# ── Run Server ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  AeroSense API starting…")
    print("  Swagger docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
