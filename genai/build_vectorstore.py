"""
STEP 5 — Build RAG Vector Store (Aerospace Knowledge Base)
==========================================================
What this script does:
  1. Creates a synthetic aerospace maintenance knowledge base
     (in production you would replace this with real Rolls-Royce docs,
      maintenance manuals, fault code libraries, etc.)
  2. Splits documents into chunks
  3. Embeds them using a FREE HuggingFace model (all-MiniLM-L6-v2)
     — runs locally, NO API calls, NO cost
  4. Saves the FAISS vector store to genai/vectorstore/

FREE Tokens needed here: NONE (embeddings run locally)

Run: python genai/05_build_vectorstore.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
VS_DIR    = BASE_DIR / "genai" / "vectorstore"
VS_DIR.mkdir(parents=True, exist_ok=True)

# ── Aerospace Knowledge Base ───────────────────────────────────
# In a real project, load PDFs/Word docs from:
#   loader = PyPDFLoader("rolls_royce_manual.pdf")
# Here we use a rich synthetic knowledge base covering real RR topics.

AEROSPACE_DOCS = [
    """
    TURBOFAN ENGINE FAULT CODES — SENSOR GUIDE
    ============================================
    Sensor 02 (Fan inlet temperature): Normal range 288–320 K.
    Readings above 340 K indicate potential compressor inlet blockage or
    abnormal atmospheric conditions. Sustained elevation > 5 cycles warrants
    ground inspection of inlet cowl and foreign object debris filters.

    Sensor 03 (LPC outlet temperature): Normal range 350–420 K.
    Elevated readings suggest Low Pressure Compressor (LPC) efficiency degradation.
    This sensor is one of the MOST PREDICTIVE indicators of remaining useful life.
    A trend of rising LPC outlet temp over 20+ cycles strongly correlates with
    reduced RUL (remaining useful life) of 30–60 cycles.

    Sensor 04 (HPC outlet temperature): Normal range 550–650 K.
    Critical indicator of High Pressure Compressor health.
    Values exceeding 680 K for more than 3 cycles indicate HPC blade erosion.
    Immediate borescope inspection recommended.
    """,

    """
    REMAINING USEFUL LIFE (RUL) INTERPRETATION GUIDE
    =================================================
    RUL Definition: The number of operational cycles remaining before a
    component reaches its failure threshold.

    RUL Thresholds for Maintenance Planning:
    ─────────────────────────────────────────
    • RUL > 100 cycles  → GREEN  — No action required, normal monitoring
    • RUL 50–100 cycles → AMBER  — Schedule preventive maintenance
    • RUL 20–50 cycles  → ORANGE — Expedite maintenance, reduce load
    • RUL < 20 cycles   → RED    — Ground immediately, do not operate

    A cycle is defined as one complete engine start-to-shutdown event,
    typically corresponding to one flight segment (takeoff → landing).
    Wide-body aircraft complete approximately 1–3 cycles per day.

    If the ML model predicts RUL = 35 cycles for Engine #42,
    and the aircraft flies 2 cycles/day, the engine should be serviced
    within 17–18 days to prevent unscheduled removal.
    """,

    """
    COMMON ENGINE FAULT SIGNATURES
    ================================
    1. COMPRESSOR FOULING
       Symptoms: Gradual rise in sensor_03, sensor_04; drop in sensor_11 (pressure ratio)
       Cause: Ingestion of dust, salt, carbon deposits on compressor blades
       Effect: Reduces compression efficiency by 1–3%; increases RUL degradation rate
       Action: Hot-section wash every 200 cycles in high-particulate environments

    2. TURBINE BLADE EROSION
       Symptoms: Rising sensor_09 (burner pressure ratio drops), rising EGT
       Cause: Hot gas path corrosion, thermal fatigue from temperature cycling
       Effect: Reduces turbine efficiency; can cause catastrophic failure if untreated
       Action: Borescope inspection; replace blades if erosion > 0.3mm at tip

    3. BEARING DEGRADATION
       Symptoms: Vibration increase in sensor_14, sensor_15; oil temperature rise
       Cause: Lubrication breakdown, metal fatigue, contamination
       Effect: Can lead to shaft failure; one of top causes of inflight shutdowns
       Action: Oil analysis every 50 cycles; replace bearings at 8,000 cycle TBO

    4. FUEL NOZZLE COKING
       Symptoms: Combustor exit temperature non-uniformity, pattern factor increase
       Cause: Fuel residue at high temperatures, poor fuel quality
       Effect: Hot spots on turbine nozzle guide vanes
       Action: Inspect and clean fuel nozzles every 500 cycles
    """,

    """
    PREDICTIVE MAINTENANCE COST ANALYSIS
    ======================================
    The cost of maintenance decisions:

    SCHEDULED MAINTENANCE (planned removal at predicted RUL):
    • Engine removal cost      : $15,000–$25,000 (labour + crane + logistics)
    • Shop visit cost          : $500,000–$2,000,000 depending on work scope
    • Aircraft out-of-service  : 2–6 weeks
    • Total (planned)          : ~$600,000

    UNSCHEDULED REMOVAL (engine fails in service):
    • Emergency AOG (Aircraft on Ground) labour premium: +40%
    • Replacement engine lease : $50,000–$100,000/week
    • Passenger compensation   : $200–$600 per passenger (EU261 regulations)
    • Reputation/rebooking costs: $100,000–$500,000
    • Total (unscheduled)      : $1,500,000–$3,000,000

    ROI of Predictive Maintenance: Every accurately predicted RUL saves
    approximately $900,000–$2,400,000 vs. reactive maintenance.

    A 10% improvement in RUL prediction accuracy across a 300-engine fleet
    is estimated to save $45M–$120M annually.
    """,

    """
    ROLLS-ROYCE TRENT ENGINE FAMILY — MAINTENANCE INTERVALS
    =========================================================
    The Trent family (Trent 700, 800, 900, 1000, XWB) powers most modern
    wide-body aircraft including the Airbus A330, Boeing 777, A380, A350.

    Standard On-Wing Inspection Intervals:
    • Engine Water Wash       : Every 200 flight cycles
    • Borescope Inspection    : Every 1,000 flight cycles or 3,000 hours
    • Hot Section Inspection  : Every 3,000 cycles
    • Engine Shop Visit (ESV) : Every 6,000–10,000 cycles (performance-based)

    Condition-Based Maintenance (CBM):
    Rolls-Royce uses the CareStore and IntelligentEngine platforms to
    monitor real-time sensor data from in-service engines. Anomalies
    trigger automatic alerts to the Engine Health Monitoring (EHM) team.
    Predictive models estimate time-to-event for each monitored failure mode.

    This project replicates the core ML + GenAI layer of that EHM system.
    """,

    """
    OPERATIONAL SETTINGS — FLIGHT ENVELOPE IMPACT ON ENGINE HEALTH
    ================================================================
    The CMAPSS dataset includes 3 operational settings that affect sensor baselines:

    Operating Condition FD001/FD003 (Single condition):
    • High altitude cruise — thin air, low pressure ratio demand
    • Typical: altitude ~35,000 ft, Mach 0.84

    Operating Condition FD002/FD004 (Six conditions mixed):
    • Ground testing at sea level → sensors read 15–20% higher absolute values
    • Short-haul cycles → faster thermal cycling, faster degradation
    • Long-haul cruise → steady state, gentler on hot section

    KEY IMPLICATION for ML models:
    Condition normalisation is CRITICAL. A sensor reading of 600 K may be
    normal at sea level but dangerous at altitude. Always check op_setting_1
    (altitude/Mach combination) before interpreting raw sensor values.
    The preprocessing step handles this by fitting one scaler per dataset.
    """,
]


def main():
    print("\n" + "=" * 55)
    print("  AeroSense — Building RAG Vector Store")
    print("=" * 55)

    # 1. Split documents into chunks
    print("\n  Splitting documents into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 600,
        chunk_overlap = 80,
        separators    = ["\n\n", "\n", ".", " "],
    )
    chunks = splitter.create_documents(AEROSPACE_DOCS)
    print(f"     {len(AEROSPACE_DOCS)} documents → {len(chunks)} chunks")

    # 2. Load FREE local embedding model (downloads ~90 MB once, then cached)
    print("\n  Loading HuggingFace embedding model (free, runs locally)…")
    print("     Model: sentence-transformers/all-MiniLM-L6-v2")
    print("     First run downloads ~90 MB — subsequent runs use cache")
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    # 3. Build FAISS vector store
    print("\n  Building FAISS vector store…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Save locally
    vectorstore.save_local(str(VS_DIR))
    print(f"     ✅  Vector store saved → genai/vectorstore/")
    print(f"     Files: {list(VS_DIR.iterdir())}")

    # 5. Quick test retrieval
    print("\n  Testing retrieval…")
    results = vectorstore.similarity_search("What does high sensor 04 reading mean?", k=2)
    print(f"     Query: 'What does high sensor 04 reading mean?'")
    print(f"     Top result: {results[0].page_content[:120]}…")

    print("\n  ✅  Vector store ready!")
    print("  Next step → python genai/06_chatbot.py\n")


if __name__ == "__main__":
    main()
