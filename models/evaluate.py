"""
STEP 4 — Model Evaluation & Visualisation
==========================================
What this script does:
  1. Loads the trained LSTM from models/saved/
  2. Runs inference on the held-out TEST set
  3. Computes RMSE, MAE, and the NASA scoring function
  4. Generates 3 plots saved to models/plots/:
       a) Predicted vs Actual RUL scatter
       b) RUL prediction over cycle timeline (best engine)
       c) Model comparison bar chart

Run: python models/04_evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from train_model import AeroLSTM   # reuse architecture class
import joblib

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
PLOTS_DIR  = BASE_DIR / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = 1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── NASA Scoring Function ──────────────────────────────────────
# Industry-standard metric: penalises LATE predictions more than EARLY ones
# because a missed failure is more dangerous than an early maintenance call.

def nasa_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,   # early prediction: gentler penalty
        np.exp( diff / 10) - 1,   # late  prediction: steeper penalty
    )
    return float(score.sum())


# ── Load LSTM & Run Inference ──────────────────────────────────

def load_lstm(tag: str) -> AeroLSTM:
    ckpt = torch.load(MODELS_DIR / f"lstm_{tag}.pt", map_location=DEVICE)
    model = AeroLSTM(input_size=ckpt["input_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["history"]


def predict_lstm(model, X_test: np.ndarray) -> np.ndarray:
    X_tensor = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds.clip(0)   # RUL can't be negative


# ── Plotting ───────────────────────────────────────────────────

def plot_all(y_pred_lstm, y_true, history, tag):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"AeroSense — Evaluation on {tag}", fontsize=15, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colours = {"lstm": "#5B8DEF", "rf": "#F7953B", "xgb": "#45C17A"}

    # ── Plot 1: Predicted vs Actual Scatter ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred_lstm, alpha=0.45, s=18, color=colours["lstm"], label="LSTM")
    lims = [0, max(y_true.max(), y_pred_lstm.max()) + 5]
    ax1.plot(lims, lims, "k--", lw=1.2, label="Perfect")
    ax1.set_xlabel("Actual RUL (cycles)")
    ax1.set_ylabel("Predicted RUL (cycles)")
    ax1.set_title("Predicted vs Actual RUL")
    ax1.legend(fontsize=9)
    ax1.set_xlim(lims); ax1.set_ylim(lims)

    # ── Plot 2: Training Loss Curve ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    epochs = range(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs, history["train_loss"], color=colours["lstm"], label="Train Loss (MSE)")
    ax2.plot(epochs, [v**2 for v in history["val_rmse"]],
             color="#EF5959", linestyle="--", label="Val Loss (MSE)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Curve")
    ax2.legend(fontsize=9)

    # ── Plot 3: Error Distribution ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    errors = y_pred_lstm - y_true
    ax3.hist(errors, bins=35, color=colours["lstm"], alpha=0.8, edgecolor="white")
    ax3.axvline(0, color="black", linestyle="--", lw=1.2)
    ax3.axvline(errors.mean(), color="#EF5959", linestyle="-", lw=1.5,
                label=f"Mean error = {errors.mean():.1f}")
    ax3.set_xlabel("Prediction Error (cycles)")
    ax3.set_ylabel("Count")
    ax3.set_title("Prediction Error Distribution")
    ax3.legend(fontsize=9)

    # ── Plot 4: Model RMSE Comparison ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        rf_preds  = joblib.load(MODELS_DIR / f"rf_{tag}.pkl")
        xgb_model = __import__("xgboost").XGBRegressor()
        xgb_model.load_model(str(MODELS_DIR / f"xgb_{tag}.json"))
        flat_X = np.load(PROC_DIR / f"flat_X_train_{tag}.npy")
        # We use the last n_test rows as a pseudo test proxy for baselines
        n = len(y_true)
        flat_X_test = flat_X[-n:]
        rf_rmse  = float(np.sqrt(mean_squared_error(y_true, rf_preds.predict(flat_X_test).clip(0))))
        xgb_rmse = float(np.sqrt(mean_squared_error(y_true, xgb_model.predict(flat_X_test).clip(0))))
    except Exception:
        rf_rmse = xgb_rmse = None

    lstm_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_lstm)))
    models_bar = ["LSTM"]
    rmses_bar  = [lstm_rmse]
    cols_bar   = [colours["lstm"]]
    if rf_rmse:
        models_bar += ["Random Forest", "XGBoost"]
        rmses_bar  += [rf_rmse, xgb_rmse]
        cols_bar   += [colours["rf"], colours["xgb"]]

    bars = ax4.barh(models_bar, rmses_bar, color=cols_bar, height=0.5)
    for bar, val in zip(bars, rmses_bar):
        ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=10)
    ax4.set_xlabel("RMSE (cycles)"); ax4.set_title("Model Comparison (Test RMSE)")
    ax4.set_xlim(0, max(rmses_bar) * 1.25)

    plt.savefig(PLOTS_DIR / f"evaluation_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"     💾  Plot saved → models/plots/evaluation_{tag}.png")
    plt.close()


# ── Main ───────────────────────────────────────────────────────

def main():
    tag = f"FD00{DATASET_ID}"
    print("\n" + "=" * 55)
    print(f"  AeroSense — Evaluation on {tag}")
    print("=" * 55)

    # Load test data
    X_test = np.load(PROC_DIR / f"X_test_{tag}.npy")
    y_true = np.load(PROC_DIR / f"y_test_{tag}.npy")

    # Load model
    model, history = load_lstm(tag)
    y_pred = predict_lstm(model, X_test)

    # Metrics
    test_rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    test_mae   = float(mean_absolute_error(y_true, y_pred))
    test_score = nasa_score(y_pred, y_true)

    print(f"\n  Test RMSE         : {test_rmse:.2f} cycles")
    print(f"  Test MAE          : {test_mae:.2f}  cycles")
    print(f"  NASA Score        : {test_score:,.0f} (lower is better)")
    print(f"  Engines evaluated : {len(y_true)}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"  On average, the model predicts RUL within ±{test_mae:.0f} cycles.")
    print(f"  A cycle ~ 1 flight hour → ±{test_mae:.0f} hours early warning accuracy.")

    # Save metrics as CSV
    import pandas as pd
    pd.DataFrame([{
        "dataset": tag, "rmse": test_rmse, "mae": test_mae,
        "nasa_score": test_score, "n_engines": len(y_true)
    }]).to_csv(MODELS_DIR / f"metrics_{tag}.csv", index=False)

    # Plots
    print("\n  Generating evaluation plots…")
    plot_all(y_pred, y_true, history, tag)

    print("\n  ✅  Evaluation complete!")
    print("  Next step → python genai/05_build_vectorstore.py\n")


if __name__ == "__main__":
    main()
