"""
STEP 3 — Model Training: LSTM + Baseline Comparison
=====================================================
What this script does:
  1. Loads preprocessed sequences from data/processed/
  2. Trains a stacked Bidirectional LSTM for RUL regression
  3. Also trains XGBoost and Random Forest baselines (on flat features)
  4. Saves all trained models to models/saved/
  5. Prints training progress and final RMSE on validation split

Run: python models/03_train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training Hyperparameters ───────────────────────────────────
DATASET_ID  = 1          # Train on FD001 (single operating condition, easiest)
BATCH_SIZE  = 256
EPOCHS      = 60
LR          = 1e-3
VAL_SPLIT   = 0.15       # 15% of training data used for validation
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── LSTM Architecture ──────────────────────────────────────────

class AeroLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for RUL regression.

    Architecture:
      Input  → BiLSTM (128 units) → Dropout
             → BiLSTM ( 64 units) → Dropout
             → Attention pooling
             → FC(64) → ReLU → FC(1)

    Bidirectional: reads sequence forward AND backward
    Attention:     learns which time steps matter most for prediction
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Attention layer — scores each timestep
        self.attention = nn.Linear(hidden_size * 2, 1)

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)

        # Attention: compute soft weight per timestep
        attn_scores = self.attention(lstm_out)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        out = self.regressor(context)        # (batch, 1)
        return out.squeeze(-1)               # (batch,)


# ── Training Utilities ─────────────────────────────────────────

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds = 0.0, []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(X_batch)
            all_preds.append(preds.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(all_preds)


# ── LSTM Training ──────────────────────────────────────────────

def train_lstm(dataset_id: int) -> dict:
    tag = f"FD00{dataset_id}"
    print(f"\n  ── Training LSTM on {tag} ─────────────────────────────")
    print(f"     Device: {DEVICE}")

    # Load data
    X = torch.tensor(np.load(PROC_DIR / f"X_train_{tag}.npy"))
    y = torch.tensor(np.load(PROC_DIR / f"y_train_{tag}.npy"))

    n_val   = int(len(X) * VAL_SPLIT)
    n_train = len(X) - n_val
    dataset = TensorDataset(X, y)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"     Train samples : {n_train:,}   Val samples : {n_val:,}")
    print(f"     Sequence shape: {X.shape[1]} steps × {X.shape[2]} features")

    # Build model
    model = AeroLSTM(input_size=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_rmse": []}

    print(f"\n     {'Epoch':>5}  {'Train Loss':>12}  {'Val RMSE':>10}  {'LR':>10}  {'Best':>6}")
    print("     " + "-" * 55)

    for epoch in range(1, EPOCHS + 1):
        t_loss  = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        _, preds = eval_epoch(model, val_loader, criterion, DEVICE)
        y_val    = y[val_ds.indices].numpy()
        v_rmse   = rmse(preds, y_val)
        lr_now   = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_rmse"].append(v_rmse)

        is_best = v_rmse < best_val_rmse
        if is_best:
            best_val_rmse = v_rmse
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            marker = "  ★" if is_best else ""
            print(f"     {epoch:>5}  {t_loss:>12.4f}  {v_rmse:>10.2f}  "
                  f"{lr_now:>10.6f}{marker}")

    # Restore best weights
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "input_size" : X.shape[2],
        "history"    : history,
    }, MODELS_DIR / f"lstm_{tag}.pt")

    print(f"\n     ✅  Best Val RMSE: {best_val_rmse:.2f} cycles")
    print(f"     💾  Saved → models/saved/lstm_{tag}.pt")
    return {"tag": tag, "model": "LSTM", "val_rmse": best_val_rmse}


# ── Baseline Models ────────────────────────────────────────────

def train_baselines(dataset_id: int) -> list:
    tag = f"FD00{dataset_id}"
    print(f"\n  ── Training Baselines on {tag} ────────────────────────")

    X = np.load(PROC_DIR / f"flat_X_train_{tag}.npy")
    y = np.load(PROC_DIR / f"flat_y_train_{tag}.npy")

    n_val   = int(len(X) * VAL_SPLIT)
    rng     = np.random.default_rng(42)
    idx     = rng.permutation(len(X))
    val_idx = idx[:n_val]; trn_idx = idx[n_val:]

    X_tr, y_tr = X[trn_idx], y[trn_idx]
    X_vl, y_vl = X[val_idx], y[val_idx]

    results = []

    # Random Forest
    print("     Training Random Forest…", end="", flush=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_rmse = rmse(rf.predict(X_vl), y_vl)
    joblib.dump(rf, MODELS_DIR / f"rf_{tag}.pkl")
    print(f"  Val RMSE: {rf_rmse:.2f}  ({time.time()-t0:.0f}s)")
    results.append({"tag": tag, "model": "RandomForest", "val_rmse": rf_rmse})

    # XGBoost
    print("     Training XGBoost…", end="", flush=True)
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       early_stopping_rounds=30, eval_metric="rmse",
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    xgb_rmse = rmse(xgb.predict(X_vl), y_vl)
    xgb.save_model(str(MODELS_DIR / f"xgb_{tag}.json"))
    print(f"  Val RMSE: {xgb_rmse:.2f}  ({time.time()-t0:.0f}s)")
    results.append({"tag": tag, "model": "XGBoost", "val_rmse": xgb_rmse})

    return results


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  AeroSense — Model Training")
    print("=" * 55)

    all_results = []

    # Train on FD001 (primary) — you can loop over [1,2,3,4] for all datasets
    lstm_result      = train_lstm(DATASET_ID)
    baseline_results = train_baselines(DATASET_ID)

    all_results.append(lstm_result)
    all_results.extend(baseline_results)

    # Summary table
    print("\n" + "=" * 55)
    print("  VALIDATION RMSE COMPARISON (FD001, lower is better)")
    print("=" * 55)
    for r in sorted(all_results, key=lambda x: x["val_rmse"]):
        bar = "█" * int(r["val_rmse"] / 3)
        print(f"  {r['model']:<15}  RMSE: {r['val_rmse']:>6.2f}  {bar}")

    print("\n  ✅  Training complete!")
    print("  Next step → python models/04_evaluate.py\n")


if __name__ == "__main__":
    main()
