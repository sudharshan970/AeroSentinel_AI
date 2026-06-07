"""
STEP 2 — Data Preprocessing & Feature Engineering
===================================================
What this script does:
  1. Loads all 4 CMAPSS sub-datasets (FD001–FD004)
  2. Calculates Remaining Useful Life (RUL) labels
  3. Clips RUL at 125 cycles (piece-wise linear degradation model)
  4. Drops constant / near-zero-variance sensors (no signal)
  5. Normalises sensor readings with MinMaxScaler (per dataset)
  6. Adds rolling-window statistical features (mean, std over 5 & 15 cycles)
  7. Saves cleaned train/test arrays + scaler to data/processed/

Run: python models/02_preprocess.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────
COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors with near-zero variance across all datasets — carry no useful signal
DROP_SENSORS = ["sensor_01", "sensor_05", "sensor_06", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]

# Rolling window sizes for statistical features
WINDOWS = [5, 15]

# Piecewise-linear RUL cap: engines degrade linearly only in last N cycles
RUL_CAP = 125


# ── Helper Functions ───────────────────────────────────────────

def load_raw(split: str, dataset_id: int) -> pd.DataFrame:
    """Load a raw CMAPSS text file into a DataFrame."""
    fpath = RAW_DIR / f"{split}_FD00{dataset_id}.txt"
    df = pd.read_csv(fpath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["dataset_id"] = dataset_id
    return df


def load_rul_targets(dataset_id: int) -> np.ndarray:
    """Load the ground-truth RUL values for the test set."""
    fpath = RAW_DIR / f"RUL_FD00{dataset_id}.txt"
    return pd.read_csv(fpath, header=None).values.flatten()


def add_rul_column(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Calculate RUL for training data.
    For each engine, max_cycle - current_cycle gives cycles remaining.
    We cap it at `rul_cap` to use the piecewise-linear degradation model:
      - Early cycles: RUL = rul_cap  (engine is healthy, not degrading fast)
      - Last rul_cap cycles: RUL decreases linearly to 0
    """
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors that provide no predictive signal."""
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_rolling_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each sensor, add rolling mean and std over multiple window sizes.
    Groups by unit_id so rolling windows don't cross engine boundaries.
    """
    new_cols = {}
    for col in feature_cols:
        for w in WINDOWS:
            grouped = df.groupby("unit_id")[col]
            new_cols[f"{col}_roll_mean_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{col}_roll_std_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def normalize(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              feature_cols: list,
              dataset_id: int):
    """
    Fit MinMaxScaler on TRAIN, apply same scaler to TEST.
    Returns scaled DataFrames and the fitted scaler.
    """
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    # Save scaler for inference at serving time
    joblib.dump(scaler, OUT_DIR / f"scaler_FD00{dataset_id}.pkl")
    return train_df, test_df, scaler


def build_sequences(df: pd.DataFrame,
                    feature_cols: list,
                    sequence_length: int = 30) -> tuple:
    """
    Build sliding-window 3D sequences for LSTM input.
    Shape: (num_samples, sequence_length, num_features)

    For each engine, we create overlapping windows of `sequence_length` cycles.
    The label for each window is the RUL at the LAST cycle in the window.
    """
    X_list, y_list = [], []

    for _, engine_df in df.groupby("unit_id"):
        features = engine_df[feature_cols].values
        labels   = engine_df["RUL"].values

        for i in range(len(features) - sequence_length + 1):
            X_list.append(features[i : i + sequence_length])
            y_list.append(labels[i + sequence_length - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_test_sequences(df: pd.DataFrame,
                         feature_cols: list,
                         sequence_length: int = 30) -> np.ndarray:
    """
    For the test set, take only the LAST `sequence_length` cycles per engine
    (that is the point where we predict remaining life).
    """
    X_list = []

    for _, engine_df in df.groupby("unit_id"):
        features = engine_df[feature_cols].values
        if len(features) >= sequence_length:
            X_list.append(features[-sequence_length:])
        else:
            # Pad with zeros at the start if engine has fewer cycles
            pad = np.zeros((sequence_length - len(features), len(feature_cols)))
            X_list.append(np.vstack([pad, features]))

    return np.array(X_list, dtype=np.float32)


# ── Main Pipeline ──────────────────────────────────────────────

def preprocess_dataset(dataset_id: int, sequence_length: int = 30) -> dict:
    """Full preprocessing pipeline for one CMAPSS sub-dataset."""
    print(f"\n  ── Processing FD00{dataset_id} ──────────────────────────")

    # 1. Load raw data
    train_df = load_raw("train", dataset_id)
    test_df  = load_raw("test",  dataset_id)
    rul_true = load_rul_targets(dataset_id)
    print(f"     Train shape : {train_df.shape}   |  {train_df['unit_id'].nunique()} engines")
    print(f"     Test shape  : {test_df.shape}    |  {test_df['unit_id'].nunique()} engines")

    # 2. Calculate RUL labels for training set
    train_df = add_rul_column(train_df, rul_cap=RUL_CAP)

    # 3. Build RUL for test set using ground-truth file
    #    Each engine in test set maps to one RUL value
    last_cycles = test_df.groupby("unit_id")["cycle"].max().reset_index()
    last_cycles["RUL"] = rul_true
    test_df = test_df.merge(last_cycles[["unit_id", "RUL"]], on="unit_id", how="left")

    # 4. Drop low-variance sensors
    train_df = drop_low_variance_sensors(train_df)
    test_df  = drop_low_variance_sensors(test_df)

    # 5. Identify sensor/feature columns (everything except metadata & target)
    meta_cols    = ["unit_id", "cycle", "dataset_id", "RUL",
                    "op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_cols  = [c for c in train_df.columns if c not in meta_cols]

    # 6. Add rolling statistical features
    print(f"     Adding rolling features (windows={WINDOWS})…")
    train_df = add_rolling_features(train_df, sensor_cols)
    test_df  = add_rolling_features(test_df,  sensor_cols)

    # 7. Final feature list (sensors + their rolling features)
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"     Total features : {len(feature_cols)}")

    # 8. Normalize
    train_df, test_df, scaler = normalize(train_df, test_df, feature_cols, dataset_id)

    # 9. Build LSTM sequences
    print(f"     Building sequences (length={sequence_length})…")
    X_train, y_train = build_sequences(train_df,  feature_cols, sequence_length)
    X_test           = build_test_sequences(test_df, feature_cols, sequence_length)
    y_test           = rul_true.astype(np.float32)

    print(f"     X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"     X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # 10. Save to disk
    tag = f"FD00{dataset_id}"
    np.save(OUT_DIR / f"X_train_{tag}.npy", X_train)
    np.save(OUT_DIR / f"y_train_{tag}.npy", y_train)
    np.save(OUT_DIR / f"X_test_{tag}.npy",  X_test)
    np.save(OUT_DIR / f"y_test_{tag}.npy",  y_test)

    # Also save flat (non-sequence) version for tree-based baselines
    flat_X_train = train_df[feature_cols].values.astype(np.float32)
    flat_y_train = train_df["RUL"].values.astype(np.float32)
    np.save(OUT_DIR / f"flat_X_train_{tag}.npy", flat_X_train)
    np.save(OUT_DIR / f"flat_y_train_{tag}.npy", flat_y_train)

    # Save feature names for later reference
    pd.Series(feature_cols).to_csv(OUT_DIR / f"feature_cols_{tag}.csv", index=False)

    print(f"     ✅  Saved to data/processed/")

    return {
        "dataset_id"   : dataset_id,
        "n_train"      : len(X_train),
        "n_test"       : len(X_test),
        "n_features"   : len(feature_cols),
        "seq_length"   : sequence_length,
        "feature_cols" : feature_cols,
    }


def main():
    print("\n" + "=" * 55)
    print("  AeroSense — Preprocessing Pipeline")
    print("=" * 55)

    summaries = []
    for ds_id in [1, 2, 3, 4]:
        summary = preprocess_dataset(ds_id, sequence_length=30)
        summaries.append(summary)

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for s in summaries:
        print(f"  FD00{s['dataset_id']}  |  "
              f"Train samples: {s['n_train']:>6,}  |  "
              f"Test engines: {s['n_test']:>3}  |  "
              f"Features: {s['n_features']}")

    print("\n  ✅  All datasets preprocessed successfully!")
    print("  Next step → python models/03_train_model.py\n")


if __name__ == "__main__":
    main()
