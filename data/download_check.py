"""
STEP 1 — Dataset Download Checker
===================================
Run this FIRST after downloading the NASA CMAPSS dataset.

Dataset URL: https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip

After unzipping, copy all .txt files into data/raw/
Then run: python data/01_download_check.py
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"

# ── Expected files ─────────────────────────────────────────────
EXPECTED_FILES = (
    [f"train_FD00{i}.txt" for i in range(1, 5)] +
    [f"test_FD00{i}.txt"  for i in range(1, 5)] +
    [f"RUL_FD00{i}.txt"   for i in range(1, 5)]
)

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i:02d}" for i in range(1, 22)]
)  # 26 columns total


def check_raw_files():
    """Verify all expected dataset files exist."""
    print("\n" + "=" * 55)
    print("  AeroSense — Dataset Integrity Check")
    print("=" * 55)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    missing, found = [], []

    for fname in EXPECTED_FILES:
        fpath = RAW_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✅  {fname:<25}  ({size_kb:>7.1f} KB)")
            found.append(fname)
        else:
            print(f"  ❌  {fname:<25}  NOT FOUND")
            missing.append(fname)

    print("-" * 55)
    print(f"  Found: {len(found)}/12    Missing: {len(missing)}/12")

    if missing:
        print("\n  ⚠️  ACTION REQUIRED:")
        print("  1. Download dataset from:")
        print("     https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip")
        print("  2. Unzip the file")
        print(f"  3. Copy all .txt files to:  {RAW_DIR}")
        print("\n  Missing files:")
        for f in missing:
            print(f"     • {f}")
        sys.exit(1)
    else:
        print("\n  ✅  All files present! Running quick sanity check…\n")
        return True


def sanity_check():
    """Load FD001 train file and print shape + sample."""
    import pandas as pd

    fpath = RAW_DIR / "train_FD001.txt"
    df = pd.read_csv(fpath, sep=r"\s+", header=None, names=COLUMN_NAMES)

    print(f"  Dataset: train_FD001.txt")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Engines: {df['unit_id'].nunique()} unique turbofan engines")
    print(f"  Cycles : {df['cycle'].min()} – {df['cycle'].max()}")
    print(f"\n  Sample (first 3 rows):\n")
    print(df.head(3).to_string(index=False))
    print("\n  Column summary:")
    print(f"    • Operational settings : op_setting_1 to op_setting_3")
    print(f"    • Sensor readings      : sensor_01 to sensor_21")
    print(f"    • Target variable      : RUL (will be calculated in Step 2)")
    print("\n  ✅  Sanity check passed. Run Step 2 next:")
    print("      python models/02_preprocess.py\n")


if __name__ == "__main__":
    if check_raw_files():
        sanity_check()
