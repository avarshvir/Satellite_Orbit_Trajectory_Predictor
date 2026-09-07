"""
preprocess.py
============================================================
Turns the clean state CSV into model-ready sequences for the LSTM.

Three jobs:
    1. SCALE     -> normalize the 6 features (position is ~6780 km,
                    velocity is ~7 km/s -- very different magnitudes).
                    The fitted scaler is SAVED so we can invert predictions
                    later (a super common rookie mistake is losing it).
    2. WINDOW    -> slice the long series into (past -> next) samples:
                    "given the last WINDOW_SIZE states, predict the next".
    3. SPLIT     -> first 80% train / last 20% test, BY TIME (no shuffle),
                    so there is zero data leakage in the time-series.

Public functions
----------------
    load_states()                 -> DataFrame from data/processed CSV
    make_windows(array)           -> (X, y) sliding-window arrays
    prepare_data()                -> full pipeline: returns splits + scaler

Run directly to build + preview the arrays:
    python preprocess.py
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config


# ------------------------------------------------------------
# 1. LOAD THE PROCESSED STATE CSV
# ------------------------------------------------------------
def load_states(csv_path: Path | None = None) -> pd.DataFrame:
    """Read the position/velocity time-series produced by propagate.py."""
    if csv_path is None:
        csv_path = config.PROCESSED_DIR / f"states_{config.NORAD_ID}.csv"

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"State CSV not found: {csv_path}\n"
            f"Run propagate.py first to create it."
        )

    df = pd.read_csv(csv_path, parse_dates=["epoch"])
    df = df.sort_values("epoch").reset_index(drop=True)
    print(f"[preprocess] Loaded {len(df)} states from {Path(csv_path).name}")
    return df


# ------------------------------------------------------------
# 2. BUILD SLIDING WINDOWS  (past -> next)
# ------------------------------------------------------------
def make_windows(
    data: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    horizon: int = config.HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a (T, 6) array of states into supervised samples.

    For each position i:
        X = states[i : i + window_size]          shape (window_size, 6)
        y = states[i + window_size + horizon-1]  shape (6,)

    Returns:
        X -> (num_samples, window_size, 6)   the input sequences
        y -> (num_samples, 6)                the target next-state
    """
    X_list, y_list = [], []
    last_start = len(data) - window_size - horizon + 1

    for i in range(last_start):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size + horizon - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ------------------------------------------------------------
# 3. FULL PIPELINE: scale -> split (by time) -> window
# ------------------------------------------------------------
def prepare_data(save_scaler: bool = True):
    """
    Run the full preprocessing pipeline.

    Order matters to avoid leakage:
        a) split raw rows by time FIRST
        b) fit the scaler on TRAIN ONLY, then apply to both
        c) window each split separately

    Returns a dict with:
        X_train, y_train, X_test, y_test, scaler, feature_cols
    """
    df = load_states()
    features = config.FEATURE_COLUMNS
    values = df[features].values.astype(np.float32)   # shape (T, 6)

    # ---- a) time-based split point (NO shuffle) ----
    split_idx = int(len(values) * config.TRAIN_SPLIT)
    train_raw = values[:split_idx]
    test_raw = values[split_idx:]
    print(f"[preprocess] Split by time -> {len(train_raw)} train / "
          f"{len(test_raw)} test rows")

    # ---- b) fit scaler on TRAIN ONLY ----
    scaler = MinMaxScaler() if config.SCALER_TYPE == "minmax" else StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    if save_scaler:
        scaler_path = config.MODELS_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"[preprocess] Saved scaler -> {scaler_path}")

    # ---- c) window each split separately ----
    X_train, y_train = make_windows(train_scaled)
    X_test, y_test = make_windows(test_scaled)

    print(f"[preprocess] X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[preprocess] X_test  {X_test.shape},  y_test  {y_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": features,
    }


# ------------------------------------------------------------
# Run directly: build the arrays and preview shapes
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessing: CSV -> scaled sliding-window sequences")
    print("=" * 60)

    data = prepare_data()

    print("\nShapes ready for the LSTM:")
    print(f"  X_train: {data['X_train'].shape}  (samples, window, features)")
    print(f"  y_train: {data['y_train'].shape}  (samples, features)")
    print(f"  X_test : {data['X_test'].shape}")
    print(f"  y_test : {data['y_test'].shape}")

    # Show one sample so it feels concrete
    if len(data["X_train"]):
        print(f"\nOne training sample:")
        print(f"  input  = last {config.WINDOW_SIZE} states "
              f"(each has {config.INPUT_SIZE} features)")
        print(f"  target = the next state -> {data['y_train'][0]}")

    print("\nDone.")
