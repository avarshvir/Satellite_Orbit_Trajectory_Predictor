"""
evaluate.py
============================================================
Evaluates the trained OrbitLSTM (8-feature, sin/cos-encoded) and produces
the headline results.

Steps:
    1. Load best checkpoint + scaler.
    2. Rebuild the test windows (same leak-free split as training).
    3. Predict -> inverse-scale back to the 8 REAL features.
    4. DECODE angles: (sin, cos) -> degrees via atan2 (wrap-aware).
    5. Per-element metrics (RMSE / MAE) in human-readable units.
    6. REAL-WORLD km error: rebuild 3D position from predicted vs. actual
       elements via SGP4, then measure position error in kilometers.
    7. Plots: learning curve, predicted-vs-actual elements, km-error hist.

Feature order (must match config.FEATURE_COLUMNS):
    0 inclination
    1 raan_sin      2 raan_cos
    3 eccentricity
    4 arg_perigee_sin  5 arg_perigee_cos
    6 mean_motion
    7 bstar

Run:
    python evaluate.py
============================================================
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from sgp4.api import Satrec, WGS72

import config
from model import build_model
from preprocess import prepare_data


PLOTS_DIR = config.MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEG2RAD = math.pi / 180.0
REVPDAY_TO_RADPMIN = (2.0 * math.pi) / 1440.0

# Column indices in the 8-feature vector
I_INCL, I_RAAN_SIN, I_RAAN_COS = 0, 1, 2
I_ECC, I_ARGP_SIN, I_ARGP_COS = 3, 4, 5
I_MM, I_BSTAR = 6, 7


# ------------------------------------------------------------
# Load the trained model
# ------------------------------------------------------------
def load_trained_model():
    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = build_model(device=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[evaluate] Loaded model (epoch {ckpt['epoch']}, "
          f"val_loss {ckpt['val_loss']:.6f})")
    return model, ckpt


# ------------------------------------------------------------
# Predict on the test set and inverse-scale to real feature units
# ------------------------------------------------------------
@torch.no_grad()
def predict_test(model, data, scaler):
    X_test = torch.from_numpy(data["X_test"]).to(config.DEVICE)
    preds_scaled = model(X_test).cpu().numpy()
    true_scaled = data["y_test"]
    y_pred = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(true_scaled)
    return y_true, y_pred


# ------------------------------------------------------------
# Decode (sin, cos) -> angle in degrees, wrapped to [0, 360)
# ------------------------------------------------------------
def decode_angle_deg(sin_vals, cos_vals):
    """atan2 is the correct wrap-aware inverse of sin/cos encoding."""
    ang = np.degrees(np.arctan2(sin_vals, cos_vals))
    return np.mod(ang, 360.0)


# ------------------------------------------------------------
# Per-element metrics (angles reported in degrees)
# ------------------------------------------------------------
def element_metrics(y_true, y_pred):
    """RMSE/MAE per human-readable element (angles decoded to degrees)."""
    # Decode angles for both sets
    raan_true = decode_angle_deg(y_true[:, I_RAAN_SIN], y_true[:, I_RAAN_COS])
    raan_pred = decode_angle_deg(y_pred[:, I_RAAN_SIN], y_pred[:, I_RAAN_COS])
    argp_true = decode_angle_deg(y_true[:, I_ARGP_SIN], y_true[:, I_ARGP_COS])
    argp_pred = decode_angle_deg(y_pred[:, I_ARGP_SIN], y_pred[:, I_ARGP_COS])

    def ang_err(a, b):
        """Smallest circular difference in degrees."""
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)

    rows = [
        _metric_row("inclination", y_true[:, I_INCL], y_pred[:, I_INCL]),
        _metric_row("raan (deg)", None, None,
                    err=ang_err(raan_pred, raan_true)),
        _metric_row("eccentricity", y_true[:, I_ECC], y_pred[:, I_ECC]),
        _metric_row("arg_perigee (deg)", None, None,
                    err=ang_err(argp_pred, argp_true)),
        _metric_row("mean_motion", y_true[:, I_MM], y_pred[:, I_MM]),
        _metric_row("bstar", y_true[:, I_BSTAR], y_pred[:, I_BSTAR]),
    ]
    table = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print("\nPer-element accuracy (real units; angles decoded to degrees):")
    print(table.to_string(index=False))
    return table


def _metric_row(name, true, pred, err=None):
    if err is None:
        err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"element": name, "RMSE": rmse, "MAE": mae}


# ------------------------------------------------------------
# Rebuild 3D position (km) from an 8-feature element vector via SGP4
# ------------------------------------------------------------
def elements_to_position(vec):
    """
    vec = [incl, raan_sin, raan_cos, ecc, argp_sin, argp_cos, mm, bstar]
    Returns ECI position (km) at the reference epoch, or None on error.
    """
    incl = vec[I_INCL]
    raan_deg = decode_angle_deg(np.array([vec[I_RAAN_SIN]]),
                                np.array([vec[I_RAAN_COS]]))[0]
    ecc = vec[I_ECC]
    argp_deg = decode_angle_deg(np.array([vec[I_ARGP_SIN]]),
                                np.array([vec[I_ARGP_COS]]))[0]
    mm = vec[I_MM]
    bstar = vec[I_BSTAR]

    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", 25544, 0.0, bstar, 0.0, 0.0,
        max(ecc, 0.0),               # eccentricity must be >= 0
        argp_deg * DEG2RAD,          # argument of perigee (rad)
        incl * DEG2RAD,              # inclination (rad)
        0.0,                         # mean anomaly (fixed; cancels in diff)
        mm * REVPDAY_TO_RADPMIN,     # mean motion (rad/min)
        raan_deg * DEG2RAD,          # RAAN (rad)
    )
    e, r, _v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if e != 0:
        return None
    return np.array(r)


# ------------------------------------------------------------
# Real-world position error in km
# ------------------------------------------------------------
def position_error_km(y_true, y_pred):
    errors = []
    for true_row, pred_row in zip(y_true, y_pred):
        p_true = elements_to_position(true_row)
        p_pred = elements_to_position(pred_row)
        if p_true is None or p_pred is None:
            continue
        errors.append(float(np.linalg.norm(p_true - p_pred)))
    return np.array(errors)


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
def plot_learning_curve():
    hist_path = config.MODELS_DIR / "history.npz"
    if not hist_path.exists():
        print("[plot] history.npz not found -- skipping learning curve.")
        return
    h = np.load(hist_path)
    plt.figure(figsize=(8, 5))
    plt.plot(h["train"], label="train loss")
    plt.plot(h["val"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = PLOTS_DIR / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] Saved {out}")


def plot_elements(y_true, y_pred):
    """Plot the 6 human-readable elements (angles decoded to degrees)."""
    raan_true = decode_angle_deg(y_true[:, I_RAAN_SIN], y_true[:, I_RAAN_COS])
    raan_pred = decode_angle_deg(y_pred[:, I_RAAN_SIN], y_pred[:, I_RAAN_COS])
    argp_true = decode_angle_deg(y_true[:, I_ARGP_SIN], y_true[:, I_ARGP_COS])
    argp_pred = decode_angle_deg(y_pred[:, I_ARGP_SIN], y_pred[:, I_ARGP_COS])

    panels = [
        ("inclination (deg)", y_true[:, I_INCL], y_pred[:, I_INCL]),
        ("raan (deg)", raan_true, raan_pred),
        ("eccentricity", y_true[:, I_ECC], y_pred[:, I_ECC]),
        ("arg_perigee (deg)", argp_true, argp_pred),
        ("mean_motion (rev/day)", y_true[:, I_MM], y_pred[:, I_MM]),
        ("bstar", y_true[:, I_BSTAR], y_pred[:, I_BSTAR]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    for i, (name, tr, pr) in enumerate(panels):
        ax = axes[i]
        ax.plot(tr, label="actual", linewidth=1.5)
        ax.plot(pr, label="predicted", linewidth=1.2, alpha=0.8)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    fig.suptitle("Predicted vs. actual orbital elements (test set)")
    out = PLOTS_DIR / "elements_pred_vs_actual.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved {out}")


def plot_km_error(errors):
    if len(errors) == 0:
        print("[plot] No km errors to plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40, color="tab:blue", alpha=0.8)
    plt.axvline(errors.mean(), color="red", linestyle="--",
                label=f"mean = {errors.mean():.1f} km")
    plt.xlabel("position error (km)")
    plt.ylabel("count")
    plt.title("Position error distribution (test set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = PLOTS_DIR / "position_error_km.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] Saved {out}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("Evaluating OrbitLSTM (sin/cos-decoded)")
    print("=" * 60)

    model, _ckpt = load_trained_model()
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    data = prepare_data(save_scaler=False)

    y_true, y_pred = predict_test(model, data, scaler)

    element_metrics(y_true, y_pred)

    errors = position_error_km(y_true, y_pred)
    if len(errors):
        print("\nReal-world POSITION error (km) -- element error -> position:")
        print(f"  mean   : {errors.mean():10.2f} km")
        print(f"  median : {np.median(errors):10.2f} km")
        print(f"  RMSE   : {np.sqrt(np.mean(errors**2)):10.2f} km")
        print(f"  max    : {errors.max():10.2f} km")

    plot_learning_curve()
    plot_elements(y_true, y_pred)
    plot_km_error(errors)

    print(f"\nAll plots saved in: {PLOTS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
