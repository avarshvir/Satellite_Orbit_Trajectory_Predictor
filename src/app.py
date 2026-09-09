"""
app.py
============================================================
Streamlit demo for the Orbit / Trajectory Predictor.

What it shows
-------------
    * Pick a sample from the held-out TEST set with a slider.
    * The trained LSTM forecasts the NEXT orbital state.
    * We decode the sin/cos angles back to degrees and show a clean
      predicted-vs-actual table.
    * We reconstruct a full 3D orbit (via SGP4) from the predicted
      elements and render it around a simple Earth sphere.
    * We report the position error (km) for that sample.

The pipeline is satellite-agnostic: everything keys off config.NORAD_ID,
so pointing it at a different object only needs that one number changed.

Run:
    streamlit run app.py
============================================================
"""

from __future__ import annotations

import math

import numpy as np
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
from sgp4.api import Satrec, WGS72

import config
from model import build_model
from preprocess import prepare_data

# Reuse the decode + reconstruction helpers from evaluate.py
from evaluate import (
    decode_angle_deg,
    elements_to_position,
    I_INCL, I_RAAN_SIN, I_RAAN_COS, I_ECC,
    I_ARGP_SIN, I_ARGP_COS, I_MM, I_BSTAR,
)

DEG2RAD = math.pi / 180.0
REVPDAY_TO_RADPMIN = (2.0 * math.pi) / 1440.0
EARTH_RADIUS_KM = 6371.0


@st.cache_resource
def load_model_and_scaler():
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
    model = build_model(device="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    return model, scaler, ckpt


@st.cache_data
def load_test_data():
    data = prepare_data(save_scaler=False)
    return data


def full_orbit_track(vec, n_points: int = 200):
    """
    Propagate one element set across a full period to trace the orbit ring.
    Returns (x, y, z) arrays in km.
    """
    incl = vec[I_INCL]
    raan_deg = decode_angle_deg(np.array([vec[I_RAAN_SIN]]),
                                np.array([vec[I_RAAN_COS]]))[0]
    ecc = max(vec[I_ECC], 0.0)
    argp_deg = decode_angle_deg(np.array([vec[I_ARGP_SIN]]),
                                np.array([vec[I_ARGP_COS]]))[0]
    mm = vec[I_MM]
    bstar = vec[I_BSTAR]

    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", 25544, 0.0, bstar, 0.0, 0.0,
        ecc, argp_deg * DEG2RAD, incl * DEG2RAD,
        0.0, mm * REVPDAY_TO_RADPMIN, raan_deg * DEG2RAD,
    )

    # One orbital period in minutes = 1440 / mean_motion
    period_min = 1440.0 / mm
    xs, ys, zs = [], [], []
    for k in range(n_points):
        tsince = (k / n_points) * period_min          # minutes into the orbit
        jd = sat.jdsatepoch
        fr = sat.jdsatepochF + tsince / 1440.0
        e, r, _v = sat.sgp4(jd, fr)
        if e == 0:
            xs.append(r[0]); ys.append(r[1]); zs.append(r[2])
    return np.array(xs), np.array(ys), np.array(zs)


def earth_mesh(radius: float = EARTH_RADIUS_KM, n: int = 30):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Orbit Predictor", page_icon="🛰️",
                       layout="wide")

    st.title("🛰️ Satellite Orbit / Trajectory Predictor")
    st.caption(
        f"LSTM forecasting of orbital-element evolution — "
        f"validated on **{config.SATELLITE_NAME}** (NORAD {config.NORAD_ID}). "
        f"Satellite-agnostic: change NORAD_ID in config.py to retarget."
    )

    model, scaler, ckpt = load_model_and_scaler()
    data = load_test_data()

    X_test = data["X_test"]
    y_test = data["y_test"]
    n_samples = len(X_test)

    # ---- Sidebar controls ----
    st.sidebar.header("Controls")
    idx = st.sidebar.slider("Test sample", 0, n_samples - 1, 0)
    st.sidebar.write(f"Model: epoch {ckpt['epoch']}, "
                     f"val loss {ckpt['val_loss']:.5f}")
    st.sidebar.write(f"Test samples: {n_samples}")

    # ---- Predict the chosen sample ----
    x_in = torch.from_numpy(X_test[idx:idx + 1])
    with torch.no_grad():
        pred_scaled = model(x_in).numpy()

    pred = scaler.inverse_transform(pred_scaled)[0]
    true = scaler.inverse_transform(y_test[idx:idx + 1])[0]

    # Decode angles for display
    raan_p = decode_angle_deg(np.array([pred[I_RAAN_SIN]]),
                              np.array([pred[I_RAAN_COS]]))[0]
    raan_t = decode_angle_deg(np.array([true[I_RAAN_SIN]]),
                              np.array([true[I_RAAN_COS]]))[0]
    argp_p = decode_angle_deg(np.array([pred[I_ARGP_SIN]]),
                              np.array([pred[I_ARGP_COS]]))[0]
    argp_t = decode_angle_deg(np.array([true[I_ARGP_SIN]]),
                              np.array([true[I_ARGP_COS]]))[0]

    # Position error for this sample
    p_true = elements_to_position(true)
    p_pred = elements_to_position(pred)
    km_err = (float(np.linalg.norm(p_true - p_pred))
              if p_true is not None and p_pred is not None else float("nan"))

    # ---- Layout: table + metric on the left, 3D plot on the right ----
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Predicted vs. actual (next state)")
        st.table({
            "element": ["inclination (deg)", "raan (deg)", "eccentricity",
                        "arg_perigee (deg)", "mean_motion (rev/day)", "bstar"],
            "actual": [f"{true[I_INCL]:.4f}", f"{raan_t:.2f}",
                       f"{true[I_ECC]:.6f}", f"{argp_t:.2f}",
                       f"{true[I_MM]:.5f}", f"{true[I_BSTAR]:.6f}"],
            "predicted": [f"{pred[I_INCL]:.4f}", f"{raan_p:.2f}",
                          f"{pred[I_ECC]:.6f}", f"{argp_p:.2f}",
                          f"{pred[I_MM]:.5f}", f"{pred[I_BSTAR]:.6f}"],
        })
        st.metric("Position error (this sample)", f"{km_err:,.1f} km")

    with right:
        st.subheader("Reconstructed orbit (3D)")
        # Earth
        ex, ey, ez = earth_mesh()
        fig = go.Figure()
        fig.add_surface(x=ex, y=ey, z=ez, colorscale="Blues",
                        showscale=False, opacity=0.6, name="Earth")

        # Actual orbit ring
        ax, ay, az = full_orbit_track(true)
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines",
                                   line=dict(width=4), name="actual orbit"))

        # Predicted orbit ring
        px, py, pz = full_orbit_track(pred)
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines",
                                   line=dict(width=4, dash="dash"),
                                   name="predicted orbit"))

        fig.update_layout(
            scene=dict(aspectmode="data",
                       xaxis_title="x (km)", yaxis_title="y (km)",
                       zaxis_title="z (km)"),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0, y=1),
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The model predicts the **evolution of orbital elements** (driven by "
        "atmospheric drag), then SGP4 reconstructs the 3D orbit. Angle "
        "wraparound is handled with sin/cos encoding."
    )


if __name__ == "__main__":
    main()
