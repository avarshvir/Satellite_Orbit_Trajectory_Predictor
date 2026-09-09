"""
config.py
============================================================
Central configuration for the Orbit / Trajectory Prediction project.

Every other module (data_loader, propagate, preprocess, train, evaluate,
app) imports its settings from HERE. Change a value once in this file and
the whole pipeline updates -- no hunting through code.

Sections:
    1. Paths            -> where data / models live
    2. Satellite(s)     -> which objects to track (NORAD IDs)
    3. Date range       -> historical window to download
    4. Propagation      -> how densely we sample the orbit
    5. Preprocessing    -> sliding-window + scaling settings
    6. Model (LSTM)     -> architecture + training hyperparameters
    7. Credentials      -> Space-Track login (loaded from .env, never hard-coded)
============================================================
"""

import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load variables from a local .env file (keeps passwords out of the code)
load_dotenv()


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
# Project root = folder that contains this file's parent (src/ -> project/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded TLE .txt files
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned position/velocity CSVs
MODELS_DIR = BASE_DIR / "models"     # saved .pth checkpoints

# Create the folders automatically if they don't exist yet
for _folder in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. SATELLITE(S) TO TRACK
# ------------------------------------------------------------
# NORAD Catalog ID is the unique number for each tracked object.
# 25544 = International Space Station (ISS) -- great starter: well-tracked,
# low Earth orbit, lots of clean historical data.
NORAD_ID = 25544
SATELLITE_NAME = "ISS (ZARYA)"


# ------------------------------------------------------------
# 3. DATE RANGE (historical TLEs to download)
# ------------------------------------------------------------
# We pull ~1 year of history to give the LSTM enough sequence data.
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 1)


# ------------------------------------------------------------
# 4. PROPAGATION SETTINGS (SGP4 baseline)
# ------------------------------------------------------------
# How often we sample the satellite's position along its orbit.
# 10 minutes is a good balance between detail and file size.
SAMPLE_STEP_MINUTES = 10

# The physical state we predict at each timestep:
# position (x, y, z) in km + velocity (vx, vy, vz) in km/s
#FEATURE_COLUMNS = ["x", "y", "z", "vx", "vy", "vz"]
#FEATURE_COLUMNS = ["inclination", "raan", "eccentricity", "arg_perigee", "mean_motion", "bstar"]

FEATURE_COLUMNS = ["inclination","raan_sin", "raan_cos", "eccentricity", "arg_perigee_sin", "arg_perigee_cos", "mean_motion", "bstar",]

# ------------------------------------------------------------
# 5. PREPROCESSING (sliding window + scaling)
# ------------------------------------------------------------
# WINDOW_SIZE  -> how many past timesteps the model looks at
# HORIZON      -> how many steps ahead we predict (1 = next step)
WINDOW_SIZE = 30
HORIZON = 1

# Fraction of data used for training (rest = testing). Split is done
# BY TIME (not random) to avoid leakage in a time series.
TRAIN_SPLIT = 0.8

# Scaler type: "minmax" or "standard" (handled in preprocess.py)
SCALER_TYPE = "minmax"


# ------------------------------------------------------------
# 6. MODEL (LSTM) HYPERPARAMETERS
# ------------------------------------------------------------
INPUT_SIZE = len(FEATURE_COLUMNS)   # 6 features in, 6 features out
OUTPUT_SIZE = len(FEATURE_COLUMNS)
HIDDEN_SIZE = 64                    # LSTM hidden units
NUM_LAYERS = 2                      # stacked LSTM layers
DROPOUT = 0.2                       # regularization between layers

# Training
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = "cuda" if os.environ.get("USE_GPU") == "1" else "cpu"  # CPU is fine here

# Where the trained model checkpoint gets saved
MODEL_PATH = MODELS_DIR / "lstm_orbit_checkpoint.pth"


# ------------------------------------------------------------
# 7. CREDENTIALS (Space-Track.org) -- loaded safely from .env
# ------------------------------------------------------------
# Create a file named ".env" in the project root with:
#   SPACETRACK_USER=your_email@example.com
#   SPACETRACK_PASS=your_password
SPACETRACK_USER = os.environ.get("SPACETRACK_USER", "")
SPACETRACK_PASS = os.environ.get("SPACETRACK_PASS", "")

# CelesTrak live-TLE URL (no login) -- used for the demo / quick tests
CELESTRAK_URL = (
    f"https://celestrak.org/NORAD/elements/gp.php?CATNR={NORAD_ID}&FORMAT=tle"
)


# ------------------------------------------------------------
# Quick self-check: run `python config.py` to verify everything is set
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Orbit Prediction -- configuration loaded successfully")
    print(f"  Satellite      : {SATELLITE_NAME} (NORAD {NORAD_ID})")
    print(f"  Date range     : {START_DATE.date()} -> {END_DATE.date()}")
    print(f"  Sample step    : {SAMPLE_STEP_MINUTES} min")
    print(f"  Window / horizon: {WINDOW_SIZE} / {HORIZON}")
    print(f"  Device         : {DEVICE}")
    print(f"  Model path     : {MODEL_PATH}")
    creds = "set" if SPACETRACK_USER else "MISSING (create a .env file)"
    print(f"  Space-Track creds: {creds}")
