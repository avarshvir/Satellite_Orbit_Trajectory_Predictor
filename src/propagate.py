"""
propagate.py
============================================================
Turns raw TLE text into a clean numeric time-series of ORBITAL ELEMENTS,
with CYCLIC ANGLE ENCODING (sin/cos) for the wrapping angles.

WHY ELEMENTS (not raw x,y,z)?
    Consecutive ISS TLEs are ~4.4 h apart but the ISS orbits every ~93 min,
    so raw position jumps ~2.8 orbits between samples (unlearnable). Orbital
    elements drift slowly/smoothly -> a real, learnable trend.

WHY sin/cos FOR ANGLES?
    RAAN and argument-of-perigee are angles that wrap 359 deg -> 0 deg.
    Raw degrees make that wrap look like a huge jump (a "cliff"), which the
    model smooths across and produces big spurious errors. Encoding each
    wrapping angle as (sin, cos) puts 359 deg and 1 deg right next to each
    other on the unit circle -> smooth, no cliff, much lower position error.

    inclination is NOT encoded: for the ISS it barely moves (~51.64 deg) and
    never wraps, so raw degrees are fine and keep the feature count small.

Output columns (must match config.FEATURE_COLUMNS, 8 features):
    epoch,
    inclination,
    raan_sin, raan_cos,
    eccentricity,
    arg_perigee_sin, arg_perigee_cos,
    mean_motion,
    bstar

Public functions
----------------
    tle_to_elements(l1, l2)     -> tuple of (epoch + 8 features)
    build_time_series(records)  -> pandas DataFrame
    save_time_series(df)        -> writes CSV to data/processed/

Run directly:
    python propagate.py
============================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
from sgp4.api import Satrec

import config
from data_loader import load_tles_from_file, TLE


# One record = epoch + the 8 feature values (see module docstring)
Elements = Tuple[datetime, float, float, float, float, float, float, float, float]

RAD2DEG = 180.0 / math.pi
RADPMIN_TO_REVPDAY = 1440.0 / (2.0 * math.pi)


# ------------------------------------------------------------
# 1. READ ELEMENTS FROM ONE TLE (with sin/cos angle encoding)
# ------------------------------------------------------------
def tle_to_elements(line1: str, line2: str) -> Optional[Elements]:
    """
    Parse one TLE and return its elements at its own epoch, with the two
    wrapping angles encoded as (sin, cos):

        (epoch,
         inclination_deg,
         raan_sin, raan_cos,
         eccentricity,
         argp_sin, argp_cos,
         mean_motion_rev_per_day,
         bstar)

    Returns None if the TLE can't be parsed.
    """
    try:
        sat = Satrec.twoline2rv(line1, line2)
    except Exception:
        return None

    inclination = sat.inclo * RAD2DEG        # degrees (stable, no wrap)
    raan_rad = sat.nodeo                      # radians (wraps) -> sin/cos
    eccentricity = sat.ecco                   # unitless
    argp_rad = sat.argpo                      # radians (wraps) -> sin/cos
    mean_motion = sat.no_kozai * RADPMIN_TO_REVPDAY  # rev/day
    bstar = sat.bstar                         # drag term

    raan_sin, raan_cos = math.sin(raan_rad), math.cos(raan_rad)
    argp_sin, argp_cos = math.sin(argp_rad), math.cos(argp_rad)

    epoch_dt = _jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)

    return (epoch_dt,
            inclination,
            raan_sin, raan_cos,
            eccentricity,
            argp_sin, argp_cos,
            mean_motion,
            bstar)


# ------------------------------------------------------------
# 2. BUILD THE FULL TIME-SERIES
# ------------------------------------------------------------
def build_time_series(records: List[TLE]) -> pd.DataFrame:
    """
    Convert (name, l1, l2) TLEs into a time-ordered DataFrame with columns
    [epoch] + config.FEATURE_COLUMNS. Skips bad TLEs, sorts, de-duplicates.
    """
    rows: List[Elements] = []
    skipped = 0

    for _name, l1, l2 in records:
        elements = tle_to_elements(l1, l2)
        if elements is None:
            skipped += 1
            continue
        rows.append(elements)

    if not rows:
        raise RuntimeError("No valid elements produced -- check the TLE file.")

    columns = ["epoch"] + config.FEATURE_COLUMNS
    df = pd.DataFrame(rows, columns=columns)

    df = df.sort_values("epoch").drop_duplicates(subset="epoch")
    df = df.reset_index(drop=True)

    print(f"[propagate] Built time-series: {len(df)} records "
          f"({skipped} skipped)")
    return df


# ------------------------------------------------------------
# 3. SAVE TO CSV
# ------------------------------------------------------------
def save_time_series(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Write the element time-series to data/processed/ as a CSV."""
    if out_path is None:
        out_path = config.PROCESSED_DIR / f"states_{config.NORAD_ID}.csv"
    df.to_csv(out_path, index=False)
    print(f"[propagate] Saved -> {out_path}")
    return out_path


# ------------------------------------------------------------
# Helper: two-part Julian date -> UTC datetime
# ------------------------------------------------------------
def _jd_to_datetime(jd: float, fr: float) -> datetime:
    """Convert a two-part Julian date (jd + fractional fr) into UTC datetime."""
    total = jd + fr
    jd_adj = total + 0.5
    Z = int(jd_adj)
    F = jd_adj - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day_frac = B - D - int(30.6001 * E) + F
    day = int(day_frac)
    frac = day_frac - day

    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    seconds_total = frac * 86400.0
    hours = int(seconds_total // 3600)
    minutes = int((seconds_total % 3600) // 60)
    seconds = int(seconds_total % 60)
    microseconds = int((seconds_total - int(seconds_total)) * 1_000_000)

    return datetime(year, month, day, hours, minutes, seconds,
                    microseconds, tzinfo=timezone.utc)


# ------------------------------------------------------------
# Run directly
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Extract orbital ELEMENTS (sin/cos angle encoding)")
    print("=" * 60)

    tle_file = config.RAW_DIR / f"tle_history_{config.NORAD_ID}.txt"
    records = load_tles_from_file(tle_file)

    df = build_time_series(records)
    save_time_series(df)

    pd.set_option("display.float_format", lambda v: f"{v:12.6f}")
    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    # Sanity: mean motion should be ~15.5 rev/day and gently rising (drag)
    n_start = df["mean_motion"].iloc[0]
    n_end = df["mean_motion"].iloc[-1]
    print(f"\nSanity check -> mean motion: {n_start:.5f} -> {n_end:.5f} rev/day")
    # Sanity: sin^2 + cos^2 should be ~1 for the encoded angles
    check = (df["raan_sin"] ** 2 + df["raan_cos"] ** 2).mean()
    print(f"Sanity check -> raan_sin^2 + raan_cos^2 (mean): {check:.5f}  (~1.0)")

    print("\nDone.")
