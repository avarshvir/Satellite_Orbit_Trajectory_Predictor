"""
data_loader.py
============================================================
Downloads TLE (Two-Line Element) data for the target satellite.

Single source -- clean and simple:
    Space-Track.org (HISTORICAL) -> via the `spacetrack` library.
                                    Needs a free account (creds in .env).
                                    This is your training + testing data.

Public functions
----------------
    download_historical_tles()  -> saves a year of TLEs to data/raw/
    load_tles_from_file(path)   -> reads a TLE file into (name, l1, l2) tuples

Run directly to fetch the data:
    python data_loader.py
============================================================
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Import our own settings (single source of truth)
import config


# A TLE record = (satellite_name, line1, line2)
TLE = Tuple[str, str, str]


# ------------------------------------------------------------
# 1. HISTORICAL TLEs  (Space-Track.org via spacetrack library)
# ------------------------------------------------------------
def download_historical_tles(
    norad_id: int = config.NORAD_ID,
    start: datetime = config.START_DATE,
    end: datetime = config.END_DATE,
    out_path: Path | None = None,
) -> Path:
    """
    Download historical TLEs for one satellite from Space-Track.org.

    The `spacetrack` library automatically respects Space-Track's rate
    limits (<30 requests/min, <300/hour), so we don't have to.

    Returns the path to the saved .txt file (one TLE set per epoch).
    """
    # Imported here so the module still loads even if spacetrack isn't
    # installed yet, and so import errors are reported clearly.
    from spacetrack import SpaceTrackClient
    import spacetrack.operators as op

    if not config.SPACETRACK_USER or not config.SPACETRACK_PASS:
        raise RuntimeError(
            "Space-Track credentials missing. Create a .env file in the "
            "project root with SPACETRACK_USER and SPACETRACK_PASS."
        )

    if out_path is None:
        out_path = config.RAW_DIR / f"tle_history_{norad_id}.txt"

    print(f"[historical] Connecting to Space-Track as {config.SPACETRACK_USER} ...")
    client = SpaceTrackClient(config.SPACETRACK_USER, config.SPACETRACK_PASS)

    # Inclusive date range on the TLE 'epoch' field
    date_range = op.inclusive_range(start, end)

    print(f"[historical] Requesting TLEs for NORAD {norad_id} "
          f"from {start.date()} to {end.date()} ...")
    lines = client.gp_history(
        norad_cat_id=norad_id,
        epoch=date_range,
        orderby="epoch asc",
        format="tle",
        iter_lines=True,
    )

    count = 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(line + "\n")
            count += 1

    if count == 0:
        print("[historical] WARNING: no data returned. Check the date range "
              "or NORAD ID.")
    else:
        # Two lines per TLE record
        print(f"[historical] Saved ~{count // 2} TLE records -> {out_path}")

    return out_path


# ------------------------------------------------------------
# 2. READ A TLE FILE BACK INTO MEMORY
# ------------------------------------------------------------
def load_tles_from_file(path: Path) -> List[TLE]:
    """
    Parse a saved TLE file into a list of (name, line1, line2) tuples.

    Handles both formats:
        - 2-line sets: line1 + line2 (no name)  -> name filled from config
        - 3-line sets: name line + line1 + line2
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TLE file not found: {path}")

    # Keep only non-empty, stripped lines
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]

    records: List[TLE] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # A TLE data line starts with "1 " (line1) or "2 " (line2).
        # A name line does not.
        if line.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            # 2-line set with no name
            records.append((config.SATELLITE_NAME, line, lines[i + 1]))
            i += 2
        elif (not line.startswith("1 ") and not line.startswith("2 ")
              and i + 2 < len(lines)
              and lines[i + 1].startswith("1 ")
              and lines[i + 2].startswith("2 ")):
            # 3-line set: name + line1 + line2
            records.append((line, lines[i + 1], lines[i + 2]))
            i += 3
        else:
            # Unexpected line -> skip it safely
            i += 1

    if not records:
        print(f"[load] WARNING: no valid TLE records parsed from {path}")
    else:
        print(f"[load] Parsed {len(records)} TLE record(s) from {path.name}")

    return records


# ------------------------------------------------------------
# Run directly: download historical TLEs and print a sample
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TLE Data Loader (Space-Track historical)")
    print("=" * 60)

    if not (config.SPACETRACK_USER and config.SPACETRACK_PASS):
        print("[historical] ERROR: no Space-Track credentials found.")
        print("             Create a .env file in the project root with:")
        print("               SPACETRACK_USER=your_email@example.com")
        print("               SPACETRACK_PASS=your_password")
    else:
        try:
            hist_file = download_historical_tles()
            records = load_tles_from_file(hist_file)
            if records:
                name, l1, l2 = records[0]
                print(f"\nSample TLE ({name}):")
                print(f"  {l1}")
                print(f"  {l2}")
        except RuntimeError as exc:
            print(f"[historical] ERROR: {exc}")

    print("\nDone.")
