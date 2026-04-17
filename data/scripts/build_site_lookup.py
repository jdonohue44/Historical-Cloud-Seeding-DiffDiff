"""
Build Site Lookup Table
=======================
Reconstructs the site_id → (state, lat, lon, project_name) mapping.

How the NC site indices are assigned (from the paper Methods):
  1. All cloud seeding records were geocoded (Target + State → lat/lon via Google Maps API).
  2. Lat/lon truncated to 2 decimal places → 139 unique sites.
  3. Sites are ordered alphabetically by state, then ascending latitude within state.

State blocks (verified against Table 1 known anchors):
  california:    0 –  28  (29 sites)
  colorado:     29 –  43  (15 sites)
  idaho:        44 –  57  (14 sites)
  kansas:       58 –  59  ( 2 sites)
  montana:      60 –  60  ( 1 site)
  nevada:       61 –  73  (13 sites)
  north dakota: 74 –  81  ( 8 sites)
  oklahoma:     82 –  82  ( 1 site)
  oregon:       83 –  83  ( 1 site)
  south dakota: 84 –  84  ( 1 site)
  texas:        85 – 117  (33 sites)
  utah:        118 – 132  (15 sites)
  wyoming:     133 – 138  ( 6 sites)

Named sites from Table 1 of the paper (exactly verified against NC lat/lon):
  33: Eastern San Juan Mountains Cloud Seeding Program (CO)
  40: Central Colorado Program (CO)
  18: Upper Tuolumne River Weather Modification Project (CA)
  21: Mokelumne / Lake Almanor (CA)
 115: PGCD Weather Modification Program (TX)
 125: Alta/Snowbird Ski Resort Cloud Seeding Program (UT)
   6: Kern River Cloud Seeding Program (CA)
  12: Kings River Cloud Seeding Program (CA)
  14: Mokelumne Cloud Seeding Program (CA)
  32: Telluride / Upper San Miguel–San Juan Basin Cloud Seeding Program (CO)

Output: data/input/site_lookup.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
NC_PATH = ROOT / "data" / "input" / "era5_seed_control_neighbor_monthly_2000_2025.nc"
OUT_PATH = ROOT / "data" / "input" / "site_lookup.csv"

ds = xr.open_dataset(NC_PATH)
lats = ds["site_latitude"].values
lons_raw = ds["site_longitude"].values
lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

# State block boundaries (start, end inclusive)
STATE_BLOCKS = [
    ( 0,  28, "california"),
    (29,  43, "colorado"),
    (44,  57, "idaho"),
    (58,  59, "kansas"),
    (60,  60, "montana"),
    (61,  73, "nevada"),
    (74,  81, "north_dakota"),
    (82,  82, "oklahoma"),
    (83,  83, "oregon"),
    (84,  84, "south_dakota"),
    (85, 117, "texas"),
    (118, 132, "utah"),
    (133, 138, "wyoming"),
]

# Project names from Table 1 (verified against NC lat/lon)
KNOWN_NAMES = {
    6:   "Kern River Cloud Seeding Program",
    12:  "Kings River Cloud Seeding Program",
    14:  "Mokelumne Cloud Seeding Program",
    18:  "Upper Tuolumne River Weather Modification Project",
    21:  "Mokelumne / Lake Almanor",
    32:  "Telluride / Upper San Miguel–San Juan Basin",
    33:  "Eastern San Juan Mountains Cloud Seeding Program",
    40:  "Central Colorado Program",
    115: "PGCD Weather Modification Program",
    125: "Alta/Snowbird Ski Resort Cloud Seeding Program",
}

rows = []
for start, end, state in STATE_BLOCKS:
    state_num = 0
    for i in range(start, end + 1):
        name = KNOWN_NAMES.get(i, "")
        rows.append({
            "site_idx":     i,
            "site_id":      f"site_{i:03d}",
            "state":        state,
            "lat":          round(float(lats[i]), 2),
            "lon":          round(float(lons[i]), 2),
            "project_name": name,
        })

lookup = pd.DataFrame(rows)
lookup.to_csv(OUT_PATH, index=False)
print(f"Saved {len(lookup)} sites → {OUT_PATH}")
print()
print(lookup[lookup["project_name"] != ""][["site_id", "state", "lat", "lon", "project_name"]].to_string(index=False))
