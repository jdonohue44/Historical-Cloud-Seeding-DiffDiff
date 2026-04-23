"""
Download PRISM monthly precipitation TIFFs.

Set `PRISM_RESOLUTION` to either:
- "4km"  -> downloads the 2.5 arc-minute product
- "800m" -> downloads the 30 arc-second product
"""

import time
from pathlib import Path
import requests
import zipfile
import io

PRISM_RESOLUTION = "4km"

RESOLUTION_CONFIG = {
    "4km": {
        "base_url": "https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/monthly",
        "zip_prefix": "prism_ppt_us_25m",
        "out_dir": Path("data/input/prism_monthly"),
    },
    "800m": {
        "base_url": "https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/monthly",
        "zip_prefix": "prism_ppt_us_30s",
        "out_dir": Path("data/input/prism_monthly_800m"),
    },
}

if PRISM_RESOLUTION not in RESOLUTION_CONFIG:
    raise ValueError(
        f"Unsupported PRISM_RESOLUTION={PRISM_RESOLUTION!r}. "
        f"Choose one of: {', '.join(RESOLUTION_CONFIG)}"
    )

config = RESOLUTION_CONFIG[PRISM_RESOLUTION]
BASE_URL = config["base_url"]
ZIP_PREFIX = config["zip_prefix"]
OUT_DIR = config["out_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {PRISM_RESOLUTION}")
print(f"Download URL root: {BASE_URL}")
print(f"Output directory: {OUT_DIR}")

for year in range(2000, 2026):
    for month in range(1, 13):

        yyyymm = f"{year}{month:02d}"
        zip_name = f"{ZIP_PREFIX}_{yyyymm}.zip"
        url = f"{BASE_URL}/{year}/{zip_name}"

        try:
            out_file = OUT_DIR / f"{ZIP_PREFIX}_{yyyymm}.tif"

            if out_file.exists():
                print(f"Skip {yyyymm}")
                continue

            print(f"Downloading {yyyymm}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()

            z = zipfile.ZipFile(io.BytesIO(r.content))

            for f in z.namelist():
                if f.endswith(".tif"):
                    with open(out_file, "wb") as out:
                        out.write(z.read(f))

            time.sleep(0.2)  # polite throttle

        except Exception as e:
            print(f"Missing {yyyymm}: {e}")