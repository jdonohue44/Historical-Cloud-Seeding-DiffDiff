import time
from pathlib import Path
import requests
import zipfile
import io

BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/monthly"
OUT_DIR = Path("data/input/prism_monthly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for year in range(2000, 2026):
    for month in range(1, 13):

        yyyymm = f"{year}{month:02d}"
        zip_name = f"prism_ppt_us_25m_{yyyymm}.zip"
        url = f"{BASE_URL}/{year}/{zip_name}"

        try:
            out_file = OUT_DIR / f"prism_ppt_us_25m_{yyyymm}.tif"

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