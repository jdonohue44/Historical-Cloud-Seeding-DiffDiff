from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
PRISM_PATH = ROOT / "data" / "input" / "prism_ppt_us_25m_2025.tif"

with rasterio.open(PRISM_PATH) as src:

    print("\n=== METADATA ===")
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Grid shape:", (src.height, src.width))
    print("Resolution:", src.res)
    print("Band count:", src.count)
    print("NoData:", src.nodata)

    data = src.read(1)

nodata = -9999
valid = data[data != nodata]

print("\n=== VALID CELLS ===")
print("Total cells:", data.size)
print("Valid cells:", len(valid))
print("Missing:", data.size-len(valid))

print("\n=== PRECIP STATS (mm) ===")
print(pd.Series(valid).describe(
    percentiles=[.01,.05,.25,.5,.75,.95,.99]
))

# -----------------------------
# FIGURES
# -----------------------------

fig, axes = plt.subplots(2,2, figsize=(14,10))

# (1) Spatial map
im=axes[0,0].imshow(
    np.where(data==nodata,np.nan,data),
    origin="upper"
)
axes[0,0].set_title("PRISM Annual Precipitation")
plt.colorbar(im, ax=axes[0,0])

# (2) Histogram
axes[0,1].hist(valid, bins=100)
axes[0,1].set_title("Distribution of Precipitation")
axes[0,1].set_xlabel("mm")

# (3) Log histogram (helps inspect tails)
axes[1,0].hist(valid, bins=100, log=True)
axes[1,0].set_title("Distribution (log count)")
axes[1,0].set_xlabel("mm")

# (4) Latitudinal precipitation gradient
# mean precip by row (north-south gradient)
row_means=np.nanmean(
    np.where(data==nodata,np.nan,data),
    axis=1
)
axes[1,1].plot(row_means)
axes[1,1].set_title("North-South Mean Precip Gradient")
axes[1,1].set_ylabel("Mean mm")

plt.tight_layout()
plt.show()