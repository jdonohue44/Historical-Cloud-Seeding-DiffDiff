"""
Build Panel from PRISM Monthly TIFs
===================================
Reads all data/input/prism_monthly/prism_ppt_us_25m_YYYYMM.tif files and
outputs cloud_seeding_monthly_panel_prism.csv in the format diff_diff.py
expects. Drop-in replacement for the ERA5 panel.

Key decisions:
- Target/control point coordinates are reused from the ERA5 NetCDF
  (site_latitude/longitude and control_latitude/longitude) so the
  PRISM-vs-ERA5 comparison isolates the precipitation source, not the
  sampling scheme.
- seeded_month_mask is reused verbatim from the NC as target_area_seeded.
- PRISM values are already monthly totals in mm (no unit conversion).
  NoData (-9999) is mapped to NaN.
- PRISM CRS is EPSG:4269 (NAD83 geographic, -180..180 longitude). Site
  coords in the NC are stored 0..360 (e.g. 242.08 for California); we
  convert to -180..180 before sampling but keep 0..360 in the output
  longitude column to match the ERA5 panel schema exactly.
- Sampling uses nearest-pixel (rasterio.sample) - matches the single
  grid-cell lookup the ERA5 panel used.
- Sites outside CONUS (AK, HI, offshore, cross-border) will return NoData
  and be dropped by dropna, same pattern as the ERA5 builder.

Output: data/input/cloud_seeding_monthly_panel_prism.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rasterio

ROOT = Path(__file__).resolve().parent.parent.parent
NC_PATH = ROOT / "data" / "input" / "era5_seed_control_neighbor_monthly_2000_2025.nc"
PRISM_DIR = ROOT / "data" / "input" / "prism_monthly"
LOOKUP_PATH = ROOT / "data" / "input" / "site_lookup.csv"
OUT_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel_prism.csv"


def to_180(lon):
    """Convert longitude from 0..360 to -180..180 (PRISM convention)."""
    return np.where(lon > 180, lon - 360, lon)


# --- Load site metadata + treatment mask from NC ---------------------------
print(f"Loading site metadata from {NC_PATH.name} ...")
ds = xr.open_dataset(NC_PATH)

n_sites = ds.sizes["site"]
n_ctrl  = ds.sizes["control"]
n_time  = ds.sizes["time"]
times   = pd.DatetimeIndex(ds["time"].values)

site_lats_out  = ds["site_latitude"].values                  # (site,)
site_lons_out  = ds["site_longitude"].values                 # (site,) 0..360, kept for output
site_lons_180  = to_180(site_lons_out)                       # (site,) for PRISM sampling
ctrl_lats      = ds["control_latitude"].values               # (site, control)
ctrl_lons_180  = to_180(ds["control_longitude"].values)      # (site, control) for sampling
seeded_mask    = ds["seeded_month_mask"].values              # (site, time) 0/1

# Flag sites whose control coords fall outside the western US (geocoder
# fallbacks to Guangzhou/Princeton etc.). See audit_era5_data.py section B.
# Any site with >=1 valid control outside the box is flagged; downstream
# analysis can filter on `bogus_control == False`.
WEST_LAT_MIN, WEST_LAT_MAX = 24.0, 50.0
WEST_LON_MIN, WEST_LON_MAX = -125.0, -95.0
_valid = ~(np.isnan(ctrl_lats) | np.isnan(ctrl_lons_180))
_in_west = (
    (ctrl_lats >= WEST_LAT_MIN) & (ctrl_lats <= WEST_LAT_MAX)
    & (ctrl_lons_180 >= WEST_LON_MIN) & (ctrl_lons_180 <= WEST_LON_MAX)
)
bogus_control_site = (_valid & ~_in_west).any(axis=1)  # (site,)
print(f"  Flagged {bogus_control_site.sum()} sites with bogus control coords.")

print(f"  {n_sites} sites, {n_ctrl} control slots per site, {n_time} months")
print(f"  Time range: {times[0].strftime('%Y-%m')} -> {times[-1].strftime('%Y-%m')}")

# --- Build flat point list (one entry per target or control slot) ----------
# We sample all 139 * (1 + 6) = 973 points in one shot per TIF.
# NaN-padded control slots keep NaN as their value.

points_xy = []     # (x=lon_-180..180, y=lat)
owner_site = []    # int site index
owner_ctrl = []    # -1 for target, 0..5 for control slot

for s in range(n_sites):
    points_xy.append((float(site_lons_180[s]), float(site_lats_out[s])))
    owner_site.append(s); owner_ctrl.append(-1)
    for c in range(n_ctrl):
        points_xy.append((float(ctrl_lons_180[s, c]), float(ctrl_lats[s, c])))
        owner_site.append(s); owner_ctrl.append(c)

n_pts = len(points_xy)
owner_site = np.array(owner_site)
owner_ctrl = np.array(owner_ctrl)
coords_arr = np.array(points_xy)
valid_pt = ~(np.isnan(coords_arr[:, 0]) | np.isnan(coords_arr[:, 1]))
sample_coords = [tuple(p) for p in coords_arr[valid_pt]]

print(f"  {n_pts} total points ({valid_pt.sum()} valid, "
      f"{(~valid_pt).sum()} NaN-padded control slots)")

# --- Discover PRISM nodata value (consistent across files) -----------------
with rasterio.open(next(PRISM_DIR.glob("prism_ppt_us_25m_*.tif"))) as probe:
    PRISM_NODATA = probe.nodata
    PRISM_CRS = probe.crs
print(f"  PRISM CRS: {PRISM_CRS}  NoData: {PRISM_NODATA}")

# --- Iterate TIFs, sample all points per month -----------------------------
tp_seed = np.full((n_sites, n_time), np.nan, dtype=np.float32)
tp_ctrl = np.full((n_sites, n_ctrl, n_time), np.nan, dtype=np.float32)

missing_files = []
for ti, t in enumerate(times):
    yyyymm = f"{t.year}{t.month:02d}"
    tif = PRISM_DIR / f"prism_ppt_us_25m_{yyyymm}.tif"
    if not tif.exists():
        missing_files.append(yyyymm)
        continue

    with rasterio.open(tif) as src:
        vals_valid = np.array(
            [v[0] for v in src.sample(sample_coords)], dtype=np.float32
        )

    vals = np.full(n_pts, np.nan, dtype=np.float32)
    vals[valid_pt] = vals_valid
    vals[vals == PRISM_NODATA] = np.nan

    is_target = owner_ctrl == -1
    tp_seed[owner_site[is_target], ti] = vals[is_target]
    is_ctrl = ~is_target
    tp_ctrl[owner_site[is_ctrl], owner_ctrl[is_ctrl], ti] = vals[is_ctrl]

    if (ti + 1) % 24 == 0 or ti == n_time - 1:
        print(f"  [{ti+1:3d}/{n_time}] sampled through {yyyymm}")

if missing_files:
    head = ", ".join(missing_files[:5])
    tail = " ..." if len(missing_files) > 5 else ""
    print(f"WARNING: {len(missing_files)} PRISM months missing: {head}{tail}")

# Mean across control points (nanmean ignores NaN padding)
with np.errstate(all="ignore"):
    tp_ctrl_mean = np.nanmean(tp_ctrl, axis=1)  # (site, time)

# --- Build long-format dataframe -------------------------------------------
print(f"Building panel: {n_sites} sites x {n_time} months = {n_sites*n_time:,} rows ...")

site_idx = np.repeat(np.arange(n_sites), n_time)
time_idx = np.tile(np.arange(n_time), n_sites)

df = pd.DataFrame({
    "site_id":                 [f"site_{i:03d}" for i in site_idx],
    "site_latitude":           site_lats_out[site_idx],
    "site_longitude":          site_lons_out[site_idx],   # 0..360 to match ERA5 panel
    "year_month":              times[time_idx].strftime("%Y-%m"),
    "year":                    times[time_idx].year,
    "month":                   times[time_idx].month,
    "target_area_precip_mm":   tp_seed[site_idx, time_idx],
    "control_area_precip_mm":  tp_ctrl_mean[site_idx, time_idx],
    "target_area_seeded":      seeded_mask[site_idx, time_idx].astype(bool),
    "bogus_control":           bogus_control_site[site_idx],
})

before = len(df)
df = df.dropna(subset=["target_area_precip_mm", "control_area_precip_mm"])
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped:,} rows with NaN precipitation "
          f"(non-CONUS sites or missing month files).")

print(f"Final panel: {df['site_id'].nunique()} sites, {len(df):,} rows")
print(f"Seeded obs: {df['target_area_seeded'].sum():,} / {len(df):,} "
      f"({df['target_area_seeded'].mean()*100:.1f}%)")

if LOOKUP_PATH.exists():
    lookup = pd.read_csv(LOOKUP_PATH)[["site_id", "state", "project_name"]]
    df = df.merge(lookup, on="site_id", how="left")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
print("\nSample rows:")
print(df.head(6).to_string(index=False))
