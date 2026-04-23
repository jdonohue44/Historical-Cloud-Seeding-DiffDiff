"""
Build Panel from ERA5 NetCDF
============================
Reads era5_seed_control_neighbor_monthly_2000_2025.nc and outputs
cloud_seeding_monthly_panel.csv in the format diff_diff.py expects.

Key decisions:
- tp (total precipitation) is used as the outcome: large-scale + convective precip.
  ERA5 monthly `tp` is stored as a mean daily rate in m/day (the monthly mean
  of daily accumulations). We convert to mm/month by multiplying by 1000
  (m -> mm) and by the number of days in that month.
- Control precip = nanmean across the up to 6 control points for each site.
  Sites with fewer than 6 control points have NaN padding in the trailing slots.
- seeded_month_mask (0/1) becomes the target_area_seeded column.
- Sites are identified by integer index (0-138) since the NC has no site names.
  To map to named sites, match by site_latitude/site_longitude after export.

Output: data/input/cloud_seeding_monthly_panel.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
NC_PATH = ROOT / "data" / "input" / "era5_seed_control_neighbor_monthly_2000_2025.nc"
LOOKUP_PATH = ROOT / "data" / "input" / "site_lookup.csv"
OUT_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"

# ---------------------------------------------------------------------------
print(f"Loading {NC_PATH.name} ...")
ds = xr.open_dataset(NC_PATH)

n_sites = ds.sizes["site"]
n_time  = ds.sizes["time"]
times   = pd.DatetimeIndex(ds["time"].values)

# ERA5 monthly tp is mean daily rate in m/day -> mm/month via
# (m/day) * 1000 (mm/m) * days_in_month.
days_in_month = times.days_in_month.values.astype(np.float32)  # (time,)

tp_seed = (
    ds["tp_seed_site"].values * 1000.0 * days_in_month[np.newaxis, :]
)                                                        # (site, time)
tp_ctrl = (
    ds["tp_control_site"].values * 1000.0
    * days_in_month[np.newaxis, np.newaxis, :]
)                                                        # (site, control, time)
seeded_mask = ds["seeded_month_mask"].values             # (site, time)  0/1

site_lats = ds["site_latitude"].values                 # (site,)
site_lons = ds["site_longitude"].values                # (site,)

# Flag sites whose control coords fall outside the western US (geocoder
# fallbacks to Guangzhou/Princeton etc.). See audit_era5_data.py section B.
# Any site with >=1 valid control outside the box is flagged; downstream
# analysis can filter on `bogus_control == False`.
WEST_LAT_MIN, WEST_LAT_MAX = 24.0, 50.0
WEST_LON_MIN, WEST_LON_MAX = -125.0, -95.0
ctrl_lat = ds["control_latitude"].values
ctrl_lon = ds["control_longitude"].values
ctrl_lon180 = np.where(ctrl_lon > 180, ctrl_lon - 360, ctrl_lon)
_valid = ~(np.isnan(ctrl_lat) | np.isnan(ctrl_lon180))
_in_west = (
    (ctrl_lat >= WEST_LAT_MIN) & (ctrl_lat <= WEST_LAT_MAX)
    & (ctrl_lon180 >= WEST_LON_MIN) & (ctrl_lon180 <= WEST_LON_MAX)
)
bogus_control_site = (_valid & ~_in_west).any(axis=1)  # (site,)
print(f"  Flagged {bogus_control_site.sum()} sites with bogus control coords.")

# Mean across control points (nanmean ignores NaN padding)
tp_ctrl_mean = np.nanmean(tp_ctrl, axis=1)             # (site, time)

# ---------------------------------------------------------------------------
# Build long-format dataframe
print(f"Building panel: {n_sites} sites × {n_time} months = {n_sites * n_time:,} rows ...")

site_idx  = np.repeat(np.arange(n_sites), n_time)
time_idx  = np.tile(np.arange(n_time), n_sites)

df = pd.DataFrame({
    "site_id":                 [f"site_{i:03d}" for i in site_idx],
    "site_latitude":           site_lats[site_idx],
    "site_longitude":          site_lons[site_idx],
    "year_month":              times[time_idx].strftime("%Y-%m"),
    "year":                    times[time_idx].year,
    "month":                   times[time_idx].month,
    "target_area_precip_mm":   tp_seed[site_idx, time_idx],
    "control_area_precip_mm":  tp_ctrl_mean[site_idx, time_idx],
    "target_area_seeded":      seeded_mask[site_idx, time_idx].astype(bool),
    "bogus_control":           bogus_control_site[site_idx],
})

# Drop rows where either precip is NaN (ERA5 gap for 2023-2025 if not present)
before = len(df)
df = df.dropna(subset=["target_area_precip_mm", "control_area_precip_mm"])
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped:,} rows with NaN precipitation (ERA5 data gap).")

print(f"Final panel: {df['site_id'].nunique()} sites, {len(df):,} rows")
print(f"Seeded obs: {df['target_area_seeded'].sum():,} / {len(df):,} "
      f"({df['target_area_seeded'].mean()*100:.1f}%)")

# Join state and project_name from lookup
if LOOKUP_PATH.exists():
    lookup = pd.read_csv(LOOKUP_PATH)[["site_id", "state", "project_name"]]
    df = df.merge(lookup, on="site_id", how="left")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print("\nSample rows:")
print(df.head(6).to_string(index=False))
