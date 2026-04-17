"""
Build Panel from ERA5 NetCDF
============================
Reads era5_seed_control_neighbor_monthly_2000_2025.nc and outputs
cloud_seeding_monthly_panel.csv in the format diff_diff.py expects.

Key decisions:
- tp (total precipitation) is used as the outcome: large-scale + convective precip.
  ERA5 stores it in meters; we convert to mm (*1000).
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

# ERA5 tp in meters → convert to mm
tp_seed = ds["tp_seed_site"].values * 1000.0          # (site, time)
tp_ctrl = ds["tp_control_site"].values * 1000.0       # (site, control, time)
seeded_mask = ds["seeded_month_mask"].values           # (site, time)  0/1

site_lats = ds["site_latitude"].values                 # (site,)
site_lons = ds["site_longitude"].values                # (site,)

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
