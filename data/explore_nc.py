"""
Quick exploration script for era5_seed_control_neighbor_monthly_2000_2025.nc

Run from the repo root:
  python data/explore_nc.py
  python data/explore_nc.py --site 125
  python data/explore_nc.py --site 33 --var lsp
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
NC_PATH = ROOT / "data" / "input" / "era5_seed_control_neighbor_monthly_2000_2025.nc"
LOOKUP_PATH = ROOT / "data" / "input" / "site_lookup.csv"

PRECIP_VARS = ["tp", "cp", "lsp", "sf"]  # total, convective, large-scale, snowfall

parser = argparse.ArgumentParser()
parser.add_argument("--site", type=int, default=None, help="Site index (0-138)")
parser.add_argument("--var",  type=str, default="tp",  help="Precip variable: tp, cp, lsp, sf")
args = parser.parse_args()

ds = xr.open_dataset(NC_PATH)
lookup = pd.read_csv(LOOKUP_PATH) if LOOKUP_PATH.exists() else None

# ── 1. Dataset overview ───────────────────────────────────────────────────────
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(ds)
print()

# ── 2. Site-level summary ─────────────────────────────────────────────────────
lats = ds["site_latitude"].values
lons = np.where(ds["site_longitude"].values > 180,
                ds["site_longitude"].values - 360,
                ds["site_longitude"].values)
mask = ds["seeded_month_mask"].values  # (site, time)
seeded_months = mask.sum(axis=1)

print("=" * 70)
print("SITE SUMMARY (top 10 by seeded months)")
print("=" * 70)
top10 = np.argsort(seeded_months)[::-1][:10]
for i in top10:
    name = ""
    state = ""
    if lookup is not None:
        row = lookup[lookup["site_idx"] == i]
        if not row.empty:
            state = row.iloc[0]["state"]
            name  = row.iloc[0]["project_name"] if pd.notna(row.iloc[0]["project_name"]) else ""
    label = f"{name} ({state})" if name else state
    print(f"  site {i:>3}  lat={lats[i]:.2f}  lon={lons[i]:.2f}  "
          f"seeded_months={int(seeded_months[i]):>3}  {label}")
print()

# ── 3. Single-site deep dive ──────────────────────────────────────────────────
site_idx = args.site if args.site is not None else int(top10[0])
var = args.var if args.var in PRECIP_VARS else "tp"

seed_var  = f"{var}_seed_site"
ctrl_var  = f"{var}_control_site"

times = pd.DatetimeIndex(ds["time"].values)
seed_mm   = ds[seed_var].sel(site=site_idx).values * 1000   # m → mm
ctrl_mm   = ds[ctrl_var].sel(site=site_idx).values * 1000   # (control, time) m → mm
ctrl_mean = np.nanmean(ctrl_mm, axis=0)
seeded    = ds["seeded_month_mask"].sel(site=site_idx).values.astype(bool)

# Site info
name = state = ""
if lookup is not None:
    row = lookup[lookup["site_idx"] == site_idx]
    if not row.empty:
        state = row.iloc[0]["state"]
        name  = row.iloc[0]["project_name"] if pd.notna(row.iloc[0]["project_name"]) else ""

print("=" * 70)
print(f"SITE {site_idx}  |  var={var}  |  {name or 'unnamed'}  ({state})")
print(f"  lat={lats[site_idx]:.2f}  lon={lons[site_idx]:.2f}")
print("=" * 70)

# Available controls
n_valid_ctrl = (~np.isnan(ctrl_mm[:, 0])).sum()
print(f"  Control points: {n_valid_ctrl} / {ctrl_mm.shape[0]}")
print(f"  Seeded months : {seeded.sum()} / {len(seeded)}")
print()

# Monthly table (most recent 24 months with data)
valid = ~np.isnan(seed_mm) & ~np.isnan(ctrl_mean)
idx = np.where(valid)[0][-24:]
print(f"{'year_month':<12} {'seed_mm':>10} {'ctrl_mm':>10} {'gap_mm':>8} {'seeded':>7}")
print("-" * 52)
for i in idx:
    gap = seed_mm[i] - ctrl_mean[i]
    s   = "YES" if seeded[i] else "-"
    print(f"{str(times[i])[:7]:<12} {seed_mm[i]:>10.2f} {ctrl_mean[i]:>10.2f} {gap:>8.2f} {s:>7}")
print()

# Seeded vs not-seeded means
if seeded.sum() > 0 and (~seeded & valid).sum() > 0:
    print("  Mean seed precip  — not seeded: "
          f"{np.nanmean(seed_mm[~seeded & valid]):>7.2f} mm  |  "
          f"seeded: {np.nanmean(seed_mm[seeded & valid]):>7.2f} mm")
    print("  Mean ctrl precip  — not seeded: "
          f"{np.nanmean(ctrl_mean[~seeded & valid]):>7.2f} mm  |  "
          f"seeded: {np.nanmean(ctrl_mean[seeded & valid]):>7.2f} mm")
    gap_ns = np.nanmean((seed_mm - ctrl_mean)[~seeded & valid])
    gap_s  = np.nanmean((seed_mm - ctrl_mean)[seeded  & valid])
    print(f"  Mean gap (seed–ctrl) — not seeded: {gap_ns:>+7.2f} mm  |  "
          f"seeded: {gap_s:>+7.2f} mm  |  diff: {gap_s - gap_ns:>+7.2f} mm")
print()
print("Tip: re-run with  --site <idx>  --var <tp|cp|lsp|sf>")
