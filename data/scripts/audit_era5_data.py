"""
ERA5 Panel + NC Integrity Audit
===============================
Scans era5_seed_control_neighbor_monthly_2000_2025.nc and the derived
cloud_seeding_monthly_panel.csv for data-quality issues that PRISM
surfaced during panel reconstruction.

Sections:
  A. Target site coordinates       - bounds, duplicates, lookup mismatch
  B. Control point coordinates     - bounds, cross-site duplicates, distance
  C. Neighbor point coordinates    - bounds sanity
  D. Seeded-month mask             - 0-seeded sites, NaN, treatment density
  E. Precipitation value sanity    - ranges, units plausibility
  F. Panel consistency             - rows, NaN patterns, 2023+ gap
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
NC_PATH = ROOT / "data" / "input" / "era5_seed_control_neighbor_monthly_2000_2025.nc"
PANEL_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"
LOOKUP_PATH = ROOT / "data" / "input" / "site_lookup.csv"

# Western US bounding box used as a "does this point make sense?" test.
# Covers CA/OR/WA/NV/AZ/UT/ID/MT/WY/CO/NM/ND/SD/NE/TX (western third) and the
# practical domain of US cloud seeding programs. Wider than strict "west of
# the Rockies" to avoid false positives from Great Plains projects.
WEST_LAT_MIN, WEST_LAT_MAX = 24.0, 50.0
WEST_LON_MIN, WEST_LON_MAX = -125.0, -95.0


def to_180(lon):
    """Normalize longitude to -180..180 regardless of input convention."""
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180, lon - 360, lon)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def in_west(lat, lon180):
    return (
        (lat >= WEST_LAT_MIN) & (lat <= WEST_LAT_MAX)
        & (lon180 >= WEST_LON_MIN) & (lon180 <= WEST_LON_MAX)
    )


# --- Load --------------------------------------------------------------------
ds = xr.open_dataset(NC_PATH)
panel = pd.read_csv(PANEL_PATH)
lookup = pd.read_csv(LOOKUP_PATH) if LOOKUP_PATH.exists() else None

n_sites = ds.sizes["site"]
n_ctrl  = ds.sizes["control"]
n_nbr   = ds.sizes["neighbor"]
n_time  = ds.sizes["time"]
times   = pd.DatetimeIndex(ds["time"].values)

slat = ds["site_latitude"].values
slon180 = to_180(ds["site_longitude"].values)
clat = ds["control_latitude"].values                   # (site, control)
clon180 = to_180(ds["control_longitude"].values)
nlat = ds["neighbor_latitude"].values                  # (site, neighbor)
nlon180 = to_180(ds["neighbor_longitude"].values)
seeded_mask = ds["seeded_month_mask"].values           # (site, time)
tp_seed = ds["tp_seed_site"].values                    # (site, time)
tp_ctrl = ds["tp_control_site"].values                 # (site, control, time)

site_ids = [f"site_{i:03d}" for i in range(n_sites)]

print(f"NC     : {NC_PATH.name}")
print(f"Panel  : {PANEL_PATH.name} ({len(panel):,} rows, "
      f"{panel['site_id'].nunique()} sites)")
print(f"Dims   : {n_sites} sites | {n_ctrl} control slots | "
      f"{n_nbr} neighbor slots | {n_time} months")
print(f"Period : {times[0].strftime('%Y-%m')} -> {times[-1].strftime('%Y-%m')}")
print(f"West-US box used for coordinate audit: "
      f"lat [{WEST_LAT_MIN},{WEST_LAT_MAX}], lon [{WEST_LON_MIN},{WEST_LON_MAX}]")


# ============================================================================
section("A. TARGET SITE COORDINATES")
# ============================================================================

target_in_west = in_west(slat, slon180)
off_target = np.where(~target_in_west)[0]
print(f"Target sites outside western-US box: {len(off_target)}")
for s in off_target:
    print(f"  - {site_ids[s]}: lat={slat[s]:.3f}, lon={slon180[s]:.3f}")

# Duplicate target coords
coords_df = pd.DataFrame({
    "site_id": site_ids, "lat": slat, "lon": slon180
})
dup_targets = coords_df[coords_df.duplicated(subset=["lat", "lon"], keep=False)]
if len(dup_targets):
    print(f"\nDuplicate target coordinates: {len(dup_targets)} sites share coords")
    for (lat, lon), grp in dup_targets.groupby(["lat", "lon"]):
        print(f"  ({lat:.3f}, {lon:.3f}) shared by: "
              f"{', '.join(grp['site_id'].tolist())}")
else:
    print("\nAll target coords unique: OK")

# Target coords vs site_lookup (should match)
if lookup is not None:
    lk = lookup.set_index("site_id")[["lat", "lon"]]
    # Lookup is in -180..180; compare on that basis
    mis = []
    for i, sid in enumerate(site_ids):
        if sid in lk.index:
            dlat = abs(lk.at[sid, "lat"] - slat[i])
            dlon = abs(lk.at[sid, "lon"] - slon180[i])
            if dlat > 0.01 or dlon > 0.01:
                mis.append((sid, lk.at[sid, "lat"], slat[i],
                            lk.at[sid, "lon"], slon180[i]))
    if mis:
        print(f"\nTarget coords disagree with site_lookup.csv for {len(mis)} sites:")
        for sid, llat, nclat, llon, nclon in mis[:10]:
            print(f"  - {sid}: lookup=({llat:.3f},{llon:.3f}) vs "
                  f"NC=({nclat:.3f},{nclon:.3f})")
    else:
        print("Target coords match site_lookup.csv: OK")


# ============================================================================
section("B. CONTROL POINT COORDINATES")
# ============================================================================

valid_c = ~(np.isnan(clat) | np.isnan(clon180))
print(f"Control slots total: {clat.size} ({valid_c.sum()} valid, "
      f"{(~valid_c).sum()} NaN-padded)")
print(f"Valid controls per site: min={valid_c.sum(axis=1).min()}, "
      f"median={int(np.median(valid_c.sum(axis=1)))}, "
      f"max={valid_c.sum(axis=1).max()}")

# Controls outside western US
out_of_box = []
for s in range(n_sites):
    for c in range(n_ctrl):
        if valid_c[s, c] and not in_west(clat[s, c], clon180[s, c]):
            out_of_box.append((site_ids[s], c, clat[s, c], clon180[s, c]))
print(f"\nControl points outside western-US box: {len(out_of_box)}")
for sid, c, lat, lon in out_of_box[:30]:
    print(f"  - {sid}[c{c}]: lat={lat:.3f}, lon={lon:.3f}")
if len(out_of_box) > 30:
    print(f"  ... and {len(out_of_box) - 30} more")

# Duplicate control points across sites (one coord reused by many sites)
ctrl_rows = []
for s in range(n_sites):
    for c in range(n_ctrl):
        if valid_c[s, c]:
            ctrl_rows.append({
                "site_id": site_ids[s], "c": c,
                "lat": round(float(clat[s, c]), 4),
                "lon": round(float(clon180[s, c]), 4),
            })
ctrl_df = pd.DataFrame(ctrl_rows)
shared = (
    ctrl_df.groupby(["lat", "lon"])["site_id"]
    .agg(lambda x: sorted(set(x)))
    .reset_index()
    .assign(n_sites=lambda d: d["site_id"].str.len())
    .sort_values("n_sites", ascending=False)
)
multi = shared[shared["n_sites"] > 1]
print(f"\nControl coords shared across multiple sites: {len(multi)} distinct coord(s)")
for _, r in multi.head(10).iterrows():
    preview = ", ".join(r["site_id"][:5])
    tail = f" (+{len(r['site_id']) - 5} more)" if len(r["site_id"]) > 5 else ""
    print(f"  ({r['lat']:.3f}, {r['lon']:.3f}) used by {r['n_sites']} sites: "
          f"{preview}{tail}")

# Control-to-target distance distribution (flag extreme outliers)
dists = []
for s in range(n_sites):
    for c in range(n_ctrl):
        if valid_c[s, c]:
            d = haversine_km(slat[s], slon180[s], clat[s, c], clon180[s, c])
            dists.append((site_ids[s], c, d))
dist_arr = np.array([d for _, _, d in dists])
print(f"\nTarget->control distance (km): "
      f"min={dist_arr.min():.1f}, median={np.median(dist_arr):.1f}, "
      f"p95={np.percentile(dist_arr, 95):.1f}, max={dist_arr.max():.1f}")
# Anything >1000 km is almost certainly wrong for a local control
far = [(sid, c, d) for sid, c, d in dists if d > 1000]
print(f"Controls >1000 km from their target: {len(far)}")
for sid, c, d in far[:15]:
    print(f"  - {sid}[c{c}]: {d:.0f} km")
# Controls at identical coords to target (DiD would be ~0 mechanically)
same = [(sid, c, d) for sid, c, d in dists if d < 1.0]
if same:
    print(f"\nControls <1 km from their target (suspiciously close): {len(same)}")
    for sid, c, d in same[:10]:
        print(f"  - {sid}[c{c}]: {d:.2f} km")


# ============================================================================
section("C. NEIGHBOR POINT COORDINATES")
# ============================================================================

valid_n = ~(np.isnan(nlat) | np.isnan(nlon180))
print(f"Neighbor slots total: {nlat.size} ({valid_n.sum()} valid, "
      f"{(~valid_n).sum()} NaN-padded)")
out_nbr = 0
example = []
for s in range(n_sites):
    for k in range(n_nbr):
        if valid_n[s, k] and not in_west(nlat[s, k], nlon180[s, k]):
            out_nbr += 1
            if len(example) < 10:
                example.append((site_ids[s], k, nlat[s, k], nlon180[s, k]))
print(f"Neighbor points outside western-US box: {out_nbr}")
for sid, k, lat, lon in example:
    print(f"  - {sid}[n{k}]: lat={lat:.3f}, lon={lon:.3f}")
if out_nbr > len(example):
    print(f"  ... and {out_nbr - len(example)} more")


# ============================================================================
section("D. SEEDED-MONTH MASK")
# ============================================================================

mask_nan = np.isnan(seeded_mask).sum()
print(f"NaN entries in seeded_month_mask: {mask_nan}")
print(f"Unique values: {np.unique(seeded_mask[~np.isnan(seeded_mask)])}")

per_site_seeded = np.nansum(seeded_mask, axis=1).astype(int)
zero_seed = np.where(per_site_seeded == 0)[0]
all_seed = np.where(per_site_seeded == n_time)[0]
print(f"\nSites with ZERO seeded months: {len(zero_seed)}")
for s in zero_seed:
    print(f"  - {site_ids[s]}: 0 seeded / {n_time} months")
print(f"Sites with ALL months seeded: {len(all_seed)}")
for s in all_seed:
    print(f"  - {site_ids[s]}: {n_time} / {n_time} months")
print(f"\nSeeded months per site: min={per_site_seeded.min()}, "
      f"median={int(np.median(per_site_seeded))}, "
      f"mean={per_site_seeded.mean():.1f}, max={per_site_seeded.max()}")

# Compare to panel after dropna -- which sites lose their seeded obs?
panel_seeded = panel.groupby("site_id")["target_area_seeded"].sum()
diff_vs_panel = []
for i, sid in enumerate(site_ids):
    nc_count = int(per_site_seeded[i])
    panel_count = int(panel_seeded.get(sid, 0))
    if nc_count > 0 and panel_count == 0:
        diff_vs_panel.append((sid, nc_count))
if diff_vs_panel:
    print(f"\nSites with seeded months in NC but NONE in panel "
          f"(dropped by ERA5 2023+ gap):")
    for sid, n in diff_vs_panel:
        print(f"  - {sid}: {n} seeded months in NC, 0 survive into panel")


# ============================================================================
section("E. PRECIPITATION VALUE SANITY")
# ============================================================================

# ERA5 tp is typically stored in meters of water equivalent over the
# accumulation window. A monthly average of hourly tp in m/hr would yield
# tiny numbers; a monthly sum in m would yield ~0.01-0.2. Multiplying by
# 1000 either gives mm/hr (tiny) or mm/month (correct). Expected mm/month
# in the western US: desert CA ~5-25, wet PNW/Sierras 100-300.
tp_seed_mm = tp_seed * 1000.0
tp_ctrl_mm = tp_ctrl * 1000.0

finite = tp_seed_mm[~np.isnan(tp_seed_mm)]
print(f"Target tp * 1000 (mm?) stats across all valid (site,time):")
print(f"  min={finite.min():.4f}, median={np.median(finite):.4f}, "
      f"mean={finite.mean():.4f}, p95={np.percentile(finite,95):.4f}, "
      f"max={finite.max():.4f}")
print(f"  → typical western-US monthly precip is 5-200 mm. "
      f"{'UNITS LOOK OK.' if np.median(finite) > 2 else 'UNITS LOOK WRONG.'}")

finite_c = tp_ctrl_mm[~np.isnan(tp_ctrl_mm)]
print(f"Control tp * 1000 (mm?) stats:")
print(f"  min={finite_c.min():.4f}, median={np.median(finite_c):.4f}, "
      f"mean={finite_c.mean():.4f}, max={finite_c.max():.4f}")

# Negative or absurdly high values
neg_t = (tp_seed_mm < 0).sum()
huge_t = (tp_seed_mm > 2000).sum()
print(f"\nPhysically-implausible target values: "
      f"{neg_t} negative, {huge_t} > 2000 mm/month")

# Per-site mean target precip — flag extreme outliers (too dry / too wet)
site_mean = np.nanmean(tp_seed_mm, axis=1)
bot = np.argsort(site_mean)[:5]
top = np.argsort(site_mean)[-5:]
print("\nDriest 5 sites by mean target precip (mm/month):")
for s in bot:
    print(f"  - {site_ids[s]}: {site_mean[s]:.3f} "
          f"(lat={slat[s]:.2f}, lon={slon180[s]:.2f})")
print("Wettest 5 sites:")
for s in top:
    print(f"  - {site_ids[s]}: {site_mean[s]:.3f} "
          f"(lat={slat[s]:.2f}, lon={slon180[s]:.2f})")


# ============================================================================
section("F. PANEL CONSISTENCY")
# ============================================================================

expected_rows = n_sites * n_time
dropped_rows = expected_rows - len(panel)
print(f"Expected rows (sites * months): {expected_rows:,}")
print(f"Actual panel rows:               {len(panel):,}")
print(f"Dropped:                         {dropped_rows:,} "
      f"(from dropna on precip NaN)")

# How many sites lost ALL their rows? none in ERA5 panel (we confirmed)
missing_in_panel = set(site_ids) - set(panel["site_id"].unique())
if missing_in_panel:
    print(f"Sites present in NC but missing from panel entirely: "
          f"{sorted(missing_in_panel)}")
else:
    print("Every NC site appears in the panel: OK")

# Count rows per site (exposes the 2023+ gap)
rows_by_site = panel.groupby("site_id").size()
short = rows_by_site[rows_by_site < n_time]
print(f"\nSites with fewer than {n_time} panel rows: {len(short)}")
print(f"  Min rows per site: {rows_by_site.min()} ({short.idxmin() if len(short) else 'n/a'})")
print(f"  Expected from ERA5 2023+ gap: rows should stop at 2022-12 for affected sites")

# Show the year_month range per short site (confirm it's the 2022 cutoff pattern)
if len(short) > 0:
    bad_sites = short.index.tolist()[:5]
    print(f"  Date range for {len(bad_sites)} sample short sites:")
    for sid in bad_sites:
        sub = panel[panel["site_id"] == sid]["year_month"]
        print(f"    - {sid}: {sub.min()} -> {sub.max()} ({len(sub)} rows)")


print()
print("=" * 78)
print("  AUDIT COMPLETE")
print("=" * 78)
