"""
DiD: Effect of Cloud Seeding on Precipitation
=========================================================

ESTIMATOR:
Within-site difference-in-differences, equal-weighted across sites.
The two "periods" are seeded vs unseeded months within the same site;
the "groups" are the target and control areas (baked into the outcome
as their precipitation gap).

For each site i:
    DiD_i = mean(target - control gap | seeded months at site i)
          - mean(target - control gap | unseeded months at site i)

Aggregate:
    ATT = mean(DiD_i) across sites    (each site weighted equally)

STANDARD ERRORS / CLUSTERING:
Each site contributes exactly one DiD_i to the aggregate.
  - Sites are the unit of observation in the aggregate.
  - SE = sd(DiD_i) / sqrt(n_sites). This is the standard error of
    the cross-site mean and allows arbitrary within-site correlation
    without modeling it explicitly.
  - Inference uses a t-distribution with (n_sites - 1) degrees of
    freedom, which is conservative for small numbers of sites.

INPUT:
 - data/input/cloud_seeding_monthly_panel.csv

OUTPUT:
  - Console: per-site seeded/unseeded month counts, site-level DiDs,
    and the equal-weighted aggregate ATT with clustered-by-site SE.
  - data/output/site_level_seeding_gaps.csv: per-site mean gap by seeding
    status and within-site DiD.
  - figures/site_precip_pageNN.png: paginated line charts of target vs
    control precipitation by site, with seeding months shaded.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# -- Load and prepare ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
df["seeded"] = df["target_area_seeded"].astype(int)
df["year_month"] = pd.to_datetime(df["year_month"])

n_sites = df["site_id"].nunique()
n_months = df["year_month"].nunique()
seeded_obs = df["seeded"].sum()

print(f"Panel: {n_sites} sites x {n_months} months = {len(df):,} obs")
print(f"Seeded obs: {seeded_obs:,} / {len(df):,}")
print()

# Sanity check: within-site DiD requires each site to have both seeded
# and unseeded months. Warn on any violations.
#
# KNOWN CAUSE OF 0-SEEDED SITES (documented):
# ERA5 precipitation in this dataset ends at Dec 2022 (276 months × 139 sites),
# but the raw NC seeded_month_mask has seeded months through 2025 for a subset
# of sites. For 10 sites, every seeded month falls within the 2023+ ERA5 gap,
# so build_panel_from_nc.py drops all their seeded rows. These sites are not
# never-treated controls, they are post-2022 programs outside ERA5 coverage.
# They contribute NaN to `within_site_did` and are excluded from the
# equal-weighted ATT below via `.dropna()`. There is no pooled TWFE here, so
# no risk of them being silently pooled as all-zero seeded controls.
per_site_counts = df.groupby("site_id")["seeded"].agg(
    n_seeded="sum", n_total="count"
)
per_site_counts["n_unseeded"] = per_site_counts["n_total"] - per_site_counts["n_seeded"]
missing_seeded = per_site_counts[per_site_counts["n_seeded"] == 0]
missing_unseeded = per_site_counts[per_site_counts["n_unseeded"] == 0]
if len(missing_seeded) or len(missing_unseeded):
    print("WARNING: within-site DiD is undefined for the following site(s)")
    print("         (likely ERA5 2023+ coverage gap for sites with 0 seeded months):")
    for sid in missing_seeded.index:
        print(f"  - {sid}: 0 seeded months  → EXCLUDED from aggregate ATT")
    for sid in missing_unseeded.index:
        print(f"  - {sid}: 0 unseeded months → EXCLUDED from aggregate ATT")
    print()
else:
    print("All sites have both seeded and unseeded months. Within-site DiD is defined everywhere.")
    print()

# -- Per-site DiD -------------------------------------------------------------
# For each site, compute DiD_i = mean(gap | seeded) - mean(gap | unseeded).
# Each site produces one scalar DiD; this is the unit of analysis for the
# aggregate ATT below.
site_summary = (
    df.groupby(["site_id", "seeded"])["precip_gap"]
    .mean()
    .unstack(fill_value=np.nan)
    .rename(columns={0: "gap_no_seeding", 1: "gap_during_seeding"})
    .sort_index()
)
site_summary["within_site_did"] = (
    site_summary["gap_during_seeding"] - site_summary["gap_no_seeding"]
)
site_summary = site_summary.join(
    per_site_counts[["n_seeded", "n_unseeded"]], how="left"
)

# -- Equal-weighted aggregate ATT ---------------------------------------------
# ATT = simple mean of DiD_i across sites (each site weighted equally).
# SE is the cross-site standard error of the mean:
#     SE = sd(DiD_i) / sqrt(n_sites)
# This is a site-clustered SE by construction: because each site is
# collapsed to one DiD before averaging, any serial correlation within
# a site is absorbed into that single number and cannot inflate the
# aggregate variance. Inference uses a t-distribution with df = n-1.
site_dids = site_summary["within_site_did"].dropna()
n_sites_used = len(site_dids)
att = site_dids.mean()
se = site_dids.std(ddof=1) / np.sqrt(n_sites_used)
t_stat = att / se
p_value = 2 * stats.t.sf(abs(t_stat), df=n_sites_used - 1)
sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

print("=" * 72)
print("  Within-site DiD: Effect of Seeding on Target-Control Precipitation Gap")
print("  Equal-weighted across sites | SE clustered at site level by construction")
print("=" * 72)
print(f"ATT = {att:.3f} mm  (SE = {se:.3f}, t = {t_stat:.2f}, "
      f"p = {p_value:.4f}) {sig}")
print(f"Based on {n_sites_used} sites with both seeded and unseeded months")
print()

# -- Site-level summary table -------------------------------------------------
print("=" * 80)
print("  SITE-LEVEL SUMMARY: Mean Target - Control Precipitation Gap (mm)")
print("=" * 80)
print(f"{'site_id':<30} {'No Seeding':>14} {'During Seeding':>16} {'Within-site DiD':>17}")
print("-" * 80)
for site_id, row in site_summary.iterrows():
    no_seed = row["gap_no_seeding"]
    seed = row["gap_during_seeding"]
    no_str = f"{no_seed:>14.2f}" if pd.notna(no_seed) else f"{'n/a':>14}"
    seed_str = f"{seed:>16.2f}" if pd.notna(seed) else f"{'n/a':>16}"
    if pd.notna(no_seed) and pd.notna(seed):
        diff = seed - no_seed
        diff_str = f"{diff:>+17.2f}"
    else:
        diff_str = f"{'n/a':>17}"
    print(f"{site_id:<30} {no_str} {seed_str} {diff_str}")
print("-" * 80)

(ROOT / "data" / "output").mkdir(parents=True, exist_ok=True)
site_summary.to_csv(ROOT / "data" / "output" / "site_level_seeding_gaps.csv")
print(f"Saved {ROOT / 'data' / 'output' / 'site_level_seeding_gaps.csv'}")

# -- Per-site line charts (paginated) -----------------------------------------
# Monthly target vs control precipitation with seeding months shaded.
# Paginated: SITES_PER_PAGE sites per PNG (ROWS_PER_PAGE × COLS_PER_PAGE grid)
# so each file stays small and browsable.

COLS_PER_PAGE = 3
ROWS_PER_PAGE = 4
SITES_PER_PAGE = COLS_PER_PAGE * ROWS_PER_PAGE

# Clear any stale pages + the old monolithic file from previous runs.
for stale in FIG_DIR.glob("site_precip_page*.png"):
    stale.unlink()
legacy = FIG_DIR / "site_precip_timeseries.png"
if legacy.exists():
    legacy.unlink()

site_ids = sorted(df["site_id"].unique())
n_sites = len(site_ids)
n_pages = (n_sites + SITES_PER_PAGE - 1) // SITES_PER_PAGE

saved_paths = []
for page in range(n_pages):
    page_sites = site_ids[page * SITES_PER_PAGE : (page + 1) * SITES_PER_PAGE]

    fig, axes = plt.subplots(
        ROWS_PER_PAGE, COLS_PER_PAGE,
        figsize=(6 * COLS_PER_PAGE, 3.2 * ROWS_PER_PAGE),
        sharex=True,
    )
    axes = axes.flatten()

    for i, site_id in enumerate(page_sites):
        ax = axes[i]
        sd = df[df["site_id"] == site_id].sort_values("year_month")

        ax.plot(sd["year_month"], sd["target_area_precip_mm"],
                color="C0", linewidth=0.6, alpha=0.85, label="Target")
        ax.plot(sd["year_month"], sd["control_area_precip_mm"],
                color="C1", linewidth=0.6, alpha=0.85, label="Control")

        seeded_months = sd[sd["seeded"] == 1]["year_month"]
        if not seeded_months.empty:
            dt = seeded_months.values
            breaks = np.where(np.diff(dt) > np.timedelta64(35, "D"))[0]
            starts = [dt[0]] + [dt[b + 1] for b in breaks]
            ends = [dt[b] for b in breaks] + [dt[-1]]
            for s, e in zip(starts, ends):
                ax.axvspan(s, e + np.timedelta64(15, "D"),
                           alpha=0.12, color="steelblue", zorder=0)

        state = sd["state"].iloc[0] if "state" in sd.columns else ""
        title = f"{site_id} ({state})" if state else site_id
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("Precip (mm)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    for j in range(len(page_sites), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Monthly Precipitation: Target vs Control — page {page + 1}/{n_pages}"
        "  (blue shading = active seeding months)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path = FIG_DIR / f"site_precip_page{page + 1:02d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(out_path)

print(f"Saved {len(saved_paths)} paginated figures:")
for p in saved_paths:
    print(f"  {p}")
