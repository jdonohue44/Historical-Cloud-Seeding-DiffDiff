"""
TWFE Estimation: Effect of Cloud Seeding on Precipitation
=========================================================

Compares target-area vs control-area precipitation during seeded months
versus non-seeded months, using Two-Way Fixed Effects (site + year-month).

The outcome variable is the precipitation gap (target minus control).
The treatment indicator is whether seeding was active that month.
If seeding works, the gap should be larger when seeding is on.

Input:
 - data/input/cloud_seeding_monthly_panel.csv

 Output:
  - Console: TWFE regression results and site-level summary table
  - data/site_level_seeding_gaps.csv: mean target-control gap by site and seeding status
  - figures/site_precip_timeseries.png: line charts of target vs control precipitation by site, with seeding months shaded
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=RuntimeWarning, module="linearmodels")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"

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

# -- TWFE regression ----------------------------------------------------------
# precip_gap ~ seeded + site fixed effects + year-month fixed effects
# Standard errors clustered at the site level.

panel = df.set_index(["site_id", "year_month"]).sort_index()
mod = PanelOLS(
    panel["precip_gap"],
    panel[["seeded"]],
    entity_effects=True,
    time_effects=True,
)
res = mod.fit(cov_type="clustered", cluster_entity=True)

print("=" * 72)
print("  TWFE: Effect of Seeding on Target-Control Precipitation Gap")
print("  Site FE + Year-Month FE | Clustered SE at site level")
print("=" * 72)
print(res.summary)

att = res.params["seeded"]
se = res.std_errors["seeded"]
p = res.pvalues["seeded"]
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
print(f"\nATT = {att:.3f} mm  (SE = {se:.3f}, p = {p:.4f}) {sig}")
print()

# -- Site-level summary table -------------------------------------------------
# For each site: mean(target - control) when not seeding vs when seeding.
import matplotlib.pyplot as plt

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

site_summary = (
    df.groupby(["site_id", "seeded"])["precip_gap"]
    .mean()
    .unstack(fill_value=np.nan)
    .rename(columns={0: "gap_no_seeding", 1: "gap_during_seeding"})
    .sort_index()
)

print("=" * 80)
print("  SITE-LEVEL SUMMARY: Mean Target - Control Precipitation Gap (mm)")
print("=" * 80)
print(f"{'site_id':<30} {'No Seeding':>14} {'During Seeding':>16} {'Difference':>12}")
print("-" * 80)
for site_id, row in site_summary.iterrows():
    no_seed = row["gap_no_seeding"]
    seed = row["gap_during_seeding"]
    no_str = f"{no_seed:>14.2f}" if pd.notna(no_seed) else f"{'n/a':>14}"
    seed_str = f"{seed:>16.2f}" if pd.notna(seed) else f"{'n/a':>16}"
    if pd.notna(no_seed) and pd.notna(seed):
        diff = seed - no_seed
        diff_str = f"{diff:>+12.2f}"
    else:
        diff_str = f"{'n/a':>12}"
    print(f"{site_id:<30} {no_str} {seed_str} {diff_str}")
print("-" * 80)

# Save as CSV
site_summary_out = site_summary.copy()
site_summary_out["difference"] = site_summary_out["gap_during_seeding"] - site_summary_out["gap_no_seeding"]
site_summary_out.to_csv(ROOT / "data" / "site_level_seeding_gaps.csv")
print(f"Saved {ROOT / 'data' / 'site_level_seeding_gaps.csv'}")

# -- Per-site line charts -----------------------------------------------------
# Monthly target vs control precipitation with seeding months shaded.

site_ids = sorted(df["site_id"].unique())
n_sites = len(site_ids)
cols = 3
rows_grid = (n_sites + cols - 1) // cols

fig, axes = plt.subplots(rows_grid, cols, figsize=(7 * cols, 4 * rows_grid), sharex=True)
axes = axes.flatten()

for i, site_id in enumerate(site_ids):
    ax = axes[i]
    sd = df[df["site_id"] == site_id].sort_values("year_month")

    ax.plot(sd["year_month"], sd["target_area_precip_mm"],
            color="C0", linewidth=0.6, alpha=0.85, label="Target area")
    ax.plot(sd["year_month"], sd["control_area_precip_mm"],
            color="C1", linewidth=0.6, alpha=0.85, label="Control area")

    # Shade seeding months
    seeded_months = sd[sd["seeded"] == 1]["year_month"]
    if not seeded_months.empty:
        dt = seeded_months.values
        breaks = np.where(np.diff(dt) > np.timedelta64(35, "D"))[0]
        starts = [dt[0]] + [dt[b + 1] for b in breaks]
        ends = [dt[b] for b in breaks] + [dt[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e + np.timedelta64(15, "D"),
                       alpha=0.12, color="steelblue", zorder=0)

    state = sd["state"].iloc[0]
    ax.set_title(f"{site_id} ({state})", fontsize=9, fontweight="bold")
    ax.set_ylabel("Precip (mm)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.2)

    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

# Hide unused subplots
for j in range(n_sites, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Monthly Precipitation: Target vs Control Area by Site\n"
    "(blue shading = active seeding months)",
    fontsize=14, y=1.01,
)
fig.tight_layout()
out_path = FIG_DIR / "site_precip_timeseries.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
