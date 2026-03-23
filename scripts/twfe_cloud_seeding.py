"""
TWFE Estimation: Effect of Cloud Seeding on Precipitation
=========================================================

Estimates the ATT of cloud seeding on the target–control precipitation gap
using Two-Way Fixed Effects on a monthly panel.

Model:  ΔP_it = α_i + γ_t + δ · Seeded_it + ε_it

where ΔP = target_precip − control_precip, α_i = site FE, γ_t = year-month FE,
and Seeded_it = 1 when site i is actively being seeded in month t.

Because treatment is time-varying (seasonal on/off), not absorbing, we use
linearmodels PanelOLS directly rather than a standard staggered-DiD estimator.

Specifications
--------------
1. Primary TWFE (DiD-eligible sites only)
   site FE + year-month FE, clustered SEs at site level

2. Seasonality-robust (DiD-eligible sites only)
   Demean precip gap by site × calendar-month before TWFE.
   Guards against site-specific seasonal patterns confounding the estimate.
   After demeaning, δ is identified from: "In January at Alta, is the gap
   larger in 2010–2025 (seeded) vs 2000–2009 (pre-seeding)?"

3. All sites (including always-treated)
   Same as Spec 1 but adds 4 always-treated sites.
   Descriptive only — those sites lack a pre-treatment period.

Data: data/cloud_seeding_monthly_panel.csv
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=RuntimeWarning, module="linearmodels")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cloud_seeding_monthly_panel.csv"

# ── Load and prepare ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
df["seeded"] = df["target_area_seeded"].astype(int)
df["year_month"] = pd.to_datetime(df["year_month"])

eligible = df[df["did_eligible"]].copy()

n_sites_elig = eligible["site_id"].nunique()
n_sites_all = df["site_id"].nunique()
n_months = eligible["year_month"].nunique()

print(f"Panel: {n_sites_all} sites × {n_months} months = {len(df):,} obs")
print(f"DiD-eligible sites: {n_sites_elig}")
print(f"Always-treated sites: {n_sites_all - n_sites_elig}")
print(f"Seeded obs (eligible): {eligible['seeded'].sum():,} / {len(eligible):,}")
print()


def run_twfe(data, outcome_col, label):
    """Run PanelOLS with entity + time FEs, clustered at site."""
    panel = data.set_index(["site_id", "year_month"]).sort_index()
    mod = PanelOLS(
        panel[outcome_col],
        panel[["seeded"]],
        entity_effects=True,
        time_effects=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)
    print(res.summary)
    return res


# ── Spec 1: Primary TWFE ─────────────────────────────────────────────────────
res1 = run_twfe(
    eligible,
    "precip_gap",
    "SPEC 1 — Primary TWFE: site FE + year-month FE\n"
    "  DiD-eligible sites only | clustered at site level"
)

# ── Spec 2: Seasonality-robust ───────────────────────────────────────────────
#
# WHY: Seeding happens in specific months (winter for mountain sites, summer
# for plains). If the target–control gap has its own seasonal cycle (e.g.,
# higher-elevation targets naturally get more winter precip), that pattern
# would inflate the seeded coefficient in Spec 1.
#
# Demeaning by site × calendar-month removes each site's average gap for that
# month-of-year. The seeded coefficient then captures only whether the gap is
# *larger than usual for that site-month* when seeding is active.

elig2 = eligible.copy()
elig2["site_x_calmonth"] = elig2["site_id"] + "_m" + elig2["month"].astype(str).str.zfill(2)
elig2["precip_gap_dm"] = elig2["precip_gap"] - elig2.groupby("site_x_calmonth")["precip_gap"].transform("mean")

res2 = run_twfe(
    elig2,
    "precip_gap_dm",
    "SPEC 2 — Seasonality-Robust: site×calendar-month demeaned\n"
    "  DiD-eligible sites only | clustered at site level"
)

# ── Spec 3: All sites ────────────────────────────────────────────────────────
res3 = run_twfe(
    df,
    "precip_gap",
    "SPEC 3 — All Sites (incl. always-treated): site FE + year-month FE\n"
    "  Descriptive for always-treated sites | clustered at site level"
)

# ── Comparison table ─────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  COMPARISON TABLE")
print("=" * 72)
print(f"{'Specification':<50} {'ATT':>8} {'SE':>8} {'p':>8}")
print("-" * 72)

specs = [
    ("1. TWFE  (eligible, site + ym FE)", res1),
    ("2. Robust (eligible, site×cal-month demeaned)", res2),
    ("3. TWFE  (all sites, site + ym FE)", res3),
]
for label, res in specs:
    att = res.params["seeded"]
    se = res.std_errors["seeded"]
    p = res.pvalues["seeded"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{label:<50} {att:>8.3f} {se:>8.3f} {p:>7.4f} {sig}")

print("-" * 72)
print()
print("Spec 1 is the primary causal estimate.")
print("Spec 2 guards against site-specific seasonal confounding.")
print("Spec 3 is descriptive — always-treated sites lack a pre-period.")

# ── Site-level summary table ─────────────────────────────────────────────────
# For each site: mean(target − control) when NOT seeding vs when seeding
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

print()
print("=" * 80)
print("  SITE-LEVEL SUMMARY: Mean Target − Control Precipitation Gap (mm)")
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

# ── Per-site line charts ─────────────────────────────────────────────────────
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
        # Find contiguous blocks to shade
        dt = seeded_months.values
        breaks = np.where(np.diff(dt) > np.timedelta64(35, "D"))[0]
        starts = [dt[0]] + [dt[b + 1] for b in breaks]
        ends = [dt[b] for b in breaks] + [dt[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e + np.timedelta64(15, "D"),
                       alpha=0.12, color="steelblue", zorder=0)

    state = sd["state"].iloc[0]
    eligible_flag = sd["did_eligible"].iloc[0]
    tag = "" if eligible_flag else " [always-treated]"
    ax.set_title(f"{site_id} ({state}){tag}", fontsize=9, fontweight="bold")
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
