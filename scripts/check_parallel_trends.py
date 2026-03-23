"""
Parallel Trends Assessment for Cloud Seeding TWFE
==================================================

For DiD-eligible sites, tests whether the target–control precipitation gap
was stable before seeding began — the core identifying assumption.

Approach
--------
1. Restrict to seeding-season months only (the months when treatment could
   occur).  This ensures we compare like-for-like across years and avoids
   contamination from off-season months that are never treated.

2. Compute each site's annual seeding-season average gap (target − control)
   in event time (years relative to treatment start).

3. Estimate an event-study regression:  gap_is = α_i + Σ_k β_k · 1[event_year=k] + ε_is
   with the last pre-treatment year (k = −1) as the reference period.
   Pre-treatment β_k ≈ 0 supports parallel trends.

4. Plot the event-study coefficients with 95% CIs.

5. Run a joint F-test on all pre-treatment coefficients (H0: all β_k = 0
   for k < 0).

Outputs
-------
  figures/parallel_trends_event_study.png
  Console: pre-trend test statistics

Data: data/cloud_seeding_monthly_panel.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="linearmodels")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cloud_seeding_monthly_panel.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Load and filter ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]

# DiD-eligible sites only
elig = df[df["did_eligible"]].copy()

# Restrict to seeding-season months for each site
elig["seeding_months"] = elig["seeding_season_months"].str.split(";")
elig["in_season"] = elig.apply(lambda r: str(r["month"]) in r["seeding_months"], axis=1)
season = elig[elig["in_season"]].copy()

print(f"DiD-eligible sites: {season['site_id'].nunique()}")
print(f"Seeding-season observations: {len(season):,}")

# ── Compute annual seeding-season gap per site ───────────────────────────────
# Event time: year relative to treatment start
season["event_year"] = season["year"] - season["treatment_start_year"]

# Average gap across seeding-season months within each site-year
site_year = (
    season.groupby(["site_id", "year", "event_year", "treatment_start_year"])
    ["precip_gap"]
    .mean()
    .reset_index()
)

# Trim to reasonable event window
EVENT_MIN, EVENT_MAX = -10, 10
site_year = site_year[
    (site_year["event_year"] >= EVENT_MIN) & (site_year["event_year"] <= EVENT_MAX)
].copy()

print(f"Site-year observations in [{EVENT_MIN}, {EVENT_MAX}]: {len(site_year):,}")
print()

# ── Event-study regression ───────────────────────────────────────────────────
# Create dummies for each event year, omitting k = -1 as reference
event_years = sorted(site_year["event_year"].unique())
ref_year = -1
dummy_years = [k for k in event_years if k != ref_year]

for k in dummy_years:
    site_year[f"ey_{k}"] = (site_year["event_year"] == k).astype(int)

dummy_cols = [f"ey_{k}" for k in dummy_years]

panel = site_year.set_index(["site_id", "event_year"]).sort_index()
mod = PanelOLS(
    panel["precip_gap"],
    panel[dummy_cols],
    entity_effects=True,
)
res = mod.fit(cov_type="clustered", cluster_entity=True)

# Extract coefficients
coefs = []
for k in event_years:
    if k == ref_year:
        coefs.append({"event_year": k, "coef": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    else:
        col = f"ey_{k}"
        ci = res.conf_int().loc[col]
        coefs.append({
            "event_year": k,
            "coef": res.params[col],
            "se": res.std_errors[col],
            "ci_lo": ci["lower"],
            "ci_hi": ci["upper"],
        })
coefs_df = pd.DataFrame(coefs).sort_values("event_year")

# ── Joint pre-trend F-test ───────────────────────────────────────────────────
pre_cols = [f"ey_{k}" for k in dummy_years if k < ref_year]
if pre_cols:
    # Restriction matrix: test that all pre-treatment coefficients = 0
    R = np.zeros((len(pre_cols), len(dummy_cols)))
    for i, col in enumerate(pre_cols):
        R[i, dummy_cols.index(col)] = 1
    f_test = res.wald_test(formula=None, restriction=R)
    f_stat = f_test.stat
    f_pval = f_test.pval
else:
    f_stat, f_pval = np.nan, np.nan

print("=" * 60)
print("  PRE-TREATMENT TREND TEST")
print("=" * 60)
print(f"  Reference period: event year = {ref_year}")
print(f"  Pre-treatment coefficients tested: {len(pre_cols)}")
print(f"  Joint Wald statistic: {f_stat:.3f}")
print(f"  p-value: {f_pval:.4f}")
if f_pval > 0.05:
    print("  → Cannot reject parallel trends (p > 0.05)")
else:
    print("  → WARNING: Pre-treatment coefficients jointly significant")
print()

print("Pre-treatment event-year coefficients:")
print(f"  {'k':>4} {'coef':>10} {'SE':>10} {'95% CI':>24}")
print("  " + "-" * 50)
for _, r in coefs_df[coefs_df["event_year"] <= 0].iterrows():
    k = int(r["event_year"])
    marker = " (ref)" if k == ref_year else ""
    print(f"  {k:>4} {r['coef']:>10.3f} {r['se']:>10.3f} "
          f"  [{r['ci_lo']:>8.3f}, {r['ci_hi']:>8.3f}]{marker}")

# ── Event-study plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

pre = coefs_df[coefs_df["event_year"] < 0]
post = coefs_df[coefs_df["event_year"] > 0]
zero = coefs_df[coefs_df["event_year"] == 0]

ax.fill_between(
    coefs_df["event_year"], coefs_df["ci_lo"], coefs_df["ci_hi"],
    alpha=0.15, color="steelblue", label="95% CI"
)
ax.plot(pre["event_year"], pre["coef"], "o-", color="gray", ms=5, label="Pre-treatment")
ax.plot(post["event_year"], post["coef"], "o-", color="steelblue", ms=5, label="Post-treatment")
ax.plot(zero["event_year"], zero["coef"], "D", color="C3", ms=7, zorder=5, label="Treatment onset")
ax.plot(ref_year, 0, "s", color="black", ms=7, zorder=5, label=f"Reference (k={ref_year})")

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.axvline(-0.5, color="gray", linewidth=1, linestyle="--", alpha=0.6)

ax.set_xlabel("Years relative to first seeding season", fontsize=11)
ax.set_ylabel("Target − Control precipitation gap (mm)\n(seeding-season months only)", fontsize=11)
ax.set_title("Event Study: Cloud Seeding Effect on Precipitation Gap", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = FIG_DIR / "parallel_trends_event_study.png"
fig.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved {out_path}")
