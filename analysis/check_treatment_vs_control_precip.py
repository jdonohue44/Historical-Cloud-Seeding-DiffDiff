"""
CHECK TREATMENT vs CONTROL PRECIPITATION ASSUMPTION
=====================================================================

This script validates the identifying assumption of our difference-in-differences estimation.
Namely, that the Average Treatment Effect on the Treated (ATT) is:

ATT = mean(target - control precip gap during seeded months) - mean(target - control precip gap during unseeded months)

For this estimator to be trustworthy, the difference in precip between target and control areas 
must be near zero in the absence of seeding

METHOD:
1) Restrict to months with target_area_seeded == 0.
2) Overall: mean gap across all unseeded site-months, with 95% CI.
3) Per-site: mean gap during unseeded months, with 95% CI and sample size.

INPUT:
 - data/input/cloud_seeding_monthly_panel.csv

OUTPUT:
 - Console: overall mean gap + per-site diagnostic table + selection
   diagnostics (correlations of n_seeded with site-level DiD and with
   control-quality gap).
 - data/output/control_quality_by_site.csv
 - figures/control_quality_unseeded_gap.png
     Per-site mean target-minus-control gap during unseeded months, with 95% CI.
 - figures/selection_diagnostics.png
     Two scatter plots: n_seeded vs site-level DiD, and n_seeded vs
     unseeded-month gap. Flags whether heavy-seeding sites differ
     systematically in effect size or control quality.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"
OUT_DIR = ROOT / "data" / "output"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
df["seeded"] = df["target_area_seeded"].astype(int)

unseeded = df[df["seeded"] == 0].copy()

print(f"Sites: {df['site_id'].nunique()}")
print(f"Total site-months:     {len(df):,}")
print(f"Unseeded site-months:  {len(unseeded):,}  "
      f"({len(unseeded) / len(df):.1%})")
print()

# ── Overall mean gap during unseeded months ──────────────────────────────────
# "Overall" = across all site-months (all sites combined), not per-site.
overall_mean = unseeded["precip_gap"].mean()
overall_se = unseeded["precip_gap"].std(ddof=1) / np.sqrt(len(unseeded))
overall_ci = 1.96 * overall_se

print("=" * 72)
print("  Mean target-minus-control gap during unseeded months (all sites)")
print("=" * 72)
print(f"  Mean gap: {overall_mean:+.3f} mm  "
      f"(95% CI: [{overall_mean - overall_ci:+.3f}, {overall_mean + overall_ci:+.3f}])")
print(
    f"  → We are 95% confident that the true mean difference in monthly "
    f"precipitation\n"
    f"    between target and control areas during unseeded months lies between\n"
    f"    {overall_mean - overall_ci:+.3f} mm and {overall_mean + overall_ci:+.3f} mm. "
    f"A small interval near zero supports the\n"
    f"    identifying assumption that control tracks target in the absence of seeding."
)
print()

# ── Per-site diagnostics ─────────────────────────────────────────────────────
rows = []
for sid, g in unseeded.groupby("site_id"):
    n = len(g)
    mean_gap = g["precip_gap"].mean()
    se = g["precip_gap"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    ci95 = 1.96 * se if pd.notna(se) else np.nan
    rows.append({
        "site_id": sid,
        "n_unseeded_months": n,
        "mean_gap_unseeded_mm": mean_gap,
        "se_mm": se,
        "ci95_mm": ci95,
    })

per_site = pd.DataFrame(rows).sort_values("mean_gap_unseeded_mm")

print("=" * 72)
print("  PER-SITE: mean gap during unseeded months")
print("=" * 72)
print(f"  {'site_id':<30} {'n':>5} {'mean_gap':>12} {'95% CI':>22}")
print("  " + "-" * 70)
for _, r in per_site.iterrows():
    n = int(r["n_unseeded_months"])
    mg = r["mean_gap_unseeded_mm"]
    ci = r["ci95_mm"]
    ci_str = (f"[{mg - ci:+.2f}, {mg + ci:+.2f}]"
              if pd.notna(ci) else "n/a")
    print(f"  {r['site_id']:<30} {n:>5d} {mg:>+12.3f} {ci_str:>22}")
print("  " + "-" * 70)

csv_out = OUT_DIR / "control_quality_by_site.csv"
per_site.round(4).to_csv(csv_out, index=False)
print(f"\nSaved {csv_out}")

# ── Plot: per-site mean gap during unseeded months ───────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(per_site))))
y = np.arange(len(per_site))
ax.errorbar(
    per_site["mean_gap_unseeded_mm"], y,
    xerr=per_site["ci95_mm"],
    fmt="o", color="#334155", ecolor="#94a3b8",
    capsize=3, markersize=5,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(per_site["site_id"], fontsize=9)
ax.set_xlabel("Target − Control gap (mm)", fontsize=11)
ax.set_title(
    "Control-quality diagnostic: mean target−control gap during unseeded months\n"
    "Per site, error bars = 95% CI",
    fontsize=12,
)
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout()
fig_out = FIG_DIR / "control_quality_unseeded_gap.png"
fig.savefig(fig_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fig_out}")

# ── Selection diagnostics ────────────────────────────────────────────────────
# Do heavy-seeding sites differ systematically? Check two correlations:
#   (1) n_seeded vs site-level DiD  → does effect size track intensity?
#   (2) n_seeded vs unseeded-month gap → does control quality track intensity?
# If either is non-zero, weighting the aggregate ATT by n_seeded would lean
# on sites that differ on the dimension in question.

site_gap_by_seed = (
    df.groupby(["site_id", "seeded"])["precip_gap"].mean().unstack()
)
site_did = (site_gap_by_seed[1] - site_gap_by_seed[0]).rename("within_site_did")
n_seeded_per_site = df.groupby("site_id")["seeded"].sum().rename("n_seeded")

diag = (
    per_site.set_index("site_id")[["mean_gap_unseeded_mm"]]
    .join(site_did)
    .join(n_seeded_per_site)
    .dropna()
)

corr_did = diag[["n_seeded", "within_site_did"]].corr().iloc[0, 1]
corr_gap = diag[["n_seeded", "mean_gap_unseeded_mm"]].corr().iloc[0, 1]

print()
print("=" * 72)
print("  Selection diagnostics (per-site, n = {})".format(len(diag)))
print("=" * 72)
print(f"  corr(n_seeded, site-level DiD)              = {corr_did:+.3f}")
print(f"  corr(n_seeded, unseeded-month gap)          = {corr_gap:+.3f}")
print(
    f"  → The first correlation ({corr_did:+.3f}) measures whether heavier-seeded\n"
    f"    sites show larger treatment effects. A value near zero means effect\n"
    f"    size does not track seeding intensity — no sign of selection into\n"
    f"    more seeding at sites where it 'works better.'\n"
    f"  → The second correlation ({corr_gap:+.3f}) measures whether heavier-seeded\n"
    f"    sites have worse control quality (a target-control gap that drifts\n"
    f"    from zero even without seeding). A value near zero means the\n"
    f"    identifying assumption holds equally well at light- and heavy-seeding\n"
    f"    sites. Together, both near zero means weighting the aggregate ATT by\n"
    f"    n_seeded would not materially shift the answer."
)
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(diag["n_seeded"], diag["within_site_did"],
                alpha=0.7, color="#334155")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Seeded months at site (n_seeded)")
axes[0].set_ylabel("Site-level DiD (mm)")
axes[0].set_title(f"Effect size vs seeding intensity\n(Pearson r = {corr_did:+.3f})")
axes[0].grid(True, alpha=0.25)

axes[1].scatter(diag["n_seeded"], diag["mean_gap_unseeded_mm"],
                alpha=0.7, color="#334155")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Seeded months at site (n_seeded)")
axes[1].set_ylabel("Mean unseeded-month gap (mm)")
axes[1].set_title(f"Control quality vs seeding intensity\n(Pearson r = {corr_gap:+.3f})")
axes[1].grid(True, alpha=0.25)

fig.tight_layout()
fig_out2 = FIG_DIR / "selection_diagnostics.png"
fig.savefig(fig_out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fig_out2}")
