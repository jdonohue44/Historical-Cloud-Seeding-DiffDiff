"""
Cloud Seeding Difference-in-Differences Explorer
======================================

Interactive site-by-site inspection of target vs control precipitation and
the difference-in-differences causal estimate:
    ATT = mean(target - control | seeded months) - mean(target - control | unseeded months)

USAGE:
 - streamlit run app.py

INPUT:
 - data/input/cloud_seeding_monthly_panel.csv

OUTPUT:
  - Streamlit dashboard: per-site precipitation series and within-site DiD
  - All-sites comparison table
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv"

st.set_page_config(page_title="Cloud Seeding DiD", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
    df["seeded"] = df["target_area_seeded"].astype(int)
    return df


df = load_data()


@st.cache_data
def compute_aggregate_att(df):
    """Equal-weighted mean of per-site DiDs, with site-clustered SE."""
    site_gap = df.groupby(["site_id", "seeded"])["precip_gap"].mean().unstack()
    dids = (site_gap[1] - site_gap[0]).dropna()
    n = len(dids)
    att = dids.mean()
    se = dids.std(ddof=1) / np.sqrt(n)
    t_stat = att / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    ci_half = stats.t.ppf(0.975, df=n - 1) * se
    ci_low, ci_high = att - ci_half, att + ci_half
    return att, se, t_stat, p_value, ci_low, ci_high, n


att, se, t_stat, p_value, ci_low, ci_high, n_sites_used = compute_aggregate_att(df)
sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

# Color the headline ATT by sign + significance. Significant positive = green
# (seeding → more precip), significant negative = red, not significant = slate.
if p_value < 0.05 and att > 0:
    att_bg, att_fg, att_border = "#dcfce7", "#14532d", "#16a34a"
elif p_value < 0.05 and att < 0:
    att_bg, att_fg, att_border = "#fee2e2", "#7f1d1d", "#dc2626"
else:
    att_bg, att_fg, att_border = "#f1f5f9", "#0f172a", "#64748b"

st.title("Cloud Seeding Difference-in-Differences Explorer")
st.caption(
    "Target vs control precipitation. We compare the "
    "target−control gap during seeded months to the same gap during unseeded months. "
    "The aggregate ATT is the equal-weighted mean of per-site DiDs, so every site "
    "counts the same regardless of how many months it seeded. The DiD is the difference in the target-control gap between seeded and unseeded months."
)

# ── Aggregate ATT ────────────────────────────────────────────────────────────
st.markdown("### Aggregate causal effect (equal-weighted across sites)")

att_cols = st.columns([2.2, 1, 1.4, 1])

att_cols[0].markdown(
    f"""
    <div style="
        background: {att_bg};
        border: 1px solid {att_border};
        border-left: 6px solid {att_border};
        border-radius: 6px;
        padding: 10px 14px;
        color: {att_fg};
    ">
      <div style="font-size: 0.80rem; opacity: 0.75; text-transform: uppercase;
                  letter-spacing: 0.04em; font-weight: 600;">
        ATT (average treatment effect on the treated)
      </div>
      <div style="font-size: 2.0rem; font-weight: 700; line-height: 1.15;">
        {att:+.3f} mm {sig}
      </div>
      <div style="font-size: 0.80rem; opacity: 0.80;">
        Equal-weighted mean of per-site DiDs (n = {n_sites_used} sites)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

att_cols[1].metric(
    "p-value",
    f"{p_value:.4f}",
    help=f"Two-sided p-value. t = {t_stat:+.2f} on {n_sites_used - 1} df "
         f"(t-distribution). Stars: * p<0.05, ** p<0.01, *** p<0.001.",
)
att_cols[2].metric(
    "95% CI",
    f"[{ci_low:+.3f}, {ci_high:+.3f}] mm",
    help="95% confidence interval for the ATT, using a t-distribution with "
         "df = n_sites − 1. Clustered at the site level by construction "
         "(each site contributes a single DiD).",
)
att_cols[3].metric(
    "Std. error",
    f"{se:.3f}",
    help="SE = sd(DiD_i) / sqrt(n_sites). Clustered at the site level by "
         "construction — each site is collapsed to one DiD before averaging, "
         "so within-site serial correlation cannot inflate the aggregate variance.",
)

st.markdown("---")

# ── Sidebar: site selector ────────────────────────────────────────────────────
site_meta = (
    df.groupby("site_id")
    .agg(
        state=("state", "first"),
        project_name=("project_name", "first"),
        lat=("site_latitude", "first"),
        lon=("site_longitude", "first"),
    )
    .sort_index()
)

def site_label(sid, row):
    name = row["project_name"] if pd.notna(row["project_name"]) else ""
    state = row["state"] if pd.notna(row["state"]) else ""
    if name:
        return f"{name} ({state})"
    return f"{sid} ({state})"

site_labels = {sid: site_label(sid, row) for sid, row in site_meta.iterrows()}

DEFAULT_SITE = "site_028"
site_options = list(site_labels.keys())
default_index = site_options.index(DEFAULT_SITE) if DEFAULT_SITE in site_options else 0

selected = st.sidebar.selectbox(
    "Select site",
    options=site_options,
    index=default_index,
    format_func=lambda x: site_labels[x],
)

meta = site_meta.loc[selected]
sd = df[df["site_id"] == selected].sort_values("year_month").copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**State:** {meta['state']}")
if pd.notna(meta["project_name"]):
    st.sidebar.markdown(f"**Program:** {meta['project_name']}")
st.sidebar.markdown(f"**Location:** {meta['lat']:.2f}°N, {meta['lon']:.2f}°W")

n_seeded = int(sd["seeded"].sum())
n_total  = len(sd)
n_unseeded = n_total - n_seeded
st.sidebar.markdown(f"**Seeded months:** {n_seeded} / {n_total}")
st.sidebar.markdown(f"**Unseeded months:** {n_unseeded} / {n_total}")

if n_seeded == 0:
    st.warning(
        f"NOTE: **{selected}** has 0 seeded months in the analysis panel. "
        "This site's seeding operations all fall in 2023+, which is outside the "
        "ERA5 precipitation coverage window in this dataset (ends Dec 2022). "
        "It is automatically excluded from the aggregate ATT above."
    )
elif n_unseeded == 0:
    st.warning(
        f"**{selected}** has 0 unseeded months. Within-site DiD is undefined "
        "and it is excluded from the aggregate ATT above."
    )

# ── Per-site within-site DiD ─────────────────────────────────────────────────
st.markdown("### Per-site causal effect estimate (within-site DiD)")

gaps_unseeded = sd.loc[sd["seeded"] == 0, "precip_gap"].dropna().values
gaps_seeded   = sd.loc[sd["seeded"] == 1, "precip_gap"].dropna().values
gap_unseeded  = gaps_unseeded.mean() if len(gaps_unseeded) else np.nan
gap_seeded    = gaps_seeded.mean()   if len(gaps_seeded)   else np.nan

# Welch's two-sample t-test: mean(gap | seeded) − mean(gap | unseeded).
# Unequal-variance SE; each month treated as an observation. Assumes months
# are independent within site (does NOT correct for serial correlation;
# if that matters, switch to Newey-West or block bootstrap later).
if len(gaps_seeded) >= 2 and len(gaps_unseeded) >= 2:
    did = gap_seeded - gap_unseeded
    var_s = gaps_seeded.var(ddof=1)
    var_u = gaps_unseeded.var(ddof=1)
    n_s, n_u = len(gaps_seeded), len(gaps_unseeded)
    site_se = np.sqrt(var_s / n_s + var_u / n_u)
    site_t = did / site_se if site_se > 0 else np.nan
    welch_df = (
        (var_s / n_s + var_u / n_u) ** 2
        / ((var_s / n_s) ** 2 / (n_s - 1) + (var_u / n_u) ** 2 / (n_u - 1))
    )
    site_p = 2 * stats.t.sf(abs(site_t), df=welch_df) if pd.notna(site_t) else np.nan
    ci_half = stats.t.ppf(0.975, df=welch_df) * site_se
    ci_low, ci_high = did - ci_half, did + ci_half
    site_sig = "***" if site_p < 0.001 else "**" if site_p < 0.01 else "*" if site_p < 0.05 else ""
else:
    did = (
        gap_seeded - gap_unseeded
        if pd.notna(gap_seeded) and pd.notna(gap_unseeded)
        else np.nan
    )
    site_se = site_t = site_p = ci_low = ci_high = np.nan
    site_sig = ""

# Color the headline per-site DiD the same way as the aggregate ATT.
if pd.notna(site_p) and site_p < 0.05 and did > 0:
    did_bg, did_fg, did_border = "#dcfce7", "#14532d", "#16a34a"
elif pd.notna(site_p) and site_p < 0.05 and did < 0:
    did_bg, did_fg, did_border = "#fee2e2", "#7f1d1d", "#dc2626"
else:
    did_bg, did_fg, did_border = "#f1f5f9", "#0f172a", "#64748b"

did_value_str = (
    f"{did:+.3f} mm {site_sig}" if pd.notna(did) else "n/a"
)
did_subtitle = (
    f"Seeded months: {n_seeded}  |  Unseeded months: {n_unseeded}"
    if pd.notna(site_t)
    else "Insufficient months for inference (need ≥2 seeded and ≥2 unseeded)"
)

did_cols = st.columns([2.2, 1, 1.4, 1])

did_cols[0].markdown(
    f"""
    <div style="
        background: {did_bg};
        border: 1px solid {did_border};
        border-left: 6px solid {did_border};
        border-radius: 6px;
        padding: 10px 14px;
        color: {did_fg};
    ">
      <div style="font-size: 0.80rem; opacity: 0.75; text-transform: uppercase;
                  letter-spacing: 0.04em; font-weight: 600;">
        Within-site DiD
      </div>
      <div style="font-size: 2.0rem; font-weight: 700; line-height: 1.15;">
        {did_value_str}
      </div>
      <div style="font-size: 0.80rem; opacity: 0.80;">
        {did_subtitle}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

did_cols[1].metric(
    "p-value",
    f"{site_p:.4f}" if pd.notna(site_p) else "n/a",
    help=(f"Two-sided p-value (Welch). t = {site_t:+.2f}. "
          "Stars: * p<0.05, ** p<0.01, *** p<0.001.")
    if pd.notna(site_t)
    else "Needs ≥2 seeded and ≥2 unseeded months to compute.",
)
did_cols[2].metric(
    "95% CI",
    f"[{ci_low:+.3f}, {ci_high:+.3f}] mm" if pd.notna(ci_low) else "n/a",
    help="95% confidence interval for the per-site DiD (Welch's t-interval).",
)
did_cols[3].metric(
    "Std. error",
    f"{site_se:.3f}" if pd.notna(site_se) else "n/a",
    help="Unequal-variance (Welch) SE on the difference in monthly gap means. "
         "Months are treated as independent within this site; serial "
         "correlation is NOT corrected for.",
)

gap_cols = st.columns(2)
gap_cols[0].metric(
    "Mean gap, unseeded months",
    f"{gap_unseeded:+.2f} mm" if pd.notna(gap_unseeded) else "n/a",
    help="Mean(target − control) in months with no active seeding. "
         "Expected to be near zero if the control is a good match.",
)
gap_cols[1].metric(
    "Mean gap, seeded months",
    f"{gap_seeded:+.2f} mm" if pd.notna(gap_seeded) else "n/a",
    help="Mean(target − control) in months with active seeding.",
)

# ── Main chart ────────────────────────────────────────────────────────────────
st.markdown("### Precipitation time series")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1.2], sharex=True)

ax1.plot(sd["year_month"], sd["target_area_precip_mm"],
         color="#2563eb", linewidth=0.9, label="Target area", zorder=2)
ax1.plot(sd["year_month"], sd["control_area_precip_mm"],
         color="#dc2626", linewidth=0.9, label="Control area", zorder=2)

# Shade seeding months
seeded_rows = sd[sd["seeded"] == 1]["year_month"].values
if len(seeded_rows) > 0:
    breaks = np.where(np.diff(seeded_rows) > np.timedelta64(35, "D"))[0]
    starts = [seeded_rows[0]] + [seeded_rows[b + 1] for b in breaks]
    ends   = [seeded_rows[b] for b in breaks] + [seeded_rows[-1]]
    for s, e in zip(starts, ends):
        ax1.axvspan(s, e + np.timedelta64(15, "D"), alpha=0.10, color="#2563eb", zorder=0)

label = meta["project_name"] if pd.notna(meta["project_name"]) else selected
ax1.set_ylabel("Precipitation (mm)", fontsize=11)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title(f"{label}  ({meta['state']})", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.15)

# Bottom panel: gap bar chart, seeded months coloured blue
ax2.bar(sd["year_month"], sd["precip_gap"],
        width=25,
        color=np.where(sd["seeded"] == 1, "#2563eb", "#94a3b8"),
        alpha=0.7, zorder=2)
ax2.axhline(0, color="black", linewidth=0.6)
ax2.set_ylabel("Gap: Target − Control (mm)", fontsize=10)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
st.pyplot(fig)
plt.close()

# ── All-sites comparison table ────────────────────────────────────────────────
@st.cache_data
def compute_all_site_stats(df):
    """Per-site DiD + Welch's t-test summary across all sites."""
    rows = []
    for sid, grp in df.groupby("site_id"):
        g_s = grp.loc[grp["seeded"] == 1, "precip_gap"].dropna().values
        g_u = grp.loc[grp["seeded"] == 0, "precip_gap"].dropna().values
        mean_s = g_s.mean() if len(g_s) else np.nan
        mean_u = g_u.mean() if len(g_u) else np.nan
        if len(g_s) >= 2 and len(g_u) >= 2:
            t_val, p_val = stats.ttest_ind(g_s, g_u, equal_var=False)
            did_val = mean_s - mean_u
        else:
            t_val = p_val = np.nan
            did_val = (
                mean_s - mean_u if pd.notna(mean_s) and pd.notna(mean_u) else np.nan
            )
        rows.append({
            "site_id": sid,
            "Mean gap (unseeded)": mean_u,
            "Mean gap (seeded)":   mean_s,
            "Within-site DiD":     did_val,
            "t-stat":              t_val,
            "p-value":             p_val,
            "n_seeded":            int(len(g_s)),
            "n_unseeded":          int(len(g_u)),
        })
    return pd.DataFrame(rows).set_index("site_id")


with st.expander("All sites comparison table"):
    summary = compute_all_site_stats(df).join(site_meta[["state", "project_name"]])
    summary = summary.sort_values("Within-site DiD", ascending=False, na_position="last")
    st.dataframe(
        summary.style.format({
            "Mean gap (unseeded)": "{:+.2f}",
            "Mean gap (seeded)":   "{:+.2f}",
            "Within-site DiD":     "{:+.2f}",
            "t-stat":              "{:+.2f}",
            "p-value":             "{:.4f}",
        }, na_rep="n/a"),
        use_container_width=True,
    )
    st.caption(
        "p-values are two-sided from Welch's (unequal-variance) t-test on the "
        "monthly target − control gap: seeded months vs unseeded months. "
        "Assumes months are independent within site (serial correlation not corrected)."
    )
