"""
Cloud Seeding TWFE Explorer
============================
Interactive site-by-site inspection of target vs control precipitation
and per-site DiD causal effect estimates.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    # Derive treatment_start_year: first month with seeding == 1
    first_seeded = (
        df[df["seeded"] == 1]
        .groupby("site_id")["year_month"]
        .min()
        .dt.year
        .rename("treatment_start_year")
    )
    df = df.merge(first_seeded, on="site_id", how="left")

    # DiD-eligible: seeding starts AFTER the first available month (has a pre-period)
    first_month = df["year_month"].min()
    df["did_eligible"] = df["treatment_start_year"].apply(
        lambda y: pd.notna(y) and pd.Timestamp(f"{int(y)}-01-01") > first_month
    )

    return df


df = load_data()

st.title("Cloud Seeding DiD Explorer")
st.caption("Target vs control precipitation — select a site to inspect")

# ── Sidebar: site selector ────────────────────────────────────────────────────
site_meta = (
    df.groupby("site_id")
    .agg(
        state=("state", "first"),
        project_name=("project_name", "first"),
        did_eligible=("did_eligible", "first"),
        treatment_start_year=("treatment_start_year", "first"),
        lat=("site_latitude", "first"),
        lon=("site_longitude", "first"),
    )
    .sort_index()
)

# Build readable labels: project name if known, else site_id (state)
def site_label(sid, row):
    name = row["project_name"] if pd.notna(row["project_name"]) else ""
    state = row["state"] if pd.notna(row["state"]) else ""
    if name:
        return f"{name} ({state})"
    return f"{sid} ({state})"

site_labels = {sid: site_label(sid, row) for sid, row in site_meta.iterrows()}

selected = st.sidebar.selectbox(
    "Select site",
    options=list(site_labels.keys()),
    format_func=lambda x: site_labels[x],
)

meta = site_meta.loc[selected]
sd = df[df["site_id"] == selected].sort_values("year_month").copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**State:** {meta['state']}")
if pd.notna(meta["project_name"]):
    st.sidebar.markdown(f"**Program:** {meta['project_name']}")
st.sidebar.markdown(f"**Location:** {meta['lat']:.2f}°N, {meta['lon']:.2f}°W")
st.sidebar.markdown(
    f"**DiD eligible:** {'Yes' if meta['did_eligible'] else 'No (always-treated or never-treated)'}"
)
if pd.notna(meta["treatment_start_year"]):
    st.sidebar.markdown(f"**First seeded year:** {int(meta['treatment_start_year'])}")

n_seeded = sd["seeded"].sum()
n_total  = len(sd)
st.sidebar.markdown(f"**Seeded months:** {n_seeded} / {n_total}")

# ── Per-site DiD estimate ─────────────────────────────────────────────────────
st.markdown("### Per-site causal effect estimate")

did_eligible = bool(meta["did_eligible"])
tx_year = int(meta["treatment_start_year"]) if pd.notna(meta["treatment_start_year"]) else None

if did_eligible and tx_year is not None:
    tx_start = pd.Timestamp(f"{tx_year}-01-01")
    pre  = sd[sd["year_month"] <  tx_start]
    post = sd[sd["year_month"] >= tx_start]

    # Simple DiD: (gap_post_seeded - gap_post_unseeded) vs pre-period baseline
    gap_pre           = pre["precip_gap"].mean()
    gap_post_seeded   = post.loc[post["seeded"] == 1, "precip_gap"].mean()
    gap_post_unseeded = post.loc[post["seeded"] == 0, "precip_gap"].mean()

    # ATT proxy: change in gap during seeded months relative to pre-period
    att_proxy = gap_post_seeded - gap_pre if pd.notna(gap_post_seeded) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pre-period mean gap", f"{gap_pre:+.2f} mm" if pd.notna(gap_pre) else "n/a",
              help="Mean(target − control) before seeding program started")
    c2.metric("Post-period gap (seeded months)", f"{gap_post_seeded:+.2f} mm" if pd.notna(gap_post_seeded) else "n/a",
              help="Mean(target − control) during active seeding months")
    c3.metric("Post-period gap (unseeded months)", f"{gap_post_unseeded:+.2f} mm" if pd.notna(gap_post_unseeded) else "n/a",
              help="Mean(target − control) in post-period but non-seeding months")
    c4.metric("ATT proxy (seeded − pre baseline)", f"{att_proxy:+.2f} mm" if pd.notna(att_proxy) else "n/a",
              help="Change in target−control gap during seeded months vs pre-period. "
                   "Positive = seeding associated with more precipitation at target relative to control.")

    n_pre  = len(pre)
    n_post_s  = (post["seeded"] == 1).sum()
    n_post_ns = (post["seeded"] == 0).sum()
    st.caption(
        f"Pre-period: {n_pre} months  |  "
        f"Post seeded: {n_post_s} months  |  "
        f"Post unseeded: {n_post_ns} months  |  "
        f"Treatment start: {tx_year}"
    )
else:
    st.info(
        "This site is **not DiD-eligible** — it is seeded throughout the full record "
        "with no clean pre-treatment period. The stats below are descriptive comparisons only."
    )
    gap_no_seed = sd.loc[sd["seeded"] == 0, "precip_gap"].mean()
    gap_seed    = sd.loc[sd["seeded"] == 1, "precip_gap"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean gap (unseeded months)", f"{gap_no_seed:+.2f} mm" if pd.notna(gap_no_seed) else "n/a")
    c2.metric("Mean gap (seeded months)",   f"{gap_seed:+.2f} mm"    if pd.notna(gap_seed) else "n/a")
    diff = gap_seed - gap_no_seed if pd.notna(gap_seed) and pd.notna(gap_no_seed) else np.nan
    c3.metric("Difference", f"{diff:+.2f} mm" if pd.notna(diff) else "n/a",
              help="Not a causal estimate — seeding months are endogenously chosen.")

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

# Treatment start line for DiD-eligible sites
if did_eligible and tx_year is not None:
    tx_start = pd.Timestamp(f"{tx_year}-01-01")
    ax1.axvline(tx_start, color="black", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)
    ylim = ax1.get_ylim()
    ax1.text(tx_start, ylim[1], f"  Program start ({tx_year})",
             fontsize=8, color="black", alpha=0.7, va="top")

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
if did_eligible and tx_year is not None:
    ax2.axvline(pd.Timestamp(f"{tx_year}-01-01"),
                color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax2.set_ylabel("Gap: Target − Control (mm)", fontsize=10)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
st.pyplot(fig)
plt.close()

# ── All-sites comparison table ────────────────────────────────────────────────
with st.expander("All sites comparison table"):
    summary = (
        df.groupby(["site_id", "seeded"])["precip_gap"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns={0: "Mean gap (unseeded)", 1: "Mean gap (seeded)"})
    )
    summary["Difference"] = summary["Mean gap (seeded)"] - summary["Mean gap (unseeded)"]
    # Join state and project name
    summary = summary.join(site_meta[["state", "project_name", "did_eligible", "treatment_start_year"]])
    summary = summary.round(2).sort_values("Difference", ascending=False)
    st.dataframe(summary, use_container_width=True)
