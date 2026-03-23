"""
Cloud Seeding TWFE Explorer
============================
Interactive site-by-site inspection of target vs control precipitation.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cloud_seeding_monthly_panel.csv"

st.set_page_config(page_title="Cloud Seeding DiD", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
    df["seeded"] = df["target_area_seeded"].astype(int)
    return df


df = load_data()

st.title("Cloud Seeding TWFE Explorer")
st.caption("Target vs Control area precipitation — select a site to inspect")

# ── Sidebar: site selector + info ────────────────────────────────────────────
site_meta = (
    df.groupby("site_id")
    .agg(
        state=("state", "first"),
        did_eligible=("did_eligible", "first"),
        treatment_start_year=("treatment_start_year", "first"),
        seeding_season_months=("seeding_season_months", "first"),
    )
    .sort_index()
)

site_labels = {
    sid: f"{sid}  ({row['state']})" for sid, row in site_meta.iterrows()
}

selected = st.sidebar.selectbox(
    "Select site",
    options=list(site_labels.keys()),
    format_func=lambda x: site_labels[x],
)

meta = site_meta.loc[selected]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**State:** {meta['state']}")
st.sidebar.markdown(f"**DiD eligible:** {'Yes' if meta['did_eligible'] else 'No (always-treated)'}")
st.sidebar.markdown(f"**Treatment start:** {int(meta['treatment_start_year'])}")
season_months = meta["seeding_season_months"].split(";")
month_names = {
    "1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr",
    "5": "May", "6": "Jun", "7": "Jul", "8": "Aug",
    "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}
season_str = ", ".join(month_names.get(m, m) for m in season_months)
st.sidebar.markdown(f"**Seeding season:** {season_str}")

sd = df[df["site_id"] == selected].sort_values("year_month").copy()

# Site-level stats
gap_no_seed = sd.loc[sd["seeded"] == 0, "precip_gap"].mean()
gap_seed = sd.loc[sd["seeded"] == 1, "precip_gap"].mean()
n_seeded = (sd["seeded"] == 1).sum()
n_total = len(sd)

target_area = sd["target_area"].iloc[0]
control_area = sd["control_area"].iloc[0]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Target area:**  \n{target_area}")
st.sidebar.markdown(f"**Control area:**  \n{control_area}")

# ── Main chart ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1.2], sharex=True)

# -- Top panel: target vs control with seeded months shaded --
ax1.plot(sd["year_month"], sd["target_area_precip_mm"],
         color="#2563eb", linewidth=0.9, label="Target area", zorder=2)
ax1.plot(sd["year_month"], sd["control_area_precip_mm"],
         color="#dc2626", linewidth=0.9, label="Control area", zorder=2)

# Shade seeding months
seeded_rows = sd[sd["seeded"] == 1]["year_month"].values
if len(seeded_rows) > 0:
    breaks = np.where(np.diff(seeded_rows) > np.timedelta64(35, "D"))[0]
    starts = [seeded_rows[0]] + [seeded_rows[b + 1] for b in breaks]
    ends = [seeded_rows[b] for b in breaks] + [seeded_rows[-1]]
    for s, e in zip(starts, ends):
        ax1.axvspan(s, e + np.timedelta64(15, "D"),
                    alpha=0.10, color="#2563eb", zorder=0)

# Treatment start line for DiD-eligible sites
if meta["did_eligible"]:
    tx_start = pd.Timestamp(f"{int(meta['treatment_start_year'])}-01-01")
    ax1.axvline(tx_start, color="black", linewidth=1.2, linestyle="--", alpha=0.5, zorder=1)
    ax1.text(tx_start, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 100,
             f"  Program start ({int(meta['treatment_start_year'])})",
             fontsize=8, color="black", alpha=0.7, va="top")

ax1.set_ylabel("Precipitation (mm)", fontsize=11)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title(f"{selected}  ({meta['state']})", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.15)

# -- Bottom panel: precipitation gap --
ax2.bar(sd["year_month"], sd["precip_gap"],
        width=25, color=np.where(sd["seeded"] == 1, "#2563eb", "#94a3b8"),
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

# ── Stats cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean gap (no seeding)", f"{gap_no_seed:+.1f} mm" if pd.notna(gap_no_seed) else "n/a")
col2.metric("Mean gap (during seeding)", f"{gap_seed:+.1f} mm" if pd.notna(gap_seed) else "n/a")
if pd.notna(gap_seed) and pd.notna(gap_no_seed):
    diff = gap_seed - gap_no_seed
    col3.metric("Difference", f"{diff:+.1f} mm")
else:
    col3.metric("Difference", "n/a")
col4.metric("Seeded months", f"{n_seeded} / {n_total}")

# ── All-sites comparison table ───────────────────────────────────────────────
with st.expander("All sites comparison table"):
    summary = (
        df.groupby(["site_id", "seeded"])["precip_gap"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns={0: "Mean gap (no seeding)", 1: "Mean gap (during seeding)"})
    )
    summary["Difference"] = summary["Mean gap (during seeding)"] - summary["Mean gap (no seeding)"]
    summary = summary.round(2).sort_values("Difference", ascending=False)
    st.dataframe(summary, use_container_width=True)
