"""
Cloud Seeding Difference-in-Differences Explorer
================================================

Interactive site-by-site inspection of target vs control precipitation
and the difference-in-differences causal estimate:

    ATT = mean(target - control | seeded months)
        - mean(target - control | unseeded months)

Use the "Dataset" radio in the sidebar to switch between the ERA5 and
PRISM panels, or view both side-by-side for direct comparison.

USAGE:
  - streamlit run app.py

INPUT:
  - data/input/cloud_seeding_monthly_panel.csv         (ERA5)
  - data/input/cloud_seeding_monthly_panel_prism.csv   (PRISM)

NOTE: rows with `bogus_control == True` (sites whose control coordinates
fall outside the western US due to geocoder fallbacks to Guangzhou/Princeton)
are filtered out by default. Remove the filter once the NC control coords
are corrected. See data/scripts/audit_era5_data.py for context.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_FILES = {
    "ERA5":  ROOT / "data" / "input" / "cloud_seeding_monthly_panel.csv",
    "PRISM": ROOT / "data" / "input" / "cloud_seeding_monthly_panel_prism.csv",
}

st.set_page_config(page_title="Cloud Seeding DiD", layout="wide")


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data(dataset):
    df = pd.read_csv(DATASET_FILES[dataset])
    df = df[~df["bogus_control"]].copy()
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
    df["seeded"] = df["target_area_seeded"].astype(int)
    return df


@st.cache_data
def compute_aggregate_att(df):
    site_gap = df.groupby(["site_id", "seeded"])["precip_gap"].mean().unstack()
    dids = (site_gap[1] - site_gap[0]).dropna()
    n = len(dids)
    att = dids.mean()
    se = dids.std(ddof=1) / np.sqrt(n)
    t_stat = att / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    ci_half = stats.t.ppf(0.975, df=n - 1) * se
    return att, se, t_stat, p_value, att - ci_half, att + ci_half, n


@st.cache_data
def compute_all_site_stats(df):
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
            did_val = (mean_s - mean_u
                       if pd.notna(mean_s) and pd.notna(mean_u) else np.nan)
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


# ── UI helpers ───────────────────────────────────────────────────────────────

def _card_colors(val, p, alpha=0.05):
    if pd.notna(p) and p < alpha and val > 0:
        return "#dcfce7", "#14532d", "#16a34a"
    if pd.notna(p) and p < alpha and val < 0:
        return "#fee2e2", "#7f1d1d", "#dc2626"
    return "#f1f5f9", "#0f172a", "#64748b"


def _big_card(label, value, subtitle, bg, fg, border):
    return f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-left: 6px solid {border};
        border-radius: 6px;
        padding: 10px 14px;
        color: {fg};
    ">
      <div style="font-size: 0.80rem; opacity: 0.75; text-transform: uppercase;
                  letter-spacing: 0.04em; font-weight: 600;">
        {label}
      </div>
      <div style="font-size: 2.0rem; font-weight: 700; line-height: 1.15;">
        {value}
      </div>
      <div style="font-size: 0.80rem; opacity: 0.80;">
        {subtitle}
      </div>
    </div>
    """


def _sig_stars(p):
    if not pd.notna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _shade_seeding(ax, sd):
    seeded_rows = sd[sd["seeded"] == 1]["year_month"].values
    if len(seeded_rows) == 0:
        return
    breaks = np.where(np.diff(seeded_rows) > np.timedelta64(35, "D"))[0]
    starts = [seeded_rows[0]] + [seeded_rows[b + 1] for b in breaks]
    ends   = [seeded_rows[b] for b in breaks] + [seeded_rows[-1]]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e + np.timedelta64(15, "D"),
                   alpha=0.10, color="#2563eb", zorder=0)


# ── Section renderers ────────────────────────────────────────────────────────

def render_aggregate_att(df, dataset_label):
    att, se, t_stat, p_value, ci_low, ci_high, n_used = compute_aggregate_att(df)
    sig = _sig_stars(p_value)
    bg, fg, border = _card_colors(att, p_value)

    cols = st.columns([2.2, 1, 1.4, 1])
    cols[0].markdown(
        _big_card(
            f"{dataset_label} — ATT (avg. treatment effect on the treated)",
            f"{att:+.3f} mm {sig}",
            f"Equal-weighted mean of per-site DiDs (n = {n_used} sites)",
            bg, fg, border,
        ),
        unsafe_allow_html=True,
    )
    cols[1].metric(
        "p-value", f"{p_value:.4f}",
        help=f"Two-sided p-value. t = {t_stat:+.2f} on {n_used - 1} df. "
             "Stars: * p<0.05, ** p<0.01, *** p<0.001.",
    )
    cols[2].metric(
        "95% CI", f"[{ci_low:+.3f}, {ci_high:+.3f}] mm",
        help="Clustered at site level by construction — each site "
             "contributes one DiD to the aggregate.",
    )
    cols[3].metric("Std. error", f"{se:.3f}")


def render_site_panel(df, selected, dataset_label):
    sd = df[df["site_id"] == selected].sort_values("year_month").copy()

    if len(sd) == 0:
        st.info(f"**{selected}** is not present in the {dataset_label} panel.")
        return

    gaps_u = sd.loc[sd["seeded"] == 0, "precip_gap"].dropna().values
    gaps_s = sd.loc[sd["seeded"] == 1, "precip_gap"].dropna().values
    mean_u = gaps_u.mean() if len(gaps_u) else np.nan
    mean_s = gaps_s.mean() if len(gaps_s) else np.nan
    n_seeded, n_unseeded = len(gaps_s), len(gaps_u)

    if n_seeded >= 2 and n_unseeded >= 2:
        did = mean_s - mean_u
        var_s, var_u = gaps_s.var(ddof=1), gaps_u.var(ddof=1)
        site_se = np.sqrt(var_s / n_seeded + var_u / n_unseeded)
        site_t = did / site_se if site_se > 0 else np.nan
        welch_df = (
            (var_s / n_seeded + var_u / n_unseeded) ** 2
            / ((var_s / n_seeded) ** 2 / (n_seeded - 1)
               + (var_u / n_unseeded) ** 2 / (n_unseeded - 1))
        )
        site_p = 2 * stats.t.sf(abs(site_t), df=welch_df) if pd.notna(site_t) else np.nan
        ci_half = stats.t.ppf(0.975, df=welch_df) * site_se
        ci_low, ci_high = did - ci_half, did + ci_half
    else:
        did = (mean_s - mean_u
               if pd.notna(mean_s) and pd.notna(mean_u) else np.nan)
        site_se = site_t = site_p = ci_low = ci_high = np.nan

    bg, fg, border = _card_colors(did, site_p)
    did_value = f"{did:+.3f} mm {_sig_stars(site_p)}" if pd.notna(did) else "n/a"
    subtitle = (
        f"Seeded months: {n_seeded}  |  Unseeded months: {n_unseeded}"
        if pd.notna(site_t)
        else "Insufficient months (need ≥2 seeded and ≥2 unseeded)"
    )

    cols = st.columns([2.2, 1, 1.4, 1])
    cols[0].markdown(
        _big_card(f"{dataset_label} — Within-site DiD",
                  did_value, subtitle, bg, fg, border),
        unsafe_allow_html=True,
    )
    cols[1].metric(
        "p-value", f"{site_p:.4f}" if pd.notna(site_p) else "n/a",
        help=(f"Two-sided p-value (Welch). t = {site_t:+.2f}. "
              "Stars: * p<0.05, ** p<0.01, *** p<0.001.")
        if pd.notna(site_t) else "Needs ≥2 seeded and ≥2 unseeded months.",
    )
    cols[2].metric(
        "95% CI",
        f"[{ci_low:+.3f}, {ci_high:+.3f}] mm" if pd.notna(ci_low) else "n/a",
        help="Welch's t-interval on the monthly-gap difference.",
    )
    cols[3].metric(
        "Std. error",
        f"{site_se:.3f}" if pd.notna(site_se) else "n/a",
        help="Unequal-variance SE. Months assumed independent within "
             "site; serial correlation not corrected.",
    )

    gap_cols = st.columns(2)
    gap_cols[0].metric(
        "Mean gap, unseeded",
        f"{mean_u:+.2f} mm" if pd.notna(mean_u) else "n/a",
    )
    gap_cols[1].metric(
        "Mean gap, seeded",
        f"{mean_s:+.2f} mm" if pd.notna(mean_s) else "n/a",
    )

    if n_seeded == 0:
        st.warning(
            f"**{selected}** has 0 seeded months in the {dataset_label} panel "
            "and is excluded from the aggregate ATT."
        )
    elif n_unseeded == 0:
        st.warning(
            f"**{selected}** has 0 unseeded months in the {dataset_label} "
            "panel and is excluded from the aggregate ATT."
        )


def render_time_series_single(df, selected, meta, dataset_label):
    sd = df[df["site_id"] == selected].sort_values("year_month").copy()
    if len(sd) == 0:
        st.info(f"No {dataset_label} data for {selected}.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    height_ratios=[3, 1.2], sharex=True)
    ax1.plot(sd["year_month"], sd["target_area_precip_mm"],
             color="#2563eb", linewidth=0.9, label="Target area", zorder=2)
    ax1.plot(sd["year_month"], sd["control_area_precip_mm"],
             color="#dc2626", linewidth=0.9, label="Control area", zorder=2)
    _shade_seeding(ax1, sd)

    label = meta["project_name"] if pd.notna(meta["project_name"]) else selected
    ax1.set_ylabel("Precipitation (mm)", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_title(f"{label}  ({meta['state']}) — {dataset_label}",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.15)

    ax2.bar(sd["year_month"], sd["precip_gap"], width=25,
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


def render_time_series_overlay(df_era5, df_prism, selected, meta):
    sd_e = df_era5[df_era5["site_id"] == selected].sort_values("year_month")
    sd_p = df_prism[df_prism["site_id"] == selected].sort_values("year_month")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5),
                                    height_ratios=[3, 1.4], sharex=True)
    ax1.plot(sd_e["year_month"], sd_e["target_area_precip_mm"],
             color="#1d4ed8", linewidth=0.9, label="Target (ERA5)")
    ax1.plot(sd_p["year_month"], sd_p["target_area_precip_mm"],
             color="#1d4ed8", linewidth=0.9, label="Target (PRISM)",
             linestyle="--", alpha=0.85)
    ax1.plot(sd_e["year_month"], sd_e["control_area_precip_mm"],
             color="#b91c1c", linewidth=0.9, label="Control (ERA5)")
    ax1.plot(sd_p["year_month"], sd_p["control_area_precip_mm"],
             color="#b91c1c", linewidth=0.9, label="Control (PRISM)",
             linestyle="--", alpha=0.85)
    _shade_seeding(ax1, sd_e if len(sd_e) else sd_p)

    label = meta["project_name"] if pd.notna(meta["project_name"]) else selected
    ax1.set_ylabel("Precipitation (mm)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right", ncol=2)
    ax1.set_title(f"{label}  ({meta['state']}) — ERA5 vs PRISM overlay",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.15)

    ax2.plot(sd_e["year_month"], sd_e["precip_gap"],
             color="#0f172a", linewidth=0.8, label="Gap (ERA5)", alpha=0.85)
    ax2.plot(sd_p["year_month"], sd_p["precip_gap"],
             color="#64748b", linewidth=0.8, label="Gap (PRISM)",
             alpha=0.85, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Gap: Target − Control (mm)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.15)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_all_sites_single(df, site_meta, dataset_label):
    with st.expander(f"All sites comparison table — {dataset_label}"):
        summary = (
            compute_all_site_stats(df)
            .join(site_meta[["state", "project_name"]])
            .sort_values("Within-site DiD", ascending=False, na_position="last")
        )
        st.dataframe(
            summary.style.format({
                "Mean gap (unseeded)": "{:+.2f}",
                "Mean gap (seeded)":   "{:+.2f}",
                "Within-site DiD":     "{:+.2f}",
                "t-stat":              "{:+.2f}",
                "p-value":             "{:.4f}",
            }, na_rep="n/a"),
            width="stretch",
        )
        st.caption(
            "p-values from Welch's t-test on monthly target − control gap "
            "(seeded vs unseeded months). Serial correlation not corrected."
        )


def render_all_sites_joined(df_era5, df_prism, site_meta):
    with st.expander("All sites comparison — ERA5 vs PRISM"):
        keep = ["Within-site DiD", "p-value", "n_seeded"]
        s_e = compute_all_site_stats(df_era5)[keep].rename(columns={
            "Within-site DiD": "DiD (ERA5)",
            "p-value":         "p (ERA5)",
            "n_seeded":        "n_seeded (ERA5)",
        })
        s_p = compute_all_site_stats(df_prism)[keep].rename(columns={
            "Within-site DiD": "DiD (PRISM)",
            "p-value":         "p (PRISM)",
            "n_seeded":        "n_seeded (PRISM)",
        })
        joined = (
            site_meta[["state", "project_name"]]
            .join(s_e, how="left")
            .join(s_p, how="left")
        )
        joined["ΔDiD (PRISM − ERA5)"] = (
            joined["DiD (PRISM)"] - joined["DiD (ERA5)"]
        )
        joined = joined.sort_values("ΔDiD (PRISM − ERA5)",
                                     ascending=False, na_position="last",
                                     key=lambda s: s.abs())
        st.dataframe(
            joined.style.format({
                "DiD (ERA5)":          "{:+.2f}",
                "p (ERA5)":            "{:.4f}",
                "DiD (PRISM)":         "{:+.2f}",
                "p (PRISM)":           "{:.4f}",
                "ΔDiD (PRISM − ERA5)": "{:+.2f}",
            }, na_rep="n/a"),
            width="stretch",
        )
        st.caption(
            "Joined per-site DiD under each panel. Sorted by |ΔDiD| so sites "
            "where the dataset choice materially changes the site-level "
            "conclusion appear at the top."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

dataset_mode = st.sidebar.radio(
    "Dataset",
    ["ERA5 only", "PRISM only", "Side by side"],
    index=0,
    help="Choose which precipitation panel to analyze, or compare both.",
)

df_era5  = load_data("ERA5")
df_prism = load_data("PRISM")

if dataset_mode == "ERA5 only":
    active = [("ERA5", df_era5)]
elif dataset_mode == "PRISM only":
    active = [("PRISM", df_prism)]
else:
    active = [("ERA5", df_era5), ("PRISM", df_prism)]

# Site meta / selector — after bogus_control filtering both panels cover the
# same 123 sites, so we build the selector from the first active dataset.
site_meta_src = active[0][1]
site_meta = (
    site_meta_src.groupby("site_id")
    .agg(
        state=("state", "first"),
        project_name=("project_name", "first"),
        lat=("site_latitude", "first"),
        lon=("site_longitude", "first"),
    )
    .sort_index()
)

def _site_label(sid, row):
    name = row["project_name"] if pd.notna(row["project_name"]) else ""
    state = row["state"] if pd.notna(row["state"]) else ""
    if name:
        return f"{name} ({state})"
    return f"{sid} ({state})"

site_labels = {sid: _site_label(sid, row) for sid, row in site_meta.iterrows()}
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

st.sidebar.markdown("---")
st.sidebar.markdown(f"**State:** {meta['state']}")
if pd.notna(meta["project_name"]):
    st.sidebar.markdown(f"**Program:** {meta['project_name']}")
st.sidebar.markdown(f"**Location:** {meta['lat']:.2f}°N, {meta['lon']:.2f}°W")

# Per-dataset availability summary for the selected site
st.sidebar.markdown("---")
for label, df in active:
    sd = df[df["site_id"] == selected]
    if len(sd) == 0:
        st.sidebar.markdown(f"**{label}:** site not in panel")
    else:
        ns = int(sd["target_area_seeded"].sum())
        st.sidebar.markdown(f"**{label}:** {ns} seeded / {len(sd)} total months")


# ── Main content ────────────────────────────────────────────────────────────

st.title("Cloud Seeding Difference-in-Differences Explorer")
st.caption(
    "Target vs control precipitation. We compare the target−control gap during "
    "seeded months to the same gap during unseeded months. The aggregate ATT is "
    "the equal-weighted mean of per-site DiDs, so every site counts the same "
    "regardless of how many months it seeded."
)

st.markdown("### Aggregate causal effect (equal-weighted across sites)")
if len(active) == 1:
    render_aggregate_att(active[0][1], active[0][0])
else:
    cols = st.columns(2)
    for col, (label, df) in zip(cols, active):
        with col:
            render_aggregate_att(df, label)

st.markdown("---")

st.markdown("### Per-site causal effect estimate (within-site DiD)")
if len(active) == 1:
    render_site_panel(active[0][1], selected, active[0][0])
else:
    cols = st.columns(2)
    for col, (label, df) in zip(cols, active):
        with col:
            render_site_panel(df, selected, label)

st.markdown("### Precipitation time series")
if len(active) == 1:
    render_time_series_single(active[0][1], selected, meta, active[0][0])
else:
    render_time_series_overlay(df_era5, df_prism, selected, meta)

if len(active) == 1:
    render_all_sites_single(active[0][1], site_meta, active[0][0])
else:
    render_all_sites_joined(df_era5, df_prism, site_meta)
