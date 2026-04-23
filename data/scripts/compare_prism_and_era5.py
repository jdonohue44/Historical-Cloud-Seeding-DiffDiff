"""
Compare ERA5 and PRISM precipitation panels.

Produces three regression-style comparison figures:
1. ERA5 DiD vs PRISM DiD
2. ERA5 target precipitation vs PRISM target precipitation
3. ERA5 control precipitation vs PRISM control precipitation

Run:
    python data/scripts/compare_prism_and_era5.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "input"
OUT_DIR = ROOT / "data" / "output"
FIG_DIR = ROOT / "figures"

ERA5_PATH = DATA_DIR / "cloud_seeding_monthly_panel.csv"
PRISM_PATH = DATA_DIR / "cloud_seeding_monthly_panel_prism.csv"


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["bogus_control"]].copy()
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["seeded"] = df["target_area_seeded"].astype(int)
    df["precip_gap"] = df["target_area_precip_mm"] - df["control_area_precip_mm"]
    return df


def compute_site_did(df: pd.DataFrame) -> pd.DataFrame:
    did = (
        df.groupby(["site_id", "seeded"])["precip_gap"]
        .mean()
        .unstack()
        .rename(columns={0: "gap_unseeded", 1: "gap_seeded"})
    )
    did["within_site_did"] = did["gap_seeded"] - did["gap_unseeded"]
    return did[["within_site_did"]].reset_index()


def summarize_relationship(data: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    pair = data[[x_col, y_col]].dropna().copy()
    x = pair[x_col].to_numpy()
    y = pair[y_col].to_numpy()
    fit = stats.linregress(x, y)
    diff = y - x
    y_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0

    return {
        "n": len(pair),
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r": fit.rvalue,
        "r2": fit.rvalue ** 2,
        "mean_diff": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "y_std": y_std,
    }


def classify_interchangeability(summary: dict[str, float]) -> str:
    slope_close = abs(summary["slope"] - 1.0) <= 0.10
    intercept_close = abs(summary["intercept"]) <= max(2.0, 0.25 * summary["y_std"])
    bias_close = abs(summary["mean_diff"]) <= max(2.0, 0.10 * summary["y_std"])

    if summary["r2"] >= 0.95 and slope_close and intercept_close and bias_close:
        return "Close agreement. These look reasonably interchangeable for this outcome."
    if summary["r2"] >= 0.70 and summary["r"] >= 0.80:
        return "Moderately strong correlation, but not one-for-one. Use as robustness checks, not as interchangeable substitutes."
    return "Weak enough agreement that these should not be treated as interchangeable."


def print_relationship_summary(
    label: str,
    summary: dict[str, float],
    x_name: str = "ERA5",
    y_name: str = "PRISM",
) -> None:
    print("\n" + "=" * 72)
    print(f"{label}")
    print("=" * 72)
    print(f"n                 = {summary['n']:,}")
    print(f"Pearson r         = {summary['r']:.3f}")
    print(f"R^2               = {summary['r2']:.3f}")
    print(f"slope             = {summary['slope']:.3f}")
    print(f"intercept         = {summary['intercept']:.3f}")
    print(f"mean({y_name} - {x_name}) = {summary['mean_diff']:.3f} mm")
    print(f"RMSE              = {summary['rmse']:.3f} mm")
    print(f"Interpretation    = {classify_interchangeability(summary)}")


def build_site_difference_summary(monthly_compare: pd.DataFrame) -> pd.DataFrame:
    df = monthly_compare.copy()
    df["target_diff"] = df["target_area_precip_mm_prism"] - df["target_area_precip_mm_era5"]
    df["control_diff"] = df["control_area_precip_mm_prism"] - df["control_area_precip_mm_era5"]
    df["gap_era5"] = df["target_area_precip_mm_era5"] - df["control_area_precip_mm_era5"]
    df["gap_prism"] = df["target_area_precip_mm_prism"] - df["control_area_precip_mm_prism"]
    df["gap_diff"] = df["gap_prism"] - df["gap_era5"]

    site_summary = (
        df.groupby("site_id")
        .agg(
            state=("state_era5", "first"),
            project_name=("project_name_era5", "first"),
            n_months=("year_month", "count"),
            mean_target_diff_mm=("target_diff", "mean"),
            mean_control_diff_mm=("control_diff", "mean"),
            mean_gap_diff_mm=("gap_diff", "mean"),
            mean_abs_target_diff_mm=("target_diff", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_control_diff_mm=("control_diff", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_gap_diff_mm=("gap_diff", lambda s: float(np.mean(np.abs(s)))),
            rmse_target_diff_mm=("target_diff", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            rmse_control_diff_mm=("control_diff", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            rmse_gap_diff_mm=("gap_diff", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        )
        .reset_index()
    )
    return site_summary.sort_values("mean_abs_gap_diff_mm", ascending=False)


def print_top_site_gaps(site_summary: pd.DataFrame, label: str, sort_col: str, n: int = 10) -> None:
    cols = [
        "site_id",
        "state",
        "n_months",
        "mean_target_diff_mm",
        "mean_control_diff_mm",
        "mean_gap_diff_mm",
        "mean_abs_target_diff_mm",
        "mean_abs_control_diff_mm",
        "mean_abs_gap_diff_mm",
        "rmse_gap_diff_mm",
    ]
    top = site_summary.sort_values(sort_col, ascending=False)[cols].head(n).round(3)

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(top.to_string(index=False))


def add_regression_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = data[[x_col, y_col]].dropna().copy()
    x = plot_df[x_col].to_numpy()
    y = plot_df[y_col].to_numpy()
    summary = summarize_relationship(plot_df, x_col, y_col)
    line_x = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 200)
    line_y = summary["intercept"] + summary["slope"] * line_x

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        x,
        y,
        s=10,
        alpha=0.18,
        color="#2563eb",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(line_x, line_x, linestyle="--", color="#64748b", linewidth=1.2, label="45-degree line")
    ax.plot(line_x, line_y, color="#dc2626", linewidth=2, label="OLS fit")

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")

    stats_text = (
        f"n = {summary['n']:,}\n"
        f"slope = {summary['slope']:.3f}\n"
        f"intercept = {summary['intercept']:.3f}\n"
        f"R^2 = {summary['r2']:.3f}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    era5 = load_panel(ERA5_PATH)
    prism = load_panel(PRISM_PATH)

    did_era5 = compute_site_did(era5).rename(columns={"within_site_did": "era5_did"})
    did_prism = compute_site_did(prism).rename(columns={"within_site_did": "prism_did"})
    did_compare = did_era5.merge(did_prism, on="site_id", how="inner").dropna()

    monthly_compare = era5[
        [
            "site_id",
            "site_latitude",
            "site_longitude",
            "year_month",
            "target_area_precip_mm",
            "control_area_precip_mm",
            "seeded",
            "bogus_control",
            "state",
            "project_name",
        ]
    ].merge(
        prism[
            [
                "site_id",
                "site_latitude",
                "site_longitude",
                "year_month",
                "target_area_precip_mm",
                "control_area_precip_mm",
                "seeded",
                "bogus_control",
                "state",
                "project_name",
            ]
        ],
        on=["site_id", "year_month"],
        how="inner",
        suffixes=("_era5", "_prism"),
    )

    seeded_mismatch = (monthly_compare["seeded_era5"] != monthly_compare["seeded_prism"]).sum()
    site_lat_mismatch = ~np.isclose(
        monthly_compare["site_latitude_era5"],
        monthly_compare["site_latitude_prism"],
        equal_nan=True,
    )
    site_lon_mismatch = ~np.isclose(
        monthly_compare["site_longitude_era5"],
        monthly_compare["site_longitude_prism"],
        equal_nan=True,
    )
    bogus_control_mismatch = (
        monthly_compare["bogus_control_era5"] != monthly_compare["bogus_control_prism"]
    ).sum()

    print(f"ERA5 rows after bogus_control filter:  {len(era5):,}")
    print(f"PRISM rows after bogus_control filter: {len(prism):,}")
    print(f"Matched site-month rows:               {len(monthly_compare):,}")
    print(f"Matched sites for DiD comparison:      {len(did_compare):,}")
    print(f"Seeded indicator mismatches:           {seeded_mismatch:,}")
    print(f"Site latitude mismatches:              {site_lat_mismatch.sum():,}")
    print(f"Site longitude mismatches:             {site_lon_mismatch.sum():,}")
    print(f"bogus_control mismatches:              {bogus_control_mismatch:,}")

    did_summary = summarize_relationship(did_compare, "era5_did", "prism_did")
    target_summary = summarize_relationship(
        monthly_compare,
        "target_area_precip_mm_era5",
        "target_area_precip_mm_prism",
    )
    control_summary = summarize_relationship(
        monthly_compare,
        "control_area_precip_mm_era5",
        "control_area_precip_mm_prism",
    )

    print_relationship_summary("ERA5 DiD vs PRISM DiD", did_summary)
    print_relationship_summary("ERA5 target precip vs PRISM target precip", target_summary)
    print_relationship_summary("ERA5 control precip vs PRISM control precip", control_summary)

    site_summary = build_site_difference_summary(monthly_compare)
    site_summary_path = OUT_DIR / "era5_prism_site_difference_summary.csv"
    site_summary.to_csv(site_summary_path, index=False)

    print_top_site_gaps(
        site_summary,
        "Sites with biggest target-area differences | sort = mean_abs_target_diff_mm",
        "mean_abs_target_diff_mm",
    )
    print_top_site_gaps(
        site_summary,
        "Sites with biggest control-area differences | sort = mean_abs_control_diff_mm",
        "mean_abs_control_diff_mm",
    )
    print_top_site_gaps(
        site_summary,
        "Sites with biggest target-control gap differences | sort = mean_abs_gap_diff_mm",
        "mean_abs_gap_diff_mm",
    )

    did_fig = FIG_DIR / "compare_era5_vs_prism_did.png"
    target_fig = FIG_DIR / "compare_era5_vs_prism_target_precip.png"
    control_fig = FIG_DIR / "compare_era5_vs_prism_control_precip.png"

    add_regression_plot(
        data=did_compare,
        x_col="era5_did",
        y_col="prism_did",
        x_label="ERA5 within-site DiD (mm)",
        y_label="PRISM within-site DiD (mm)",
        title="ERA5 DiD vs PRISM DiD",
        output_path=did_fig,
    )
    add_regression_plot(
        data=monthly_compare,
        x_col="target_area_precip_mm_era5",
        y_col="target_area_precip_mm_prism",
        x_label="ERA5 target precipitation (mm)",
        y_label="PRISM target precipitation (mm)",
        title="ERA5 Target Precipitation vs PRISM Target Precipitation",
        output_path=target_fig,
    )
    add_regression_plot(
        data=monthly_compare,
        x_col="control_area_precip_mm_era5",
        y_col="control_area_precip_mm_prism",
        x_label="ERA5 control precipitation (mm)",
        y_label="PRISM control precipitation (mm)",
        title="ERA5 Control Precipitation vs PRISM Control Precipitation",
        output_path=control_fig,
    )

    print("\nSaved figures:")
    print(f"  {did_fig}")
    print(f"  {target_fig}")
    print(f"  {control_fig}")
    print("\nSaved site summary:")
    print(f"  {site_summary_path}")


if __name__ == "__main__":
    main()
