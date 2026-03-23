from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
CSV_FILE = "cloud_seeding_did.csv"
DEMO_IF_NO_PRECIP = False

DATA_PATH = ROOT / "data" / CSV_FILE
df = pd.read_csv(DATA_PATH)
df = df[df["did_eligible"]].copy()
if df["precip_target_area"].isna().all():
    if not DEMO_IF_NO_PRECIP:
        print("No precip data; fill precip_target_area and precip_control_area, or set DEMO_IF_NO_PRECIP = True.")
        exit(0)
    df["precip_target_area"] = df.groupby("site_id")["year"].transform(lambda x: x.rank())
    df["precip_control_area"] = df.groupby("site_id")["year"].transform(lambda x: x.rank() * 0.95)
    demo = True
else:
    demo = False
df = df.dropna(subset=["precip_target_area", "precip_control_area"])
if df.empty:
    print("No precip data.")
    exit(0)

df["rel_year"] = df["year"] - df["treatment_start_year"]
target = df.groupby("rel_year")["precip_target_area"].mean().rename("Target")
control = df.groupby("rel_year")["precip_control_area"].mean().rename("Control")
plot_df = pd.concat([target, control], axis=1).sort_index()

FIG_DIR.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=0)
ax.axvspan(plot_df.index.min(), 0, alpha=0.08, color="gray")
ax.axvspan(0, plot_df.index.max(), alpha=0.08, color="steelblue")
ax.plot(plot_df.index, plot_df["Target"], color="C0", marker="o", ms=3, label="Target area")
ax.plot(plot_df.index, plot_df["Control"], color="C1", marker="s", ms=3, label="Control area")
ymin, ymax = plot_df.min().min(), plot_df.max().max()
yr = ymax - ymin or 1
ax.text(-abs(plot_df.index.min()) / 2, ymax - 0.03 * yr, "Pre", fontsize=10, color="gray", ha="center")
ax.text(abs(plot_df.index.max()) / 2, ymax - 0.03 * yr, "Post", fontsize=10, color="steelblue", ha="center")
ax.set_xlabel("Years relative to treatment start")
ax.set_ylabel("Precipitation")
ax.legend(loc="best")
ax.set_title("Pre- and post-treatment trends: target vs control")
ax.grid(True, alpha=0.3)
fig.tight_layout()
OUT_FILE = "did_trends_demo.png" if demo else "did_trends.png"
out = FIG_DIR / OUT_FILE
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")
