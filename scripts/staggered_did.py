from pathlib import Path
import pandas as pd
from linearmodels.panel import PanelOLS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cloud_seeding_did.csv"
df = pd.read_csv(DATA_PATH)
df = df[df["did_eligible"]].dropna(subset=["precip_target_area", "precip_control_area"])
if df.empty:
    print("No precip data; fill precip_target_area and precip_control_area in cloud_seeding_did.csv.")
    exit(0)
target = df[["site_id", "year", "treatment_start_year", "precip_target_area"]].copy()
target = target.rename(columns={"precip_target_area": "precip"})
target["treated"] = 1
control = df[["site_id", "year", "treatment_start_year", "precip_control_area"]].copy()
control = control.rename(columns={"precip_control_area": "precip"})
control["treated"] = 0
long = pd.concat([target, control], ignore_index=True)
long["post"] = (long["year"] >= long["treatment_start_year"]).astype(int)
long["did"] = long["treated"] * long["post"]
long["entity"] = long["site_id"] + "_" + long["treated"].astype(str)
long = long.set_index(["entity", "year"]).sort_index()
exog = long[["did"]]
mod = PanelOLS(long["precip"], exog, entity_effects=True, time_effects=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)
print(res.summary)
