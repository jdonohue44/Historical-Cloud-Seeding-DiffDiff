# Historical Cloud Seeding: Difference-in-Differences

Does cloud seeding actually increase precipitation? This project estimates the effect of cloud seeding on precipiation by using historical cloud seeding operations data across the Western United States from 2000 to 2025.

## Data Sources
1. Cloud Seeding Activities in the United States (2000-2025) — [Zenodo](https://zenodo.org/records/16754931)
2. Precipitation data — ERA5 reanalysis

Each cloud seeding program has a **target area** (where they seeded) and a **control area** (a nearby similar region that isn't seeded). Seeding is seasonal, where programs operate for a few months each year and are inactive the rest. For example, in mountainous ski areas like Alta and Snowbird, glaciogenic cloud seeding is used to increse snow during winter months. 

## Data Processing: Building the Panel
A full panel of seeding x precipitation data is constructed in preparation for the difference-in-differences comparison.

| Column | Description |
|--------|-------------|
| `site_id` | Cloud seeding program identifier |
| `year_month` | Calendar month (e.g., `2015-01`) |
| `target_area_precip_mm` | Precipitation in the seeded target area |
| `control_area_precip_mm` | Precipitation in the nearby control area |
| `target_area_seeded` | Whether seeding was active that month |

## Difference-in-Differences Method

We compare the precipitation gap between target and control areas during **seeded months** versus **non-seeded months**. If seeding works, the gap should be larger when seeding is active.

For each site *i* in month *t*, define the **precipitation gap**:

$$\Delta P_{it} = P^{\text{target}}_{it} - P^{\text{control}}_{it}$$

Then estimate:

$$\Delta P_{it} = \alpha_i + \gamma_t + \delta \cdot \text{Seeded}_{it} + \varepsilon_{it}$$

- **Site fixed effects** ($\alpha_i$) account for permanent differences between sites (elevation, geography, baseline climate).
- **Year-month fixed effects** ($\gamma_t$) account for weather shocks that affect all sites in a given month (El Nino, drought, etc.).
- **$\delta$** is the estimate: how much extra precipitation (in mm) falls in the target area relative to the control area when seeding is active.

Standard errors are clustered at the site level to account for correlation within sites over time.

*Why is this valid? What assumptions are we making?*

Cloud seeding switches on and off seasonally within the same site. This means we're comparing the same target-control pair under seeding versus without seeding, across many sites and years. The site fixed effects remove any permanent geographic differences, and the time fixed effects remove shared weather patterns.

The key assumption is that the target-control gap would behave the same in seeded and non-seeded months if seeding had no effect. The fact that control areas are chosen by program operators to be climatologically similar to the target supports this.

## Running the Analysis

```bash
python analysis/twfe_cloud_seeding.py
```

Outputs:
- Regression results with the estimated effect of seeding
- Site-level summary table (mean gap with and without seeding)
- Per-site time series charts saved to `figures/`
- Streamlit dashboard to see site-level precipitation trends and the gap between target and control precipitation
