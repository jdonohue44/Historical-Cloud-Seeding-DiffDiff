# Historical Cloud Seeding: Difference-in-Differences

Does cloud seeding actually increase precipitation? This project estimates the effect of cloud seeding on precipiation by using historical cloud seeding operations data across the Western United States from 2000 to 2025.

## Input Data Sources

1. Cloud Seeding Activities in the United States (2000-2025) — [Zenodo](https://zenodo.org/records/16754931)
2. Precipitation data — ERA5 reanalysis

Each cloud seeding program has a **target area** (where they seeded) and a **control area** (a nearby similar region that isn't seeded).

## Data Pre-Processing: Building the Panel

A full panel of seeding × precipitation data is constructed in preparation for the difference-in-differences comparison. The panel is available at `data/input/cloud_seeding_monthly_panel.csv`

### Columns

| Column                   | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `site_id`                | Unique identifier for the cloud seeding program / site                       |
| `site_latitude`          | Latitude of the seeding site (decimal degrees)                               |
| `site_longitude`         | Longitude of the seeding site (decimal degrees)                              |
| `year_month`             | Calendar month in `YYYY-MM` format (e.g., `2015-01`)                         |
| `year`                   | Calendar year                                                                |
| `month`                  | Calendar month as an integer (1–12)                                          |
| `target_area_precip_mm`  | Total precipitation (mm) in the seeded target area for the month             |
| `control_area_precip_mm` | Total precipitation (mm) in the nearby control area for the month            |
| `target_area_seeded`     | Indicator (0/1) for whether seeding was active in the target area that month |
| `state`                  | U.S. state where the seeding project is located                              |
| `project_name`           | Name of the cloud seeding project                                            |

## Difference-in-Differences Method

We compare the precipitation gap between target and control areas during **seeded months** versus **non-seeded months**. If seeding works, the gap should be larger when seeding is active.

For each site *i* in month *t*, define the **precipitation gap**:

$$\Delta P_{it} = P^{\text{target}}_{it} - P^{\text{control}}_{it}$$

### Estimator: equal-weighted within-site DiD

For each site *i*, compute the site-level DiD on the gap:

$$\widehat{DiD}_i = \overline{\Delta P}_{i,\,\text{seeded}} \;-\; \overline{\Delta P}_{i,\,\text{unseeded}}$$

That is, the mean gap during months when seeding was active minus the mean gap during months when it was not. Because the outcome is already the target − control difference, the site-level "pre/post" comparison automatically differences out any permanent site-specific offset. Each site's single DiD scalar is the unit of analysis.

The aggregate effect is the simple mean across sites:

$$\widehat{ATT} = \tfrac{1}{N}\sum_{i=1}^{N}\widehat{DiD}_i$$

Every site counts the same regardless of how many months it seeded. Sites with zero seeded months or zero unseeded months in the panel are dropped automatically.

### Standard errors

Each site contributes exactly one $\widehat{DiD}_i$ to the aggregate, so sites are the unit of observation:

$$\widehat{SE} = \frac{\mathrm{sd}\!\left(\widehat{DiD}_i\right)}{\sqrt{N}}$$

This is clustered at the site level by construction: any within-site serial correlation is absorbed into the single site-level DiD before averaging, so it cannot inflate the aggregate variance. Inference uses a *t*-distribution with $N - 1$ degrees of freedom.

### Identifying assumption

Cloud seeding switches on and off seasonally within the same site, so we compare the same target–control pair under seeding versus without seeding across many sites and years. The key assumption is that the target−control gap would behave the same in seeded and unseeded months if seeding had no effect. Control areas are chosen by program operators to be climatologically similar to the target, which supports this.

As a diagnostic, the average target-minus-control gap during unseeded months (all sites) is +0.079 mm (95% CI: [+0.064, +0.093]). A small interval near zero supports the identifying assumption that control tracks target in the absence of seeding. To reproduce:

```bash
python analysis/check_treatment_vs_control_precip.py
```

### Data caveat: ERA5 coverage gap

The panel's ERA5 precipitation ends at Dec 2022, but the seeding mask extends into 2025. Ten sites (000, 004, 008, 019, 041, 043, 083, 084, 100, 117) have all of their seeded months in 2023+ and therefore appear with zero seeded months in the panel. These are **not** never-treated controls and **not** a data bug — they are post-2022 programs outside the ERA5 window. They are excluded from the aggregate ATT automatically.

## Running the DiD Analysis

```bash
python analysis/diff_diff.py
```

Outputs:
- Aggregate ATT with clustered-by-site SE, *t*-stat, and *p*-value
- Site-level summary table (mean gap with and without seeding, per-site DiD) saved to `data/output/site_level_seeding_gaps.csv`
- Paginated per-site precipitation time-series charts in `figures/site_precip_pageNN.png`

For interactive per-site exploration with site-level inference:

```bash
streamlit run app.py
```
