# HistoricalCloudSeedingDiffDiff

Estimates the causal effect of cloud seeding on incremental precipitation using a difference-in-differences (DiD) design for cloud seeding operations between 2000–2025 in the Western United States.

## Data Sources
1. Cloud Seeding Activities in the United States (2000–2025) [https://zenodo.org/records/16754931](https://zenodo.org/records/16754931)
2. Precipitation Data []()

## Causal Methodology

### Panel Structure

We construct a balanced monthly panel at the site level spanning January 2000 through December 2025 (up to 312 months per site). Each observation records:

| Column | Description |
|--------|-------------|
| `site_id` | Cloud seeding program identifier |
| `year_month` | Calendar year-month (e.g., `2015-01`) |
| `target_area_precip` | Total precipitation (mm) in the designated target area |
| `control_area_precip` | Total precipitation (mm) in the designated control area |
| `target_area_seeded` | Binary: whether seeding operations were active at this site in this month |

Control areas are pre-specified by program operators as climatologically similar nearby regions not targeted by seeding. Monthly resolution captures the seasonal structure of seeding operations: most Western mountain programs seed during winter months (roughly November–April) for snowpack augmentation and spring runoff, while Southern Plains programs (e.g., Texas) seed during summer months for rainfall enhancement. The `target_area_seeded` indicator is `TRUE` only during the 3–4 months per year when a program is actively seeding.

### Estimating Equation

For each site *i* in month *t*, define the **precipitation gap**:

$$\Delta P_{it} = P^{\text{target}}_{it} - P^{\text{control}}_{it}$$

The primary specification estimates:

$$\Delta P_{it} = \alpha_i + \gamma_t + \delta \cdot \mathbf{1}[\text{Seeded}_{it}] + \varepsilon_{it}$$

where:
- $\alpha_i$ = **site fixed effects**, absorbing time-invariant differences in the target–control precipitation gap (elevation, orographic effects, baseline climate)
- $\gamma_t$ = **year-month fixed effects**, absorbing precipitation shocks common to all sites in a given month (ENSO phases, regional drought, shared climate variability)
- $\mathbf{1}[\text{Seeded}_{it}]$ = indicator equal to 1 when site *i* is actively being seeded in month *t*
- $\delta$ = **average treatment effect on the treated (ATT)** — the estimated incremental precipitation attributable to cloud seeding

This is a two-way fixed effects (TWFE) difference-in-differences design applied to the target–control precipitation gap. Because seeding is seasonal (active for only a few months per year and absent in the remaining months), treatment status varies within site over time rather than switching on permanently. This time-varying treatment structure avoids the well-documented biases of TWFE under staggered adoption with absorbing treatment (Goodman-Bacon, 2021; de Chaisemartin & D'Haultfoeuille, 2020), and allows identification from a straightforward within-site comparison of seeded versus non-seeded months.

### Identification

Causal identification of $\delta$ requires the **parallel trends assumption**: absent seeding, the target–control precipitation gap would have evolved identically across seeded and non-seeded periods.

Two features of the setting support this assumption:

1. **Pre-specified control areas.** Control areas are designated by program operators as climatologically similar to the target, providing geographic comparability by construction.

2. **Staggered program initiation.** Of 211 sites in the dataset, 130 initiate seeding within the 2000–2025 window (treatment start years range from 2001 to 2022). For these "DiD-eligible" sites, the pre-treatment months — same calendar months, same site, before any seeding began — provide a direct test of whether the target–control gap was stable prior to intervention.

The remaining 81 sites were seeded throughout the observation period and lack an untreated pre-period. These contribute only **descriptive comparisons** (mean target–control differences), not causal estimates.

### Parallel Trends Assessment

For DiD-eligible sites, we assess the parallel trends assumption through:

1. **Pre-treatment event study.** Using `MultiPeriodDiD`, we estimate dynamic treatment effects relative to the first seeding season, plotting the target–control gap in each pre-treatment period. Pre-treatment coefficients statistically indistinguishable from zero support the parallel trends assumption.

2. **Slope-based and distributional tests.** We test for differential pre-treatment trends using both a standard slope test (`check_parallel_trends`) and a Wasserstein-distance permutation test (`check_parallel_trends_robust`), which is robust to non-normality and outliers.

### Inference

Standard errors are **clustered at the site level** to account for within-site serial correlation (Bertrand, Duflo, & Mullainathan, 2004). With 130+ DiD-eligible clusters, asymptotic cluster-robust inference is reliable. As a robustness check, we also report **wild cluster bootstrap** p-values.

### Robustness and Diagnostics

1. **Site-specific seasonality controls.** Because seeding is seasonal and the target–control gap may exhibit its own seasonal pattern (e.g., orographic effects stronger in winter), we test sensitivity to adding site × calendar-month fixed effects, which absorb site-specific seasonal precipitation differences and ensure identification comes only from across-year variation within the same calendar months.

2. **Placebo timing tests.** We re-estimate with fake treatment start dates (shifting program initiation 2–3 years earlier) to confirm effects appear only at the true treatment date.

3. **Leave-one-out analysis.** We re-estimate dropping one site at a time to assess sensitivity to influential individual sites.

4. **Descriptive comparisons for always-treated sites.** For sites seeded throughout the observation window, we report mean target–control precipitation differences as descriptive contrasts, clearly distinguished from causal estimates.

### Implementation

Estimation uses the [`diff-diff`](https://pypi.org/project/diff-diff/) Python library:

- **Primary specification**: `TwoWayFixedEffects` with cluster-robust standard errors
- **Event study**: `MultiPeriodDiD` for dynamic treatment effect visualization
- **Parallel trends**: `check_parallel_trends` and `check_parallel_trends_robust`
- **Placebo tests**: Built-in placebo diagnostic tools
- **Bootstrap inference**: Wild cluster bootstrap for robustness

## Context

We estimate the causal effect of cloud seeding on precipitation using a difference-in-differences (DiD) design for seeding programs that initiate during the 2000–2025 observation window. Restricting attention to these programs provides an untreated pre-intervention period within the available data, enabling credible causal estimation using DiD. 

For each site meeting this criterion, we define the treatment start as the first year in which seeding activity is reported and compare changes in precipitation before and after treatment between the seeded target area and a nearby unseeded control area. This approach measures relative changes over time, differencing out time-invariant spatial heterogeneity under the assumption of parallel trends. To assess the plausibility of the parallel trends assumption, we examine differential precipitation trends in the years immediately preceding seeding to ensure that treated and control areas exhibit similar precipitation dynamics prior to treatment.

For sites that are seeded throughout the 2000–2025 record and therefore lack a clearly defined untreated pre-period, DiD estimation is not feasible. For these locations, we report precipitation differences between the target and control areas as descriptive comparisons of climatological signatures rather than causal effect estimates. These descriptive contrasts should be interpreted more cautiously, as treatment is endogenous and the counterfactual untreated outcomes are not observed.

All causal estimates are therefore identified from sites with observed seeding initiation within the study window, while long-running programs contribute to the analysis through descriptive comparison.