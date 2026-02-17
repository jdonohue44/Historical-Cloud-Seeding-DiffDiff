# HistoricalCloudSeedingDiffDiff

Estimates the causal effect of cloud seeding on precipitation using a difference-in-differences (DiD) design for programs initiating between 2000–2025, leveraging pre-treatment periods for credible causal inference and parallel trends diagnostics.

## Context

We estimate the causal effect of cloud seeding on precipitation using a difference-in-differences (DiD) design for seeding programs that initiate during the 2000–2025 observation window. Restricting attention to these programs provides an untreated pre-intervention period within the available data, enabling credible causal estimation using DiD. 

For each site meeting this criterion, we define the treatment start as the first year in which seeding activity is reported and compare changes in precipitation before and after treatment between the seeded target area and a nearby unseeded control area. This approach measures relative changes over time, differencing out time-invariant spatial heterogeneity under the assumption of parallel trends. To assess the plausibility of the parallel trends assumption, we examine differential precipitation trends in the years immediately preceding seeding to ensure that treated and control areas exhibit similar precipitation dynamics prior to treatment.

For sites that are seeded throughout the 2000–2025 record and therefore lack a clearly defined untreated pre-period, DiD estimation is not feasible. For these locations, we report precipitation differences between the target and control areas as descriptive comparisons of climatological signatures rather than causal effect estimates. These descriptive contrasts should be interpreted more cautiously, as treatment is endogenous and the counterfactual untreated outcomes are not observed.

All causal estimates are therefore identified from sites with observed seeding initiation within the study window, while long-running programs contribute to the analysis through descriptive comparison.

## Code

The pipeline uses a **single processed CSV** (produced offline): `data/cloud_seeding_did.csv`. Schema: `site_id`, `year`, `treatment_start_year`, `did_eligible`, `target_area`, `control_area`, `precip_target_area`, `precip_control_area`.

## Steps to Run

1. **Place your processed CSV** at `data/cloud_seeding_did.csv`. (If you have an existing `cloud_seeding_did_processed.csv` or similar, rename or copy it to `cloud_seeding_did.csv`.)

2. **Run DiD**
   ```bash
   python3 scripts/DiD.py
   ```

3. **Plot pre/post trends**
   ```bash
   python3 scripts/plot_did_trends.py
   ```
   Saves `figures/did_trends.png`. Set `DEMO_IF_NO_PRECIP = True` at top of script to use dummy precip when the CSV has no precip.
