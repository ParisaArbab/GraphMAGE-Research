# Why temporal + adaptive graph changes with forecast horizon

## Scientific scope
This is an automated, limited GraphMAGE-inspired **proxy** experiment.
It does not reproduce the paper's official architecture, original features, or statewide Illinois test population.

## Fixed data design
- Chicago traps selected using only years up to 2023; the new test year is **2026**.
- Training 2007–2023, validation 2024–2025, test 2026.
- Validation years were examined as held-out years in earlier experiments, so this experiment is an extension, not a completely independent discovery study.
- 2026 was not used in previous pilots or in configuration/threshold selection.
- Historical temperature, rainfall and humidity are based on Chicago-center ERA5 reanalysis from Open-Meteo, using the seven days strictly **before** the forecast week.
- Weather is city-level and identical at all traps, NOT climate-aware graph edges.
- 3 seeds; 20 maximum epochs; histories 2/4/8; edge mix alpha 0, 0.5, or 1; weather on/off; horizons 1/3/7 weeks.

## Validation-only selected configurations

| Horizon | Selected configuration | Gain ROC-AUC over fixed temporal baseline | Gain PR-AUC | Promotion gate |
|---:|---|---:|---:|---|
| 1 weeks | hist8_edge0p5_weather1 | +0.0089 | +0.0250 | fail |
| 3 weeks | hist8_edge0p0_weather1 | +0.0093 | +0.0310 | fail |
| 7 weeks | hist8_edge0p0_weather0 | +0.0017 | +0.0023 | fail |

Selection is from **validation only**: 0.6 × ROC-AUC + 0.4 × PR-AUC. A method must improve both metrics by 0.01 over the fixed temporal baseline.

## New 2026 held-out results, compared with frozen baselines

| Horizon | Frozen model | ROC-AUC | PR-AUC | F1 | Sensitivity | Samples |
|---:|---|---:|---:|---:|---:|---:|
| 1 weeks | chosen | 0.778 | 0.361 | 0.436 | 0.631 | 736 |
| 1 weeks | fixed_temporal | 0.758 | 0.330 | 0.415 | 0.618 | 736 |
| 1 weeks | original_proxy | 0.744 | 0.304 | 0.403 | 0.629 | 736 |
| 1 weeks | prior_adaptive_temporal | 0.762 | 0.333 | 0.403 | 0.556 | 736 |
| 3 weeks | chosen | 0.752 | 0.325 | 0.444 | 0.753 | 736 |
| 3 weeks | fixed_temporal | 0.740 | 0.303 | 0.416 | 0.691 | 736 |
| 3 weeks | original_proxy | 0.738 | 0.298 | 0.416 | 0.710 | 736 |
| 3 weeks | prior_adaptive_temporal | 0.745 | 0.305 | 0.445 | 0.767 | 736 |
| 7 weeks | chosen | 0.706 | 0.308 | 0.404 | 0.628 | 655 |
| 7 weeks | fixed_temporal | 0.701 | 0.304 | 0.411 | 0.683 | 655 |
| 7 weeks | original_proxy | 0.697 | 0.293 | 0.405 | 0.689 | 655 |
| 7 weeks | prior_adaptive_temporal | 0.693 | 0.297 | 0.392 | 0.617 | 655 |

## Interpretation and safeguards

All 2026 evaluations were run **after** validation-only selection. No 2026 metric influenced model choice, checkpoint selection, or decision thresholds.
We include both the prior proxy baseline (no GRU, fixed graph, no weather) and the fixed-graph GRU baseline to separate temporal gains from other changes.
Differences in parameter counts and 3-seed randomness limit conclusions.
No climate information varies between traps, so this is a **weather-covariate ablation**, not a location-dependent climate-graph claim.
The 2026 season provides a new held-out year, but a single year cannot show general robustness across epidemics, regions or years.

## Next confirmatory work

1. Freeze a single hypothesis before a *new independent season* or independent region.
2. Compare model-size-matched variants and original GraphMAGE code.
3. Obtain spatially varying weather and land-use features and test unseen rural traps.
4. Add uncertainty over trap clusters and years, not only random seeds.

Files: validation_search_runs.csv, validation_summary.csv, decision.json, holdout_2026.csv, metadata.json, validation_method_comparison.png, holdout_2026_comparison.png, run.log.