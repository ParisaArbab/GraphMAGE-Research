# Third experiment: Horizon-dependent weather and graph learning

**Date:** 2026-11-18  
**Workflow:** [GitHub Actions #44366420813](https://github.com/ParisaArbab/GraphMAGE-Research/actions/runs/44366420813)  
**Full artifact:** [graphmage-climate2026-results](https://github.com/ParisaArbab/GraphMAGE-Research/actions/runs/44366420813/artifacts/14456772761)  
**Protocol:** [docs/CLIMATE_2026_PROTOCOL.md](../docs/CLIMATE_2026_PROTOCOL.md)

## Executive finding

The selected models improved held-out **2026 ROC-AUC and PR-AUC at 1 and 3 weeks**
against both the fixed-graph temporal baseline and the original no-GRU proxy.
These are **exploratory, hypothesis-generating improvements**, NOT a demonstrated
improvement over the published GraphMAGE paper.

**No model passed the predefined validation gate** (+0.01 on both ROC-AUC and
PR-AUC). The measured changes in validation AUC were +0.0089 (1 week),
+0.0093 (3 weeks), and +0.0017 (7 weeks).

## Reproducible evaluation

- Data: Chicago public mosquito test results. 48 traps chosen from pre-2024 data.
- Training 2007–2023, validation 2024–2025, **new holdout 2026**.
- Citywide historical ERA5 temperature, precipitation, humidity, using seven
  days strictly before forecast week (Open-Meteo). This weather **does not vary by trap**.
- 3 random seeds, 20 maximum epochs, 1/3/7-week target horizons.
- Compared 2/4/8 weeks of temporal history, graph attention mixing 0/0.5/1,
  and weather off/on.
- Total **207 model fits**: 162 candidate fits, 9 no-GRU reference fits,
  36 post-selection held-out 2026 evaluations.
- The **2026 test year was not used in earlier pilots**. All selections,
  checkpoints, and thresholds were fixed using validation before 2026 evaluation.

## Frozen selection using validation only

| Horizon | Validation-selected method | Val AUC gain | Val PR-AUC gain | Gate |
|---|---|---:|---:|---|
| 1 week | GRU 8-week history + 50/50 adaptive graph + citywide weather | +0.0089 | +0.0250 | Fail |
| 3 weeks | GRU 8-week history + fixed graph + citywide weather | +0.0093 | +0.0310 | Fail |
| 7 weeks | GRU 8-week history + fixed graph + no weather | +0.0017 | +0.0023 | Fail |

These gains are against a *fixed-graph GRU with 4-week history and no weather*.
ROC-AUC improved by less than the required +0.01 at all three horizons.

## New 2026 held-out results

| Horizon | Model | ROC-AUC | PR-AUC | F1 | Sensitivity | Brier |
|---|---|---:|---:|---:|---:|---:|
| 1 week | Frozen selected | **0.778** | **0.361** | **0.436** | 0.631 | 0.233 |
| 1 week | Fixed temporal baseline | 0.758 | 0.330 | 0.415 | 0.618 | **0.210** |
| 1 week | Original proxy, no GRU | 0.744 | 0.304 | 0.403 | 0.629 | 0.212 |
| 3 weeks | Frozen selected | **0.752** | **0.325** | **0.444** | 0.753 | 0.227 |
| 3 weeks | Fixed temporal baseline | 0.740 | 0.303 | 0.416 | 0.691 | 0.211 |
| 3 weeks | Original proxy, no GRU | 0.738 | 0.298 | 0.416 | 0.710 | **0.205** |
| 7 weeks | Frozen selected | **0.706** | **0.308** | 0.404 | 0.628 | **0.194** |
| 7 weeks | Fixed temporal baseline | 0.701 | 0.304 | **0.411** | **0.683** | 0.198 |
| 7 weeks | Original proxy, no GRU | 0.697 | 0.293 | 0.405 | 0.689 | 0.201 |

Mean of three matched seeds. 2026 has 736 labeled test samples at weeks 1 and 3,
and 655 at week 7, with a positive rate of about 16.7% and 18.6%, respectively.
Samples share years and traps and are not statistically independent.

### Why short horizons may benefit

The following *controlled validation ablations* hold the other components fixed:

| Horizon | Single modification | Change in validation AUC | Change in validation PR-AUC |
|---|---|---:|---:|
| 1 | Add weather with 4-week history and 50/50 graph | +0.0044 | +0.0253 |
| 1 | Extend history from 4 to 8 weeks, with weather and 50/50 graph | +0.0031 | +0.0138 |
| 1 | Add 50/50 adaptive graph to an 8-week/weather model | +0.0145 | +0.0369 |
| 3 | Add weather to fixed graph with 8-week history | +0.0103 | +0.0321 |
| 3 | Extend history from 4 to 8 weeks with weather and fixed graph | +0.0065 | +0.0274 |
| 3 | Add 50/50 adaptive graph to 8-week/weather model | -0.0045 | -0.0121 |
| 7 | Add weather to fixed graph with 8-week history | -0.0149 | -0.0325 |
| 7 | Add 50/50 adaptive graph to 8-week/no-weather model | -0.0087 | -0.0125 |

This suggests an interesting **horizon-dependent pattern**: the graph and weather
features help most near the forecast date, while their useful information may
fade or become noisy at 7 weeks. This explanation is a hypothesis, not a
causal conclusion.

### Important tradeoff: probability calibration

For the selected 1-week model, Brier error **increased** from 0.210 to 0.233
versus the fixed temporal baseline. At 3 weeks, it increased from 0.211 to
0.227. Thus, better infection ranking does not imply better calibrated
infection probabilities. This matters for public-health risk prediction.

## Research decision

**No candidate promoted for a publication improvement claim.** The 2026
results are an encouraging independent-season signal at 1 and 3 weeks, but the
precommitted validation gate failed. No thresholds were changed after seeing results.

A focused next hypothesis is to learn horizon-specific graph mixing while
regularizing edge weights toward the geographic graph, alongside calibration
on validation data. It must then be frozen and evaluated on a *different*
untouched geographic cohort or a future season, rather than tuned against 2026.

The original Illinois GraphMAGE model, weather at trap level, land-cover
features, and statistical uncertainty across years/areas are still needed.

## Files

- [Automated report](../results/climate2026/REPORT.md)
- [All 171 validation fits](../results/climate2026/validation_search_runs.csv)
- [Mean validation metrics](../results/climate2026/validation_summary.csv)
- [All 36 2026 evaluation fits](../results/climate2026/holdout_2026.csv)
- [Frozen decisions](../results/climate2026/decision.json)
- [Provenance](../results/climate2026/metadata.json)
- [Complete GitHub Actions artifact](https://github.com/ParisaArbab/GraphMAGE-Research/actions/runs/44366420813/artifacts/14456772761), includes plots and logs
