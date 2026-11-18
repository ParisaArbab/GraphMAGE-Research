# GraphMAGE third experiment: history, graph mixing, and archived weather

## Aim and status

Investigate why the adaptive graph + temporal GRU combination showed its strongest
validation gain at one-week forecasting, without assuming this improvement generalizes.

This protocol was fixed before opening the **new 2026 holdout**. The earlier
multi-horizon pilot's 2024–2025 test results are not independent confirmation
and are now assigned to validation only for this diagnostic.

## Design

- **Dataset:** City of Chicago WNV mosquito surveillance, 2007–2026
  (https://data.cityofchicago.org/d/jqe8-8r6s).
- **Trap selection:** 48 traps, chosen using pre-2024 years only.
- **Train:** 2007–2023; **validation:** 2024–2025; **new test:** 2026.
- **Horizon:** 1, 3, and 7 weeks.
- **History length:** 2, 4, or 8 weeks.
- **Graph edge mixture:** 0 = fixed distance kNN, 0.5 = fixed/learned blend,
  1 = learned normalized attention only, constrained to the same geographic kNN.
- **Weather:** on vs off. On uses ERA5 historical reanalysis from
  https://open-meteo.com/en/docs/historical-weather-api for the Chicago city center.
  The features are temperature (7-day mean), precipitation (7-day sum), and humidity
  (7-day mean) **strictly before** each prediction week's start.
- **Important:** Weather is the *same across traps* and does not introduce
  location-dependent climate connections or land-cover covariates.
- **Seeds:** 42, 43, 44; 20 epoch maximum and patience 5.
- **Validation search:** 3 horizons x 3 histories x 3 graph mixtures x 2 weather
  settings x 3 seeds = 162 fits.
- **Reference:** 3 horizons x 3 seeds = 9 old-proxy fits (no GRU).
- **Frozen holdout:** 4 control configurations per horizon x 3 seeds
  = 36 additional fits, including the chosen model and both baselines.
- **Total model fits:** 207.

## Prespecified selection and decision

Models are ranked per horizon using **0.6 × validation ROC-AUC +
0.4 × validation PR-AUC**, averaged over the same three random seeds.
Selection, early stopping, and classification thresholds use **validation only**.

For each horizon, promote an improved configuration only when its validation
ROC-AUC and PR-AUC are **each at least 0.01 higher** than the fixed-graph,
4-week-history **GRU** baseline.

After these configuration choices are frozen, run the same selected models
and controls on **2026**, which prior experiments did not include. Compare
the frozen model to:

1. Fixed graph + GRU + four weeks' history + no weather.
2. Original no-GRU proxy baseline + fixed graph + no weather.
3. Prior adaptive-graph + GRU (mix 0.5), four weeks' history + no weather.

Do not tune model architecture or decisions after reading 2026 scores. A single
unseen year and three seeds cannot establish a generally reliable result.

## Output

GitHub Actions uploads **graphmage-climate2026-results**, with:
- validation_search_runs.csv (171 per-seed validation runs including old baselines)
- validation_summary.csv and decision.json (frozen per-horizon selections)
- holdout_2026.csv (36 per-seed evaluations AFTER selection)
- REPORT.md, run.log and diagnostic charts
- metadata.json (data and weather sources, alignment, settings, weather SHA-256)

## Scientific limitations

This is a lightweight **GraphMAGE-inspired proxy**, not the paper authors'
official Illinois GraphMAGE model. The three weather features represent a
single citywide grid cell, not trap-level weather. Model parameter counts are
not matched. The 2026 season may be partially recorded; report its actual
tested sample counts. This work tests a hypothesis and cannot prove improvement
over the original 2025 paper.

First workflow launch is tied to the protocol commit's
`[run-climate2026]` tag.
