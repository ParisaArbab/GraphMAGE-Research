# Multi-horizon Chicago follow-up: preregistered screening protocol

This is the **second exploratory screening experiment** based on the six existing
GraphMAGE-inspired proxy variants. It is not the original paper reproduction.

## Exact experiment configuration

- Data source: public Chicago mosquito test records, using the corrected coordinate parser
- Trap count: 48, chosen using only training years
- Historical input: 4 weeks
- Forecast horizons: **1, 3, and 7 weeks**
- Methods: baseline, adaptive graph, temporal GRU, focal loss, adaptive + temporal, all three
- Seeds: **42, 43, 44, 45, 46**, matched across every configuration
- Maximum epochs: **20**; early stopping and the decision threshold use validation only
- Time split: train 2007–2022, validation 2023, exploratory test 2024–2025
- **90 model fits:** 6 models x 5 seeds x 3 horizons
- Compute: GitHub Actions CPU runner, same model settings as the previous pilot

## Decisions fixed before this run

The model ranking is based only on a **0.6 validation ROC-AUC +
0.4 validation PR-AUC** score averaged equally across the three horizons
and the five seeds.

A candidate qualifies for **independent replication** only if:
1. It is the top method by the prespecified validation score.
2. Its paired, across-horizon validation ROC-AUC gain versus baseline is **at least 0.01**.
3. Its paired, across-horizon validation PR-AUC gain versus baseline is **at least 0.01**.
4. Both validation gains are positive in **at least 2 of 3** horizons.

The 95% bootstrap intervals resample the five *matched random seeds*.
They measure sensitivity to initialization/training randomness, not uncertainty
over independent years, mosquito traps, regions, or epidemics. Do not claim
statistical significance about generalization from these intervals alone.

**Important:** Because the 2024–2025 test years have already been inspected
in the prior pilot, they are **not untouched independent evidence**. They
will be shown descriptively only and will not influence the winner. To
validate a method scientifically, use newly held-out data/years or an
independent geographical cohort after model selection.

## Run manually again

Go to the repository's
[GraphMAGE pilot workflow](../.github/workflows/graphmage-pilot.yml)
in GitHub Actions, select Run workflow, and choose
`experiment = multi_horizon`.

The job uploads `graphmage-multihorizon-results` containing:

- `REPORT.md`: plain-English research report
- `overall_summary.csv`, `horizon_summary.csv`: comparison tables
- `all_runs.csv`: 90 rows, one per method/seed/horizon
- `paired_deltas.csv`, `paired_seed_bootstrap.csv`: matched-seed improvement estimates
- `decision.json`: validation selection and gate status
- `horizon_comparison.png`: horizon comparison plot
- `horizon_1/`, `horizon_3/`, `horizon_7/`: source logs and individual reports

## Claim limits

This model is a small **proxy** of GraphMAGE, not a faithful reimplementation;
Chicago-only mode lacks weather and land-use information. Parameter counts
are not matched across variants. Results can identify *engineering candidates*
for further study, not demonstrate improvement over the published paper.

GitHub Actions automatically launches the **first** multi-horizon experiment
from the tagged commit that introduced this protocol.
