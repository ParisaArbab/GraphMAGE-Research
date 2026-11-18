# Automated GraphMAGE Pilot Report

**Data source:** synthetic (programmatically generated; not clinical evidence)
**Years:** train [2018, 2019, 2020], validation [2021], test [2022, 2023]
**Horizon:** 3 week(s); **traps:** 48; **seeds:** 2

**Important:** These experiments use a lightweight GraphMAGE-inspired proxy, not the authors' official implementation. Real-data results, if any, should not be compared directly with published paper metrics. Synthetic-data runs only show that the experiment pipeline works.

## Screening result
**Decision:** INCONCLUSIVE: no validated improvement above the pilot screening threshold.
The **validation-selected** candidate is `all_three` (validation 0.6 ROC-AUC + 0.4 PR-AUC).
Its held-out test ROC-AUC is **0.708** vs **0.708** for the proxy baseline; PR-AUC is **0.502** vs **0.499**.

A better result on a small pilot is a **hypothesis for replication**, not evidence of a guaranteed improvement.

## Full comparison (mean across seeds)

| Variant | Val AUC | Val PR-AUC | Test AUC | Test PR-AUC | Test F1 | Test Sensitivity | Test Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_three | 0.718 | 0.551 | 0.708 | 0.502 | 0.527 | 0.661 | 0.227 |
| temporal | 0.716 | 0.551 | 0.701 | 0.490 | 0.517 | 0.617 | 0.221 |
| adaptive_temporal | 0.717 | 0.549 | 0.704 | 0.495 | 0.539 | 0.764 | 0.220 |
| focal | 0.716 | 0.540 | 0.705 | 0.496 | 0.533 | 0.687 | 0.229 |
| baseline | 0.716 | 0.540 | 0.708 | 0.499 | 0.529 | 0.668 | 0.217 |
| adaptive | 0.715 | 0.539 | 0.708 | 0.503 | 0.519 | 0.629 | 0.219 |

## Recommended next steps
1. Repeat this screening with 5-20 seeds on real Illinois data and multiple forecast horizons.
2. Replace the proxy with the published authors' GraphMAGE code when available, preserving identical data and splits.
3. Add actual weather and land-cover covariates; Chicago-only mode does not test climate awareness.
4. Test unseen rural traps and uncertainty intervals before claiming a real improvement.
5. Keep the test years untouched during architecture/hyperparameter selection.

## Leakage safeguards
- Future labels are never placed into input features.
- Earlier observed outcomes may appear as lagged history, but the target week outcome is masked.
- Locations are selected from training years only in Chicago mode.
- Model checkpoints, thresholds, and preferred variant use validation observations only.
- Test data are used for reporting after selection, not for threshold tuning.