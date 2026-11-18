# Chicago Multi-Horizon GraphMAGE Pilot

**Status:** INCONCLUSIVE, no promotion by the validation rule

**Research scope:** GraphMAGE-inspired proxy, NOT the original paper's model. Chicago surveillance data have no weather or land-use covariates.

**Design:** 5 matched seeds, horizons [1, 3, 7], 20 epoch maximum per run, 48 traps. Each method uses the same chronological train/validation/test years.
**Years:** Train 2007-2022; validation 2023; exploratory test 2024-2025.

**Warning:** The 2024-2025 results have already been inspected in an earlier experiment. They cannot be treated as fresh confirmation or used to choose the method.

## Overall mean across horizons and seeds (validation ranked)

| Method | Val AUC | Val PR-AUC | Test AUC (exploratory) | Test PR-AUC (exploratory) | Test F1 |
|---|---:|---:|---:|---:|---:|
| adaptive_temporal | 0.809 | 0.480 | 0.781 | 0.451 | 0.523 |
| all_three | 0.809 | 0.479 | 0.782 | 0.454 | 0.516 |
| baseline | 0.809 | 0.475 | 0.773 | 0.436 | 0.507 |
| temporal | 0.807 | 0.476 | 0.781 | 0.451 | 0.518 |
| adaptive | 0.808 | 0.474 | 0.776 | 0.443 | 0.507 |
| focal | 0.808 | 0.474 | 0.774 | 0.435 | 0.508 |

## Per-horizon comparison (means over matched seeds)

| Weeks | Method | Val AUC | Val PR-AUC | Test AUC | Test PR-AUC | Test F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | adaptive | 0.805 | 0.471 | 0.790 | 0.444 | 0.509 |
| 1 | adaptive_temporal | 0.812 | 0.494 | 0.812 | 0.491 | 0.546 |
| 1 | all_three | 0.813 | 0.489 | 0.808 | 0.485 | 0.534 |
| 1 | baseline | 0.805 | 0.472 | 0.787 | 0.439 | 0.509 |
| 1 | focal | 0.804 | 0.471 | 0.793 | 0.454 | 0.520 |
| 1 | temporal | 0.806 | 0.477 | 0.791 | 0.450 | 0.513 |
| 3 | adaptive | 0.817 | 0.477 | 0.780 | 0.429 | 0.504 |
| 3 | adaptive_temporal | 0.817 | 0.475 | 0.782 | 0.425 | 0.513 |
| 3 | all_three | 0.817 | 0.477 | 0.782 | 0.432 | 0.503 |
| 3 | baseline | 0.818 | 0.476 | 0.781 | 0.426 | 0.506 |
| 3 | focal | 0.817 | 0.475 | 0.775 | 0.412 | 0.499 |
| 3 | temporal | 0.815 | 0.475 | 0.793 | 0.449 | 0.525 |
| 7 | adaptive | 0.801 | 0.475 | 0.758 | 0.454 | 0.507 |
| 7 | adaptive_temporal | 0.799 | 0.470 | 0.750 | 0.437 | 0.509 |
| 7 | all_three | 0.797 | 0.471 | 0.756 | 0.446 | 0.510 |
| 7 | baseline | 0.803 | 0.476 | 0.752 | 0.443 | 0.506 |
| 7 | focal | 0.801 | 0.475 | 0.753 | 0.440 | 0.506 |
| 7 | temporal | 0.800 | 0.477 | 0.760 | 0.455 | 0.516 |

## Paired validation improvements versus baseline

The 95% bootstrap intervals below resample the five *matched seeds*, after averaging over horizons. These are NOT sampling-confidence intervals over years/traps.

| Method | Delta Val AUC (95% seed interval) | Delta Val PR-AUC (95% seed interval) |
|---|---:|---:|
| baseline | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| adaptive | -0.0011 [-0.0024, +0.0003] | -0.0002 [-0.0031, +0.0027] |
| temporal | -0.0015 [-0.0027, -0.0003] | +0.0018 [-0.0022, +0.0061] |
| focal | -0.0012 [-0.0022, -0.0005] | -0.0008 [-0.0015, +0.0000] |
| adaptive_temporal | +0.0006 [-0.0014, +0.0029] | +0.0051 [+0.0025, +0.0073] |
| all_three | +0.0001 [-0.0031, +0.0034] | +0.0045 [+0.0026, +0.0059] |

## Decision and limitations

**Validation-selected:** adaptive_temporal.
**Both validation metrics positive in:** 1 of 3 horizons.
**Promotion for independent replication:** no.

Even a promising method needs an untouched temporal/spatial test set, real climate and land-use features, matched model capacity, and the authors' exact GraphMAGE implementation before a publication claim.

Supporting files: all_runs.csv, horizon_summary.csv, paired_deltas.csv, paired_seed_bootstrap.csv, overall_summary.csv, decision.json, horizon_comparison.png, and individual horizon folders.