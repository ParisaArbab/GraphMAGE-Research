# Chicago Real-Data Pilot, 18 November 2026

## Status and provenance

- **GitHub Actions run:** [44359741522](https://github.com/ParisaArbab/GraphMAGE-Research/actions/runs/44359741522)
- **Original artifact:** [graphmage-pilot-results](https://github.com/ParisaArbab/GraphMAGE-Research/actions/runs/44359741522/artifacts/14454210545)
- **Workflow result:** success, both verify and pilot jobs passed.
- **Model:** lightweight GraphMAGE-inspired **proxy**, not an exact reproduction of Greiffenstein et al. (2025).
- **Dataset:** public Chicago mosquito-test records, 2007 to 2025; **weather and land-use features were not included**.
- **Selected traps:** 48, selected using training years only.
- **Training:** 2007–2022 (9,927 labeled observations); **validation:** 2023 (627); **held-out test:** 2024–2025 (1,401).
- **Prediction horizon:** 3 weeks. **Seeds:** 42, 43, 44 (3 each; 18 model runs). **Epoch limit:** 20.

## Mean scores across three random seeds

| Method | Validation ROC-AUC | Validation PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test sensitivity | Test Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline proxy | 0.819 | 0.480 | 0.776 | 0.419 | 0.497 | 0.696 | 0.185 |
| Adaptive graph | 0.818 | 0.481 | **0.783** | **0.448** | 0.504 | 0.662 | 0.178 |
| Temporal GRU | 0.815 | 0.470 | 0.781 | 0.423 | **0.507** | 0.714 | **0.165** |
| Focal loss | 0.818 | 0.477 | 0.777 | 0.424 | 0.499 | 0.702 | 0.190 |
| Adaptive + temporal | 0.817 | 0.472 | 0.776 | 0.415 | 0.503 | **0.750** | 0.188 |
| All three | 0.817 | 0.477 | 0.781 | 0.421 | 0.503 | 0.705 | 0.197 |

### Interpretation

1. **Automated decision: INCONCLUSIVE.** The baseline was selected by the predefined validation score (0.6 ROC-AUC + 0.4 PR-AUC). No model cleared the validation improvement gate of +0.01 on both ROC-AUC and PR-AUC.
2. Adaptive graph improved **held-out test** ROC-AUC by 0.0076 and PR-AUC by 0.0284 compared with the baseline average, but test sensitivity decreased by 0.0337. This *does not* validate an improvement, especially since validation did not support promotion.
3. The temporal GRU had the lowest test Brier score (0.165 versus 0.185) and highest test F1 (0.507 versus 0.497), but poorer validation ranking.
4. Adaptive + temporal had the highest test sensitivity (0.750), but weaker PR-AUC and no validation advantage.
5. Results vary across the three seeds. These runs do not give confidence intervals or establish statistical significance.

### Research recommendation

**Keep adaptive graph learning and temporal GRU as hypotheses, not established winners.** The most important next step is to run a predeclared multi-seed comparison on horizons 1, 3, and 7 weeks, with appropriate independent test years and uncertainty intervals. The formal study must use the original GraphMAGE implementation and comparable covariates/data before comparing against published results. Do not tune using the held-out 2024–2025 scores.

### Data-integrity note

The preceding failed CI run was caused by Chicago's reversed coordinate columns. The corrected loader detects reversed coordinates (45,361 rows) and leaves invalid/missing coordinates excluded. The successful run ingested 45,481 rows and found 40,108 rows with valid dates and labels, before applying the full cleaning and sampling filters.

*All metrics are copied from the automatically generated artifact for run 44359741522. The artifact contains full per-seed runs, report, decision.json, metadata.json, chart, and logs.*
