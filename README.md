# GraphMAGE Research | Automated Pilot Agent

An **automated small experiment agent** for the 2025 paper [Forecasting West Nile virus with deep graph encoders](https://arxiv.org/abs/2509.22657).

It tests six variants: a lightweight GraphMAGE-inspired baseline, learned adaptive graph, temporal GRU, focal loss, adaptive+temporal, and all three. It runs matched seeds and train/validation/test years, chooses the leading model using **validation only**, and writes run-level metrics, a ranked summary, and a plot.

**Research scope warning:** The model is a **proxy inspired by the paper's graph embedding recurrence**, not the authors' model trained on their paper's original Illinois dataset. A successful experiment here establishes engineering feasibility, not publication-level improvement or a direct numerical comparison against the original paper.

## Run a local synthetic smoke test

```bash
pip install -r requirements.txt
python agent.py --data synthetic --max-traps 48 --horizon 3 --epochs 18 --seeds 2 --out results/synthetic
```

## Run on **real** Chicago public mosquito tests

With internet access:

```bash
python agent.py --data chicago --max-traps 48 --horizon 3 --epochs 20 --seeds 3 --out results/chicago
```

If you already downloaded the CSV from https://data.cityofchicago.org/d/jqe8-8r6s:

```bash
python agent.py --data chicago --csv /path/to/chicago_wnv.csv --max-traps 48 --epochs 20 --seeds 3 --out results/chicago
```

The agent caches Chicago's public CSV in `data/chicago_wnv.csv`. The CSV schema is detected flexibly. Chicago data lacks the Illinois paper's full meteorological and land-use covariates, so adaptive connections in Chicago mode use available spatial, seasonal, and lagged-surveillance features only. It is **not** a climate-aware comparison. The paper's exact original experiment must be run on the authors' dataset/code before claiming improved results.

## Output files

- `results/.../REPORT.md`: easy-to-read results and proposed follow-up.
- `results/.../summary.csv`: per-method average scores and validation-ranked selection.
- `results/.../runs.csv`: scores for each random seed (including separate validation/test).
- `results/.../metadata.json`: data source, split, settings, and observation counts.
- `results/.../comparison.png`: AUC and PR-AUC on held-out years.
- `results/.../decision.json`: screening status. A candidate needs at least 3 seeds and improvements of at least 0.01 in both validation AUC and validation PR-AUC.

## What is protected against data leakage?

- Split by **target year**, never random rows. Last 2 years are test, the preceding year validation, all earlier years train.
- Select nodes based on **training years only**.
- Week `t` can see only features and **prior** observations, not the outcome at week `t` or future week `t+h`.
- Missing/unmeasured trap-weeks remain `NaN`, **never counted as negative cases**.
- Classification threshold is optimized on **validation**, never test.
- The winner is chosen by the validation ROC-AUC/PR-AUC composite, **not test metrics**.

## Limitations and future research

- Synthetic data: sanity check only. Do not use its scores as evidence of real-world gains.
- Chicago mode: real observations, but not the paper's exact statewide dataset; weather covariates absent.
- Small runs (1-3 seeds) are exploration, not statistical confirmation.
- Train/test years are chronological, but unseen-region/rural splits are still needed.
- Candidate selection across variants must be isolated from final test reporting in a formal paper; this agent surfaces test metrics for exploratory comparison, but do not repeatedly tune using them.
- For proper publication experiments, obtain author's original code/data, use 20+ seeds and all eight horizons, independent temporal and spatial holdouts, uncertainty intervals, feature ablation, class imbalance baselines, and calibration.

## Architecture

The baseline follows GraphMAGE's defining recurrence: initial node embedding `h0` is concatenated with aggregated neighboring representations **at every graph layer**, rather than with the previous self-embedding. Variants optionally add learned edge weights (restricted to geographically plausible kNN edges), a GRU of the last four weeks, and focal loss. All versions use the same small depth and comparable preprocessing.

Research context: [Greiffenstein et al. 2025](https://arxiv.org/abs/2509.22657). City data: [City of Chicago WNV mosquito results](https://data.cityofchicago.org/d/jqe8-8r6s).

**Model complexity caveat:** The GRU and adaptive attention add parameters, so these are not parameter-matched ablations. Match model size and compute budgets in the full study.