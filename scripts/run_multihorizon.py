#!/usr/bin/env python3
"""Three-horizon, five-seed GraphMAGE proxy screening experiment.

All selection uses validation metrics only. Previously examined 2024-2025 test
years are exploratory, not an untouched independent test set.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("val_auc", "val_pr_auc", "test_auc", "test_pr_auc",
           "test_f1", "test_sensitivity", "test_brier")
VARIANTS = ("baseline", "adaptive", "temporal", "focal",
            "adaptive_temporal", "all_three")


def compare(runs, horizons, seeds, output, metadata=None):
    """Compute matched-seed deltas; do not use test scores to select a model."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {(h, seed, variant) for h in horizons
                for seed in range(42, 42 + seeds) for variant in VARIANTS}
    actual = set(zip(runs["horizon"], runs["seed"], runs["variant"]))
    if expected != actual or len(runs) != len(expected):
        raise ValueError("Missing or duplicate horizon/seed/variant runs")
    runs.to_csv(output / "all_runs.csv", index=False)
    by_horizon = runs.groupby(["horizon", "variant"], as_index=False)[list(METRICS)].mean()
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)

    # A pairing compares each candidate to the baseline with the SAME seed
    # and prediction horizon. Seeds across horizons are NOT 15 independent
    # training data replicates, so aggregate per seed before bootstrapping.
    baseline = (runs[runs.variant == "baseline"]
                .set_index(["horizon", "seed"]))
    diffs = []
    for _, row in runs.iterrows():
        ref = baseline.loc[(row["horizon"], row["seed"])]
        rec = {"horizon": int(row["horizon"]), "seed": int(row["seed"]),
               "variant": row["variant"]}
        for metric in METRICS:
            rec["delta_" + metric] = float(row[metric] - ref[metric])
        diffs.append(rec)
    paired = pd.DataFrame(diffs)
    paired.to_csv(output / "paired_deltas.csv", index=False)
    per_seed = paired.groupby(["variant", "seed"], as_index=False).mean(numeric_only=True)
    delta_cols = ["delta_" + metric for metric in METRICS]
    rng = np.random.default_rng(20261118)
    records = []
    for variant in VARIANTS:
        part = per_seed[per_seed.variant == variant]
        rec = {"variant": variant, "n_seeds": len(part)}
        for metric in delta_cols:
            values = part[metric].to_numpy(dtype=float)
            boot = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
            rec[metric] = float(values.mean())
            rec[metric + "_ci_low"] = float(np.quantile(boot, 0.025))
            rec[metric + "_ci_high"] = float(np.quantile(boot, 0.975))
        records.append(rec)
    delta_summary = pd.DataFrame(records)
    delta_summary.to_csv(output / "paired_seed_bootstrap.csv", index=False)

    overall = runs.groupby("variant", as_index=False)[list(METRICS)].mean()
    overall["selection_score"] = 0.6 * overall.val_auc + 0.4 * overall.val_pr_auc
    overall = overall.sort_values("selection_score", ascending=False)
    overall.to_csv(output / "overall_summary.csv", index=False)
    winner = str(overall.iloc[0]["variant"])
    win_diffs = delta_summary.set_index("variant").loc[winner]
    wins_horizons = 0
    for horizon in horizons:
        d = paired[(paired.variant == winner) & (paired.horizon == horizon)]
        if d.delta_val_auc.mean() > 0 and d.delta_val_pr_auc.mean() > 0:
            wins_horizons += 1
    promoted = (winner != "baseline"
                and win_diffs.delta_val_auc >= 0.01
                and win_diffs.delta_val_pr_auc >= 0.01
                and wins_horizons >= 2)
    decision = {
        "selection": "validation-only mean of equal-weight horizons",
        "validation_score": "0.6 * ROC-AUC + 0.4 * PR-AUC",
        "validation_winner": winner,
        "promoted_for_independent_replication": bool(promoted),
        "positive_validation_horizons": int(wins_horizons),
        "promotion_rule": "mean paired validation ROC-AUC and PR-AUC gains each >= 0.01, "
                          "and both positive in at least 2 of 3 horizons",
        "test_data_warning": "2024-2025 test years already viewed in prior pilot: "
                             "all test metrics are exploratory only.",
        "uncertainty_warning": "Seed-bootstrap CIs quantify training randomness only. "
                               "Five seeds do not capture uncertainty across seasons or locations.",
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7))
    for metric, ax, title in (("test_auc", axs[0], "Exploratory test ROC-AUC"),
                               ("test_pr_auc", axs[1], "Exploratory test PR-AUC")):
        for variant in VARIANTS:
            sub = by_horizon[by_horizon.variant == variant].sort_values("horizon")
            ax.plot(sub.horizon, sub[metric], marker="o", label=variant)
        ax.set_xlabel("Forecast horizon (weeks)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.2)
    axs[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "horizon_comparison.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Chicago Multi-Horizon GraphMAGE Pilot",
        "",
        "**Status:** " + ("candidate for independent replication" if promoted else
                         "INCONCLUSIVE, no promotion by the validation rule"),
        "",
        "**Research scope:** GraphMAGE-inspired proxy, NOT the original paper's model. "
        "Chicago surveillance data have no weather or land-use covariates.",
        "",
        f"**Design:** {seeds} matched seeds, horizons {horizons}, 20 epoch maximum per run, "
        "48 traps. Each method uses the same chronological train/validation/test years.",
        "**Years:** Train 2007-2022; validation 2023; exploratory test 2024-2025.",
        "",
        "**Warning:** The 2024-2025 results have already been inspected in an earlier experiment. "
        "They cannot be treated as fresh confirmation or used to choose the method.",
        "",
        "## Overall mean across horizons and seeds (validation ranked)",
        "",
        "| Method | Val AUC | Val PR-AUC | Test AUC (exploratory) | Test PR-AUC (exploratory) | Test F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in overall.iterrows():
        lines.append(f"| {r['variant']} | {r.val_auc:.3f} | {r.val_pr_auc:.3f} | "
                     f"{r.test_auc:.3f} | {r.test_pr_auc:.3f} | {r.test_f1:.3f} |")
    lines += [
        "", "## Per-horizon comparison (means over matched seeds)", "",
        "| Weeks | Method | Val AUC | Val PR-AUC | Test AUC | Test PR-AUC | Test F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in by_horizon.sort_values(["horizon", "variant"]).iterrows():
        lines.append(f"| {int(r.horizon)} | {r['variant']} | {r.val_auc:.3f} | "
                     f"{r.val_pr_auc:.3f} | {r.test_auc:.3f} | {r.test_pr_auc:.3f} | {r.test_f1:.3f} |")
    lines += [
        "", "## Paired validation improvements versus baseline",
        "",
        "The 95% bootstrap intervals below resample the five *matched seeds*, "
        "after averaging over horizons. These are NOT sampling-confidence intervals over years/traps.",
        "",
        "| Method | Delta Val AUC (95% seed interval) | Delta Val PR-AUC (95% seed interval) |",
        "|---|---:|---:|",
    ]
    for _, r in delta_summary.iterrows():
        lines.append(
            f"| {r['variant']} | {r.delta_val_auc:+.4f} "
            f"[{r.delta_val_auc_ci_low:+.4f}, {r.delta_val_auc_ci_high:+.4f}] | "
            f"{r.delta_val_pr_auc:+.4f} "
            f"[{r.delta_val_pr_auc_ci_low:+.4f}, {r.delta_val_pr_auc_ci_high:+.4f}] |"
        )
    lines += [
        "", "## Decision and limitations", "",
        f"**Validation-selected:** {winner}.",
        f"**Both validation metrics positive in:** {wins_horizons} of {len(horizons)} horizons.",
        f"**Promotion for independent replication:** {'yes' if promoted else 'no'}.",
        "",
        "Even a promising method needs an untouched temporal/spatial test set, "
        "real climate and land-use features, matched model capacity, and "
        "the authors' exact GraphMAGE implementation before a publication claim.",
        "",
        "Supporting files: all_runs.csv, horizon_summary.csv, paired_deltas.csv, "
        "paired_seed_bootstrap.csv, overall_summary.csv, decision.json, "
        "horizon_comparison.png, and individual horizon folders.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if metadata is not None:
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return overall, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["chicago", "synthetic"], default="chicago")
    parser.add_argument("--csv", default=None, help="Local CSV; optional for Chicago")
    parser.add_argument("--max-traps", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 7])
    parser.add_argument("--out", default="results/multihorizon-chicago")
    args = parser.parse_args()
    if args.seeds < 1 or args.epochs < 1 or any(h < 1 for h in args.horizons):
        parser.error("seeds, epochs, and horizons must be positive")
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("horizons must be unique")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": args.data, "seeds": args.seeds, "epochs": args.epochs,
                "max_traps": args.max_traps, "horizons": args.horizons,
                "head_sha": os.environ.get("GITHUB_SHA", "local"),
                "research_scope": "exploratory proxy screen"}
    frames = []
    for h in args.horizons:
        sub = output / f"horizon_{h}"
        sub.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "agent.py"), "--data", args.data,
               "--max-traps", str(args.max_traps), "--horizon", str(h),
               "--epochs", str(args.epochs), "--seeds", str(args.seeds),
               "--out", str(sub)]
        if args.csv is not None:
            cmd += ["--csv", args.csv]
        print("Running forecast horizon", h, "weeks", flush=True)
        with (sub / "run.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            tail = (sub / "run.log").read_text(errors="replace")[-6000:]
            raise RuntimeError(f"Horizon {h} failed (exit {result.returncode}):\n{tail}")
        frame = pd.read_csv(sub / "runs.csv")
        frame.insert(0, "horizon", h)
        frames.append(frame)
        print("Completed horizon", h, "weeks", flush=True)
    overall, decision = compare(pd.concat(frames, ignore_index=True),
                                args.horizons, args.seeds, output, metadata)
    print("Validation-ranked methods:", flush=True)
    print(overall[["variant", "val_auc", "val_pr_auc", "test_auc", "test_pr_auc"]].round(3).to_string(index=False), flush=True)
    print("Decision:", decision, flush=True)
    print("Report:", output / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
