"""Unit tests for paired, validation-only multihorizon analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.run_multihorizon import compare, VARIANTS


def test_paired_horizon_selection_uses_validation_not_test(tmp_path):
    rows = []
    for horizon in (1, 3, 7):
        for seed in (42, 43):
            for variant in VARIANTS:
                is_adaptive = variant == "adaptive"
                rows.append({
                    "horizon": horizon,
                    "seed": seed,
                    "variant": variant,
                    "val_auc": .80 + (.02 if is_adaptive else 0),
                    "val_pr_auc": .40 + (.02 if is_adaptive else 0),
                    # Adversarial test metrics: the baseline is far stronger
                    # on test data, but this must not change model selection.
                    "test_auc": .55 if is_adaptive else .99,
                    "test_pr_auc": .35 if is_adaptive else .89,
                    "test_f1": .4,
                    "test_sensitivity": .5,
                    "test_brier": .2,
                })
    summary, decision = compare(pd.DataFrame(rows), [1, 3, 7], 2, tmp_path)
    assert decision["validation_winner"] == "adaptive"
    assert decision["promoted_for_independent_replication"]
    assert decision["positive_validation_horizons"] == 3
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "paired_seed_bootstrap.csv").exists()
    diffs = pd.read_csv(tmp_path / "paired_deltas.csv")
    assert len(diffs) == 36
    adaptive = diffs[diffs.variant == "adaptive"]
    assert all(abs(adaptive.delta_val_auc - .02) < 1e-9)
    assert summary.iloc[0]["variant"] == "adaptive"


def test_unpaired_or_missing_run_is_rejected(tmp_path):
    rows = []
    for horizon in (1, 3, 7):
        for variant in VARIANTS:
            rows.append({"horizon": horizon, "seed": 42, "variant": variant,
                         "val_auc": .75, "val_pr_auc": .5, "test_auc": .7,
                         "test_pr_auc": .4, "test_f1": .5,
                         "test_sensitivity": .6, "test_brier": .2})
    df = pd.DataFrame(rows).iloc[:-1]
    try:
        compare(df, [1, 3, 7], 1, tmp_path)
    except ValueError as e:
        assert "Missing or duplicate" in str(e)
    else:
        raise AssertionError("Missing run was incorrectly accepted")
