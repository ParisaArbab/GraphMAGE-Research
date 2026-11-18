"""Regression tests for the 2026 holdout and time-safe weather ablations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.diagnose_climate2026 import (
    MixGraphMAGE, choose_and_summarize, format_config,
    historical_weather_by_week, split_new_year,
)


def test_weather_uses_prior_seven_days_only():
    dates = pd.date_range("2020-04-01", "2020-05-12", freq="D")
    daily = pd.DataFrame({
        "temperature_2m_mean": np.repeat(15., len(dates)),
        "precipitation_sum": np.repeat(2., len(dates)),
        "relative_humidity_2m_mean": np.repeat(60., len(dates)),
    }, index=dates)
    initial = historical_weather_by_week(daily, [pd.Timestamp("2020-05-04")])
    # Future week weather must NOT enter forecasts made at May 4.
    daily.loc["2020-05-04":, "temperature_2m_mean"] = 1000.
    future_changed = historical_weather_by_week(daily, [pd.Timestamp("2020-05-04")])
    assert np.allclose(initial, future_changed)
    assert np.allclose(initial[:, 0], 0.)
    assert np.allclose(initial[:, 1], np.log1p(14) / 4)
    assert np.allclose(initial[:, 2], 0.)


def test_weather_missing_history_is_error():
    daily = pd.DataFrame({
        "temperature_2m_mean": [15.] * 5,
        "precipitation_sum": [0.] * 5,
        "relative_humidity_2m_mean": [60.] * 5,
    }, index=pd.date_range("2020-04-29", periods=5))
    with pytest.raises(ValueError, match="weather missing"):
        historical_weather_by_week(daily, [pd.Timestamp("2020-05-04")])


def test_2026_is_a_separate_new_holdout():
    years = np.array([2015, 2022, 2023, 2024, 2025, 2026, 2026])
    tr, va, te = split_new_year(years)
    assert set(years[tr]) == {2015, 2022, 2023}
    assert set(years[va]) == {2024, 2025}
    assert set(years[te]) == {2026}
    with pytest.raises(ValueError):
        split_new_year(years[years != 2026])


def test_graph_mix_forward_and_gradients():
    torch.manual_seed(0)
    x = torch.randn(3, 9, 8, 13)
    graph = torch.ones(9, 9) - torch.eye(9)
    graph = graph / graph.sum(dim=1, keepdim=True)
    for alpha in (0., .5, 1.):
        m = MixGraphMAGE(13, alpha=alpha)
        y = m(x, graph)
        assert y.shape == (3, 9)
        y.mean().backward()
        assert all(p.grad is not None for p in m.parameters())


def test_selection_uses_only_validation_scores(tmp_path):
    rows = []
    for horizon in (1, 3, 7):
        for seed in (42, 43, 44):
            for config, auc, pr in (
                ("hist4_edge0p0_weather0", .75, .42),
                ("hist8_edge0p5_weather1", .77, .45),
            ):
                rows.append({
                    "horizon": horizon, "seed": seed, "config": config,
                    "val_auc": auc, "val_pr_auc": pr,
                    "val_f1": .5, "val_brier": .2, "parameters": 1500,
                })
    ranked, decisions = choose_and_summarize(pd.DataFrame(rows), tmp_path)
    assert len(decisions) == 3
    assert all(d["chosen_config"] == "hist8_edge0p5_weather1" for d in decisions)
    assert all(d["passes_validation_gate"] for d in decisions)
    assert "test2026_auc" not in ranked.columns
    assert format_config(8, 0.5, True) == "hist8_edge0p5_weather1"
