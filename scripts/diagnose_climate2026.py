#!/usr/bin/env python3
"""Predeclared Chicago 2026 holdout: temporal history, graph mixing, weather.

The 2026 labels are NEVER used to select configurations or thresholds. Earlier
GraphMAGE pilots only evaluated up to 2025. This is a GraphMAGE-inspired proxy.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import (acquire_chicago, as_windows, load_chicago, loss_fn,
                   make_knn, metrics, set_seed, validation_threshold)

WEATHER_SOURCE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_DAILY = ("temperature_2m_mean", "precipitation_sum",
                 "relative_humidity_2m_mean")
HORIZONS = (1, 3, 7)
HISTORIES = (2, 4, 8)
EDGES = (0.0, 0.5, 1.0)
SEEDS = (42, 43, 44)
SPLIT = {"train": "2007-2023", "validation": "2024-2025",
         "test_2026": "2026 (not viewed by earlier pilot)"}


def load_weather(cache, last_day="2026-10-31"):
    """Fetch Chicago-center ERA5 reanalysis; never silently invent weather."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        params = {
            "latitude": 41.88, "longitude": -87.64,
            "start_date": "2007-01-01", "end_date": last_day,
            "daily": ",".join(WEATHER_DAILY),
            "timezone": "America/Chicago", "models": "era5",
        }
        url = WEATHER_SOURCE + "?" + urllib.parse.urlencode(params)
        print("Downloading historical weather from", WEATHER_SOURCE, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "GraphMAGE-Research-2026/1.0"})
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.load(response)
        if payload.get("error") or "daily" not in payload:
            raise RuntimeError("Open-Meteo returned no weather daily data: " + str(payload)[:400])
        daily = pd.DataFrame(payload["daily"])
        if not set(WEATHER_DAILY).issubset(daily.columns):
            raise ValueError("Missing requested weather columns: " + str(daily.columns.tolist()))
        daily.to_csv(cache, index=False)
    daily = pd.read_csv(cache)
    for col in ("time",) + WEATHER_DAILY:
        if col not in daily:
            raise ValueError("Missing weather field: " + col)
    daily["time"] = pd.to_datetime(daily["time"], errors="raise")
    daily = daily.set_index("time").sort_index()
    if daily.index.has_duplicates:
        raise ValueError("Duplicate weather dates")
    for col in WEATHER_DAILY:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    return daily, hashlib.sha256(cache.read_bytes()).hexdigest()


def historical_weather_by_week(daily, week_starts):
    """Use days [week_start-7, week_start-1], never the forecast week or future.

    Chicago-wide weather is applied to every trap (not location-specific climate).
    """
    out = []
    for week in week_starts:
        start = pd.Timestamp(week)
        days = pd.date_range(start - pd.Timedelta(days=7),
                             start - pd.Timedelta(days=1), freq="D")
        window = daily.reindex(days)
        if len(window) != 7 or window[list(WEATHER_DAILY)].isna().any().any():
            raise ValueError(f"Historical weather missing for week {start.date()}; "
                             "do not impute with future dates")
        temp = float(window["temperature_2m_mean"].mean())
        rain = float(window["precipitation_sum"].clip(lower=0).sum())
        humidity = float(window["relative_humidity_2m_mean"].mean())
        out.append([(temp - 15) / 15, np.log1p(rain) / 4, (humidity - 60) / 30])
    a = np.asarray(out, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite historical weather features")
    return a


def split_new_year(years):
    """2026 is a completely new holdout relative to older 2025-limited runs."""
    years = np.asarray(years)
    tr = np.flatnonzero(years <= 2023)
    va = np.flatnonzero(np.isin(years, (2024, 2025)))
    te = np.flatnonzero(years == 2026)
    if any(len(x) == 0 for x in (tr, va, te)):
        raise ValueError("Missing training, 2024-25 validation or new 2026 test targets")
    if not (np.max(years[tr]) < np.min(years[va])
            and np.max(years[va]) < np.min(years[te])):
        raise ValueError("Bad temporal split")
    return tr, va, te


class MixGraphMAGE(nn.Module):
    """Original-feature skip recurrence with tunable fixed-vs-learned edges.

    alpha=0 fixed geographic kNN; 0.5 half; 1 learned attention within kNN.
    This is a controlled *proxy*, not the official GraphMAGE code.
    """
    def __init__(self, dim, alpha=0., hidden=24, depth=3, temporal=True):
        super().__init__()
        self.alpha = float(alpha)
        self.temporal = temporal
        self.embed = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        if temporal:
            self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
            for _ in range(depth)
        ])
        if alpha > 0:
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, history, graph):
        b, n, l, f = history.shape
        h0 = self.embed(history[:, :, -1])
        if self.temporal:
            _, h = self.gru(history.reshape(b * n, l, f))
            h0 = h0 + h[-1].reshape(b, n, -1)
        if self.alpha > 0:
            scores = torch.matmul(self.query(h0),
                                  self.key(h0).transpose(1, 2)) / (h0.shape[-1] ** 0.5)
            scores = scores.masked_fill(graph[None] <= 0, -1e5)
            learned = torch.softmax(scores, dim=-1)
            adjacency = (1.0 - self.alpha) * graph[None] + self.alpha * learned
        else:
            adjacency = graph[None]
        current = h0
        for layer in self.layers:
            neighbor = torch.bmm(adjacency.expand(b, -1, -1), current)
            current = layer(torch.cat((h0, neighbor), dim=-1))
        return self.out(current).squeeze(-1)


def train_model(x, y, tr, va, graph, seed, alpha=0., temporal=True,
                epochs=20, test_index=None):
    """Checkpoint, threshold and all search choices determined by validation."""
    set_seed(seed)
    model = MixGraphMAGE(x.shape[-1], alpha=alpha, temporal=temporal)
    opt = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.004)
    tx = torch.from_numpy(x).float()
    ty = torch.from_numpy(y).float()
    labels = y[tr][np.isfinite(y[tr])]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Training labels missing either class")
    n_pos = max(1., float(np.sum(labels > .5)))
    n_neg = float(np.sum(labels <= .5))
    pos_weight = min(8., max(1., n_neg / n_pos))
    best = -float("inf")
    best_state = None
    best_epoch = 0
    stalls = 0
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(tr)
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 16)))):
            opt.zero_grad()
            logits = model(tx[batch], graph)
            loss = loss_fn(logits, ty[batch], focal=False, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
        model.eval()
        with torch.no_grad():
            vprob = torch.sigmoid(model(tx[va], graph)).numpy()
        val = metrics(y[va], vprob, .5)
        rank = .6 * val["auc"] + .4 * val["pr_auc"]
        if np.isfinite(rank) and rank > best + 1e-4:
            best, best_epoch, stalls = rank, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stalls += 1
        if stalls >= 5:
            break
    if best_state is None:
        raise ValueError("No valid validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(tx[va], graph)).numpy()
    threshold = validation_threshold(y[va], val_prob)
    out = {"seed": int(seed), "best_epoch": best_epoch,
           "parameters": sum(p.numel() for p in model.parameters()),
           "threshold": threshold}
    out.update({"val_" + k: v for k, v in metrics(y[va], val_prob, threshold).items()})
    if test_index is not None:
        with torch.no_grad():
            p = torch.sigmoid(model(tx[test_index], graph)).numpy()
        out.update({"test2026_" + k: v for k, v in metrics(y[test_index], p, threshold).items()})
    return out


def format_config(hist, alpha, weather):
    return f"hist{hist}_edge{str(alpha).replace('.', 'p')}_weather{int(weather)}"


def choose_and_summarize(search, output):
    """Selection without any access to 2026 labels or metrics."""
    output = Path(output)
    ranking = search.groupby(["horizon", "config"], as_index=False).agg(
        val_auc=("val_auc", "mean"), val_pr_auc=("val_pr_auc", "mean"),
        val_f1=("val_f1", "mean"), val_brier=("val_brier", "mean"),
        parameter_count=("parameters", "mean"),
        n_seeds=("seed", "size"),
    )
    ranking["selection_score"] = .6 * ranking.val_auc + .4 * ranking.val_pr_auc
    ranking.to_csv(output / "validation_summary.csv", index=False)
    baseline = "hist4_edge0p0_weather0"
    decisions = []
    for horizon, group in ranking.groupby("horizon"):
        top = group.sort_values("selection_score", ascending=False).iloc[0]
        ref = group[group.config == baseline]
        if ref.empty:
            raise ValueError("Missing fixed graph + temporal baseline")
        ref = ref.iloc[0]
        da = float(top.val_auc - ref.val_auc)
        dp = float(top.val_pr_auc - ref.val_pr_auc)
        decisions.append({
            "horizon": int(horizon),
            "chosen_config": str(top.config),
            "baseline_config": baseline,
            "val_auc_gain": da,
            "val_pr_auc_gain": dp,
            "passes_validation_gate": bool(da >= .01 and dp >= .01),
            "test_year": 2026,
        })
    (output / "decision.json").write_text(json.dumps(decisions, indent=2))
    return ranking, decisions


def make_figures(summary, holdout, out):
    out = Path(out)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    baseline = "hist4_edge0p0_weather0"
    for ax, h in zip(axes, HORIZONS):
        group = summary[summary.horizon == h].sort_values("selection_score", ascending=False)
        best = group.iloc[:6]
        ax.barh(best.config.iloc[::-1], best.val_pr_auc.iloc[::-1])
        ax.set_xlim(0, max(.05, min(1, float(best.val_pr_auc.max()) + .06)))
        ax.set_title(f"{h}-week: validation PR-AUC")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "validation_method_comparison.png", dpi=150)
    plt.close(fig)
    if not holdout.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ("test2026_auc", "test2026_pr_auc")):
            sub = holdout.groupby(["horizon", "group"], as_index=False)[metric].mean()
            for name, vals in sub.groupby("group"):
                ax.plot(vals.horizon, vals[metric], "-o", label=name)
            ax.set_title(metric + " (2026 held-out)")
            ax.set_xticks(HORIZONS)
            ax.set_xlabel("Forecast horizon (weeks)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "holdout_2026_comparison.png", dpi=150)
        plt.close(fig)


def write_report(summary, decisions, holdout, metadata, out):
    out = Path(out)
    lines = [
        "# Why temporal + adaptive graph changes with forecast horizon",
        "",
        "## Scientific scope",
        "This is an automated, limited GraphMAGE-inspired **proxy** experiment.",
        "It does not reproduce the paper's official architecture, original features, "
        "or statewide Illinois test population.",
        "",
        "## Fixed data design",
        "- Chicago traps selected using only years up to 2023; the new test year is **2026**.",
        "- Training 2007–2023, validation 2024–2025, test 2026.",
        "- Validation years were examined as held-out years in earlier experiments, "
        "so this experiment is an extension, not a completely independent discovery study.",
        "- 2026 was not used in previous pilots or in configuration/threshold selection.",
        "- Historical temperature, rainfall and humidity are based on Chicago-center "
        "ERA5 reanalysis from Open-Meteo, using the seven days strictly **before** the forecast week.",
        "- Weather is city-level and identical at all traps, NOT climate-aware graph edges.",
        f"- {metadata['seeds']} seeds; {metadata['epochs']} maximum epochs; histories 2/4/8; "
        "edge mix alpha 0, 0.5, or 1; weather on/off; horizons 1/3/7 weeks.",
        "",
        "## Validation-only selected configurations",
        "",
        "| Horizon | Selected configuration | Gain ROC-AUC over fixed temporal baseline | Gain PR-AUC | Promotion gate |",
        "|---:|---|---:|---:|---|",
    ]
    for d in decisions:
        lines.append(f"| {d['horizon']} weeks | {d['chosen_config']} | {d['val_auc_gain']:+.4f} | "
                     f"{d['val_pr_auc_gain']:+.4f} | "
                     f"{'pass' if d['passes_validation_gate'] else 'fail'} |")
    lines += [
        "",
        "Selection is from **validation only**: 0.6 × ROC-AUC + 0.4 × PR-AUC. "
        "A method must improve both metrics by 0.01 over the fixed temporal baseline.",
        "",
        "## New 2026 held-out results, compared with frozen baselines",
        "",
        "| Horizon | Frozen model | ROC-AUC | PR-AUC | F1 | Sensitivity | Samples |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    groups = holdout.groupby(["horizon", "group"], as_index=False).mean(numeric_only=True)
    for _, r in groups.sort_values(["horizon", "group"]).iterrows():
        lines.append(f"| {int(r.horizon)} weeks | {r.group} | {r.test2026_auc:.3f} | "
                     f"{r.test2026_pr_auc:.3f} | {r.test2026_f1:.3f} | "
                     f"{r.test2026_sensitivity:.3f} | {int(r.test2026_n)} |")
    lines += [
        "", "## Interpretation and safeguards",
        "",
        "All 2026 evaluations were run **after** validation-only selection. "
        "No 2026 metric influenced model choice, checkpoint selection, or decision thresholds.",
        "We include both the prior proxy baseline (no GRU, fixed graph, no weather) "
        "and the fixed-graph GRU baseline to separate temporal gains from other changes.",
        "Differences in parameter counts and 3-seed randomness limit conclusions.",
        "No climate information varies between traps, so this is a **weather-covariate ablation**, "
        "not a location-dependent climate-graph claim.",
        "The 2026 season provides a new held-out year, but a single year cannot show "
        "general robustness across epidemics, regions or years.",
        "",
        "## Next confirmatory work",
        "",
        "1. Freeze a single hypothesis before a *new independent season* or independent region.",
        "2. Compare model-size-matched variants and original GraphMAGE code.",
        "3. Obtain spatially varying weather and land-use features and test unseen rural traps.",
        "4. Add uncertainty over trap clusters and years, not only random seeds.",
        "",
        "Files: validation_search_runs.csv, validation_summary.csv, decision.json, "
        "holdout_2026.csv, metadata.json, validation_method_comparison.png, "
        "holdout_2026_comparison.png, run.log.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chicago-csv", default=None)
    p.add_argument("--weather-csv", default=None)
    p.add_argument("--out", default="results/climate2026")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-traps", type=int, default=48)
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--histories", type=int, nargs="+", default=list(HISTORIES))
    args = p.parse_args()
    if args.seeds < 1 or args.epochs < 1 or args.max_traps < 5:
        p.error("Invalid seeds/epochs/max-traps")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.chicago_csv) if args.chicago_csv else acquire_chicago("data/chicago_wnv.csv")
    data = load_chicago(dataset, max_traps=args.max_traps, through_year=2026)
    weather, weather_sha = load_weather(args.weather_csv or "data/openmeteo_era5_chicago_2007_2026.csv")
    weather_week = historical_weather_by_week(weather, data.dates)
    weather_x = np.broadcast_to(weather_week[:, None, :],
                                (len(data.dates), len(data.coords), 3)).copy()
    graph = make_knn(data.coords)
    metadata = {
        "chicago_source": "https://data.cityofchicago.org/d/jqe8-8r6s",
        "weather_source": WEATHER_SOURCE,
        "weather_fields": WEATHER_DAILY,
        "weather_sha256": weather_sha,
        "weather_alignment": "7 full days strictly before each forecast-week start",
        "location_dependence": "single Chicago-center grid cell, equal at all traps",
        "n_traps": len(data.coords), "seeds": args.seeds, "epochs": args.epochs,
        "split": SPLIT, "horizons": args.horizons, "histories": args.histories,
        "edge_mix": EDGES, "target_year_test": 2026, "seeds_start": 42,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    configs = []
    search_runs = []
    for h in args.horizons:
        for hist in args.histories:
            for weather_on in (False, True):
                for alpha in EDGES:
                    configs.append((h, hist, weather_on, alpha))
    # No 2026 evaluation in this search loop; test_idx is intentionally omitted.
    windows = {}
    for h in args.horizons:
        for hist in args.histories:
            for weather_on in (False, True):
                features = np.concatenate([data.x, weather_x], axis=-1) if weather_on else data.x
                x, y, years, _ = as_windows(
                    type(data)(features, data.y, data.dates,
                               data.coords, data.node_names, data.source),
                    horizon=h, history=hist)
                tr, va, te = split_new_year(years)
                windows[(h, hist, weather_on)] = (x, y, tr, va, te)
    print("2026 data available. No 2026 model metric will be used for selection.", flush=True)
    print("Validation candidates:", len(configs) * args.seeds, flush=True)
    for k, (h, hist, weather_on, alpha) in enumerate(configs, start=1):
        name = format_config(hist, alpha, weather_on)
        x, y, tr, va, te = windows[(h, hist, weather_on)]
        for seed in range(42, 42 + args.seeds):
            run = train_model(x, y, tr, va, graph, seed, alpha=alpha,
                              temporal=True, epochs=args.epochs)
            search_runs.append({"horizon": h, "config": name,
                                "history": hist, "alpha": alpha,
                                "weather": int(weather_on), **run})
        if k % 6 == 0 or k == len(configs):
            print(f"Validation search: {k}/{len(configs)} configurations", flush=True)
    # Historical, no-GRU baseline is a separate reference and never a grid candidate.
    for h in args.horizons:
        x, y, tr, va, te = windows[(h, 4, False)]
        for seed in range(42, 42 + args.seeds):
            run = train_model(x, y, tr, va, graph, seed,
                              alpha=0., temporal=False, epochs=args.epochs)
            search_runs.append({"horizon": h, "config": "proxy_no_gru",
                                "history": 4, "alpha": 0., "weather": 0, **run})
    search = pd.DataFrame(search_runs)
    search.to_csv(out / "validation_search_runs.csv", index=False)
    # Exclude old baseline from selecting among fixed-GRU configurations.
    summary, decisions = choose_and_summarize(
        search[search.config != "proxy_no_gru"], out)
    print("Validation choices frozen:", decisions, flush=True)
    holdout_runs = []
    # ONLY NOW query new 2026 scores for the frozen validation candidates + controls.
    for d in decisions:
        h = d["horizon"]
        selected = summary[(summary.horizon == h) &
                           (summary.config == d["chosen_config"])].iloc[0]
        spec = (int(selected.config.split("_")[0][4:]),
                float(selected.config.split("_")[1][4:].replace("p", ".")),
                bool(int(selected.config.split("_")[2][-1])))
        ref_configs = [
            ("chosen", spec[0], spec[1], spec[2], True),
            ("fixed_temporal", 4, 0., False, True),
            ("prior_adaptive_temporal", 4, .5, False, True),
            ("original_proxy", 4, 0., False, False),
        ]
        for group, hist, alpha, w, temporal in ref_configs:
            x, y, tr, va, te = windows[(h, hist, w)]
            for seed in range(42, 42 + args.seeds):
                run = train_model(x, y, tr, va, graph, seed,
                                  alpha=alpha, temporal=temporal,
                                  epochs=args.epochs, test_index=te)
                holdout_runs.append({"horizon": h, "group": group,
                                     "config": (format_config(hist, alpha, w)
                                                if temporal else "proxy_no_gru"),
                                     **run})
        print(f"Frozen 2026 evaluation complete: {h}-week", flush=True)
    holdout = pd.DataFrame(holdout_runs)
    holdout.to_csv(out / "holdout_2026.csv", index=False)
    write_report(summary, decisions, holdout, metadata, out)
    make_figures(summary, holdout, out)
    print("Validation ranking by horizon:", flush=True)
    for d in decisions:
        print(d, flush=True)
    print("New 2026 held-out mean scores:", flush=True)
    print(holdout.groupby(["horizon", "group"])[
        ["test2026_auc", "test2026_pr_auc", "test2026_f1"]].mean().round(3), flush=True)
    print("Results:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
