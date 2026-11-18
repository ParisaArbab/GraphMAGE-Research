#!/usr/bin/env python3
"""Small, reproducible research-screening agent for GraphMAGE variants.

This is a GraphMAGE-inspired PROXY, not the published authors' implementation.
It compares architectures fairly on identical observations and chronological splits.
"""
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F

CITY_CSV = 'https://data.cityofchicago.org/api/views/jqe8-8r6s/rows.csv?accessType=DOWNLOAD'
VARIANTS = {
    'baseline': (False, False, False),
    'adaptive': (True, False, False),
    'temporal': (False, True, False),
    'focal': (False, False, True),
    'adaptive_temporal': (True, True, False),
    'all_three': (True, True, True),
}

@dataclass
class Data:
    x: np.ndarray  # [T,N,F], known at forecast time
    y: np.ndarray  # [T,N], nan when not tested
    dates: list
    coords: np.ndarray  # [N,2] latitude, longitude
    node_names: list
    source: str


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def generate_synthetic(n_nodes=48, seed=7):
    """Synthetic, spatially + temporally patterned data for SMOKE TEST ONLY."""
    rng = np.random.default_rng(seed)
    years = range(2018, 2024)
    weeks = 22
    coords = np.column_stack((41.65 + .5 * rng.random(n_nodes),
                              -87.9 + .5 * rng.random(n_nodes)))
    land = rng.uniform(-1, 1, n_nodes)
    n_times = len(years) * weeks
    y = np.full((n_times, n_nodes), np.nan, np.float32)
    x = np.zeros((n_times, n_nodes, 10), dtype=np.float32)
    dates = []
    prev_status = np.zeros(n_nodes)
    prev_obs = np.zeros(n_nodes)
    t = 0
    for year in years:
        prev_status *= 0
        prev_obs *= 0
        for week in range(weeks):
            d = pd.Timestamp(year=year, month=5, day=4) + pd.Timedelta(weeks=week)
            dates.append(d)
            phase = 2 * np.pi * week / weeks
            seasonal = np.sin(phase - 1)
            geo_pattern = np.sin((coords[:, 0] - 41.65) * 13) * np.cos((coords[:, 1] + 87.9) * 9)
            temp = .65 * seasonal + .35 * geo_pattern + rng.normal(0, .22, n_nodes)
            rain = -.25 * seasonal + .35 * land + rng.normal(0, .3, n_nodes)
            latent = (-1.35 + 1.15 * temp + .65 * rain + .65 * land
                      + .6 * prev_status + .35 * geo_pattern)
            prob = 1 / (1 + np.exp(-latent))
            label = rng.binomial(1, prob).astype(np.float32)
            observed = rng.random(n_nodes) < .84
            y[t, observed] = label[observed]
            # History only from time t-1, not current outcome, and never future outcomes.
            x[t] = np.stack(((coords[:, 0]-41.9)*4, (coords[:, 1]+87.65)*4,
                             np.full(n_nodes, np.sin(phase)), np.full(n_nodes, np.cos(phase)),
                             land, temp, rain, prev_status, prev_obs,
                             np.full(n_nodes, 1.)), axis=-1).astype(np.float32)
            prev_status = np.where(observed, label, prev_status)
            prev_obs = observed.astype(float)
            t += 1
    return Data(x, y, dates, coords, [f'Trap_{i:03d}' for i in range(n_nodes)],
                'synthetic (programmatically generated; not clinical evidence)')


def _pick_col(df, *candidates, required=True):
    canonical = {''.join(c.lower().split('_')): c for c in df.columns}
    for s in candidates:
        key = ''.join(s.lower().split('_'))
        if key in canonical:
            return canonical[key]
    if required:
        raise ValueError(f'Missing one of {candidates}; CSV has columns: {list(df.columns)}')
    return None


def load_chicago(path, max_traps=48, through_year=2025):
    """Read public City of Chicago surveillance data. Never label untested weeks as negative."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    trap = _pick_col(df, 'trap', 'trap_id')
    lat = _pick_col(df, 'latitude')
    lon = _pick_col(df, 'longitude')
    result = _pick_col(df, 'result', 'wnvpresent', 'wnv_present', 'test_result')
    date = _pick_col(df, 'test_date', 'date', 'collection_date', 'date_collected', required=False)
    if date:
        df['_date'] = pd.to_datetime(df[date], errors='coerce')
    else:
        yr = _pick_col(df, 'season_year', 'year')
        wk = _pick_col(df, 'week', 'week_number')
        yr_data = pd.to_numeric(df[yr], errors='coerce')
        wk_data = pd.to_numeric(df[wk], errors='coerce')
        df['_date'] = pd.to_datetime(yr_data.astype('Int64').astype(str) + '-W' +
                        wk_data.astype('Int64').astype(str).str.zfill(2) + '-1',
                        format='%G-W%V-%u', errors='coerce')
    df['_lat'] = pd.to_numeric(df[lat], errors='coerce')
    df['_lon'] = pd.to_numeric(df[lon], errors='coerce')
    df['_trap'] = df[trap].astype(str).str.strip()
    positive = {'positive', 'pos', '1', 'true', 'yes', 'wnv positive'}
    negative = {'negative', 'neg', '0', 'false', 'no', 'wnv negative'}
    label_text = df[result].astype(str).str.strip().str.lower()
    df['_y'] = np.where(label_text.isin(positive), 1.,
                        np.where(label_text.isin(negative), 0., np.nan))
    df = df.dropna(subset=['_date', '_lat', '_lon', '_y'])
    df = df[(df['_date'].dt.year <= through_year) &
            (df['_lat'].between(40, 43)) & (df['_lon'].between(-89, -86))]
    if df.empty:
        raise ValueError('No valid labeled trap records after cleaning. Check schema/result values.')
    # Choose trap population using ONLY years earlier than validation/test.
    all_years = sorted(df['_date'].dt.year.unique().tolist())
    if len(all_years) < 5:
        raise ValueError('Need >= 5 calendar years for a trustworthy temporal pilot.')
    training_years = all_years[:-3]
    train_df = df[df['_date'].dt.year.isin(training_years)]
    top = train_df['_trap'].value_counts().head(max_traps).index.tolist()
    df = df[df['_trap'].isin(top)].copy()
    pos = train_df[train_df['_trap'].isin(top)].groupby('_trap')[['_lat', '_lon']].median()
    nodes = [n for n in top if n in pos.index]
    if len(nodes) < 5:
        raise ValueError('Fewer than 5 traps with training history.')
    coords = pos.loc[nodes, ['_lat', '_lon']].to_numpy(dtype=np.float32)
    # Only observed test results count as a label. Aggregate pools in the same trap/week.
    df['_week'] = df['_date'].dt.to_period('W-SUN').dt.start_time
    labels = df.groupby(['_week', '_trap'])['_y'].max()
    years = all_years
    week_dates = []
    # Seasonal sampling window includes months May-Oct, including empty weeks (unobserved masks).
    for year in years:
        start = pd.Timestamp(year, 5, 1).to_period('W-SUN').start_time
        end = pd.Timestamp(year, 10, 31).to_period('W-SUN').start_time
        week_dates.extend(pd.date_range(start, end, freq='7D').tolist())
    n_time, n_node = len(week_dates), len(nodes)
    y = np.full((n_time, n_node), np.nan, np.float32)
    pos_lookup = {k: i for i, k in enumerate(nodes)}
    date_lookup = {d: i for i, d in enumerate(week_dates)}
    for (week, trap_id), val in labels.items():
        i, j = date_lookup.get(week), pos_lookup.get(trap_id)
        if i is not None and j is not None:
            y[i, j] = val
    x = np.zeros((n_time, n_node, 10), np.float32)
    historical = np.zeros(n_node)
    observed_prev = np.zeros(n_node)
    pos_recent = np.zeros((3, n_node))
    for t, date_val in enumerate(week_dates):
        if t == 0 or date_val.year != week_dates[t-1].year:
            historical[:] = 0
            observed_prev[:] = 0
            pos_recent[:] = 0
        phase = 2 * np.pi * ((date_val.dayofyear - 120) / 183)
        x[t] = np.stack(((coords[:, 0]-41.9)*4, (coords[:, 1]+87.65)*4,
                         np.full(n_node, np.sin(phase)), np.full(n_node, np.cos(phase)),
                         np.zeros(n_node), np.zeros(n_node), np.zeros(n_node),
                         historical, observed_prev,
                         pos_recent.mean(axis=0)), axis=-1)
        # Test labels become available no earlier than the following weekly step.
        tested = np.isfinite(y[t]); observed_prev = tested.astype(float)
        historical = np.where(tested, y[t], historical)
        pos_recent = np.roll(pos_recent, 1, axis=0)
        pos_recent[0] = np.where(tested, y[t], 0.)
    return Data(x, y, week_dates, coords, nodes,
                f'Chicago public mosquito trap tests, {min(years)}-{max(years)}; weather not included')


def acquire_chicago(cache_path, force=False):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not cache_path.exists():
        try:
            import urllib.request
            request = urllib.request.Request(CITY_CSV, headers={'User-Agent':'GraphMAGE-screening-agent/0.1'})
            with urllib.request.urlopen(request, timeout=90) as resp, open(cache_path, 'wb') as out:
                import shutil
                shutil.copyfileobj(resp, out)
        except Exception as e:
            raise RuntimeError('Could not download Chicago data. Run on a machine with internet, '
                               'or pass --csv /path/to/chicago.csv. Original error: '+str(e)) from e
    return cache_path


def make_knn(coords, k=6):
    lat = np.deg2rad(coords[:, 0]); lon = np.deg2rad(coords[:, 1]);
    a = np.sin((lat[:, None]-lat[None,:])/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    distance = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(distance, np.inf)
    k = min(k, len(coords)-1)
    nbr = np.argsort(distance, axis=1)[:, :k]
    weights = np.zeros_like(distance, dtype=np.float32)
    rows = np.arange(len(coords))[:, None]
    d = distance[rows, nbr]
    scale = max(float(np.median(d)), .01)
    weights[rows, nbr] = np.exp(-d/scale)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


class TinyGraphMAGE(nn.Module):
    """Lightweight GraphMAGE recurrence, with optional temporal & learned-edge modules."""
    def __init__(self, feature_dim, hidden=24, depth=3, adaptive=False, temporal=False):
        super().__init__()
        self.adaptive = adaptive
        self.temporal = temporal
        self.embed = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU())
        if temporal:
            self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden, hidden),
                                                  nn.ReLU()) for _ in range(depth)])
        if adaptive:
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, history, graph):
        # history: B,N,L,F (time always increases across L).
        if self.temporal:
            b,n,l,f = history.shape
            _, h = self.gru(history.reshape(b*n,l,f))
            h0 = self.embed(history[:, :, -1]) + h[-1].reshape(b,n,-1)
        else:
            h0 = self.embed(history[:, :, -1])
        if self.adaptive:
            scale = h0.shape[-1] ** -.5
            logits = torch.matmul(self.query(h0), self.key(h0).transpose(1,2))*scale
            allowed = graph.unsqueeze(0) > 0
            logits = logits.masked_fill(~allowed, -1e5)
            learned = torch.softmax(logits, dim=-1)
            adj = .5 * graph.unsqueeze(0) + .5 * learned
        else:
            adj = graph.unsqueeze(0)
        h = h0
        for layer in self.layers:
            neighbors = torch.bmm(adj.expand(h.shape[0], -1, -1), h)
            h = layer(torch.cat((h0, neighbors), dim=-1))
        return self.out(h).squeeze(-1)


def as_windows(data, horizon=3, history=4):
    xs = [] ; ys = []; years = [] ; dates = []
    by_date = {pd.Timestamp(dt): idx for idx, dt in enumerate(data.dates)}
    for t, dt in enumerate(data.dates):
        target_date = pd.Timestamp(dt) + pd.Timedelta(weeks=horizon)
        target_id = by_date.get(target_date)
        if target_id is None or target_date.year != pd.Timestamp(dt).year:
            continue
        if not np.isfinite(data.y[target_id]).any():
            continue
        start = max(0, t-history+1)
        if pd.Timestamp(data.dates[start]).year != pd.Timestamp(dt).year:
            # Pad with 0 rather than importing older season's labels or weather.
            start = next(i for i in range(start, t+1) if pd.Timestamp(data.dates[i]).year == pd.Timestamp(dt).year)
        window = data.x[start:t+1]
        if window.shape[0] < history:
            window = np.pad(window, ((history-window.shape[0],0),(0,0),(0,0)))
        xs.append(np.transpose(window,(1,0,2)))
        ys.append(data.y[target_id]); years.append(target_date.year); dates.append(str(target_date.date()))
    if not xs:
        raise ValueError('No labeled future targets. Check horizon and observations.')
    return np.stack(xs), np.stack(ys), np.array(years), dates


def year_split(years):
    distinct = sorted(set(years.tolist()))
    if len(distinct) < 4:
        raise ValueError(f'Need at least 4 years with targets; have {distinct}')
    train_years = distinct[:-3]; val_year = distinct[-3]; test_years = distinct[-2:]
    train = np.flatnonzero(np.isin(years, train_years))
    val = np.flatnonzero(years == val_year)
    test = np.flatnonzero(np.isin(years, test_years))
    return train, val, test, {'train':train_years, 'validation':[val_year], 'test':test_years}


def metrics(y, prob, threshold):
    mask = np.isfinite(y)
    true = y[mask].astype(int)
    score = prob[mask]
    if len(true) == 0 or len(np.unique(true)) < 2:
        return {'n':len(true), 'auc':np.nan,'pr_auc':np.nan,'f1':np.nan,
                'sensitivity':np.nan,'precision':np.nan,'brier':np.nan}
    predicted = (score >= threshold).astype(int)
    return {'n':int(len(true)), 'prevalence':round(float(np.mean(true)),4),
            'auc':float(roc_auc_score(true,score)),
            'pr_auc':float(average_precision_score(true,score)),
            'f1':float(f1_score(true,predicted,zero_division=0)),
            'sensitivity':float(recall_score(true,predicted,zero_division=0)),
            'precision':float(precision_score(true,predicted,zero_division=0)),
            'brier':float(brier_score_loss(true,score))}


def validation_threshold(y, prob):
    mask = np.isfinite(y); true=y[mask].astype(int); scores=prob[mask]
    if not len(true): return .5
    candidates = np.quantile(scores, np.linspace(.1,.9,41)).tolist()
    # Must be validation only; never tune on test.
    return float(max(candidates,key=lambda th: f1_score(true,scores >= th,zero_division=0)))


def loss_fn(logits, target, focal=False, pos_weight=1.):
    mask = torch.isfinite(target)
    y = target[mask]; logit = logits[mask]
    if not len(y): return logits.sum()*0
    ce = F.binary_cross_entropy_with_logits(logit,y,reduction='none')
    weights = torch.where(y > .5, pos_weight, 1.)
    if focal:
        p_t = torch.exp(-ce)
        ce = ce * (1-p_t)**2
    return (ce * weights).mean()


def run_variant(variant, seed, x, y, train_idx, val_idx, test_idx, graph, epochs=18, batch_size=16):
    set_seed(seed)
    adaptive, temporal, focal = VARIANTS[variant]
    model = TinyGraphMAGE(x.shape[-1], adaptive=adaptive, temporal=temporal)
    optim = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.004)
    tx = torch.from_numpy(x).float(); ty = torch.from_numpy(y).float()
    valid_train_labels = y[train_idx][np.isfinite(y[train_idx])]
    n_pos = max(1.,float(np.sum(valid_train_labels>0.5)))
    n_neg = float(np.sum(valid_train_labels<=0.5))
    pos_weight = min(8., max(1., n_neg/n_pos))
    best_score = -1.; best_epoch = 0; best_state = None; patience = 5; stalls = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        for chunk in np.array_split(perm, max(1, int(np.ceil(len(perm)/batch_size)))):
            optim.zero_grad()
            logits = model(tx[chunk],graph)
            loss = loss_fn(logits, ty[chunk], focal=focal, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optim.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(tx[val_idx], graph)).numpy()
        m = metrics(y[val_idx], scores, .5)
        score = .6*m['auc'] + .4*m['pr_auc']
        if np.isfinite(score) and score > best_score+1e-4:
            best_score = score; best_epoch=epoch+1; stalls=0
            best_state = {key:value.detach().clone() for key,value in model.state_dict().items()}
        else:
            stalls += 1
        if stalls>=patience: break
    if best_state is None:
        raise RuntimeError(f'No valid checkpoint for {variant}; check label prevalence in validation.')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(model(tx[val_idx],graph)).numpy()
        test_score = torch.sigmoid(model(tx[test_idx],graph)).numpy()
    threshold = validation_threshold(y[val_idx],val_score)
    val_m = metrics(y[val_idx],val_score,threshold)
    test_m = metrics(y[test_idx],test_score,threshold)
    return {'variant':variant,'seed':seed,'threshold':threshold,'best_epoch':best_epoch,
            'parameters':sum(p.numel() for p in model.parameters()),
            'seconds':round(time.perf_counter()-t0,3),
            **{f'val_{k}':v for k,v in val_m.items()},
            **{f'test_{k}':v for k,v in test_m.items()}}


def save_report(df, metadata, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'runs.csv',index=False)
    summary = df.groupby('variant',sort=False).agg(
        runs=('seed','size'),val_auc=('val_auc','mean'),val_pr_auc=('val_pr_auc','mean'),
        test_auc=('test_auc','mean'),test_pr_auc=('test_pr_auc','mean'),
        test_f1=('test_f1','mean'),test_sensitivity=('test_sensitivity','mean'),
        test_brier=('test_brier','mean'),seconds=('seconds','sum'))
    summary['selection_score'] = .6*summary.val_auc + .4*summary.val_pr_auc
    summary = summary.sort_values('selection_score',ascending=False)
    summary.to_csv(out/'summary.csv')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    fig,ax=plt.subplots(figsize=(10,4.8))
    names=summary.index.tolist(); xs=np.arange(len(names)); width=.38
    ax.bar(xs-width/2,summary.test_auc,width,label='Test ROC-AUC')
    ax.bar(xs+width/2,summary.test_pr_auc,width,label='Test PR-AUC')
    ax.set_xticks(xs); ax.set_xticklabels(names,rotation=30,ha='right')
    ax.set_ylim(0,1); ax.set_title('Held-out years: model screening (not the paper replication)')
    ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    winner=summary.index[0]; baseline=summary.loc['baseline']
    best=summary.loc[winner]
    # Conservative screening rule. Avoid promoting a complicated model for negligible gains.
    promoted = (winner != 'baseline' and metadata['seeds'] >= 3 and
                best.val_auc >= baseline.val_auc + .01 and
                best.val_pr_auc >= baseline.val_pr_auc + .01)
    status = ('POTENTIAL CANDIDATE FOR REPLICATION' if promoted else
              'INCONCLUSIVE: no validated improvement above the pilot screening threshold')
    (out/'decision.json').write_text(json.dumps({'status':status, 'validation_selected':winner,
        'promotion_gate':{'min_seeds':3,'val_auc_gain':.01,'val_pr_auc_gain':.01},
        'delta_val_auc':float(best.val_auc-baseline.val_auc),
        'delta_val_pr_auc':float(best.val_pr_auc-baseline.val_pr_auc)},indent=2))
    lines=['# Automated GraphMAGE Pilot Report','',
           f"**Data source:** {metadata['source']}",
           f"**Years:** train {metadata['split']['train']}, validation {metadata['split']['validation']}, test {metadata['split']['test']}",
           f"**Horizon:** {metadata['horizon']} week(s); **traps:** {metadata['n_traps']}; **seeds:** {metadata['seeds']}",
           '',
           '**Important:** These experiments use a lightweight GraphMAGE-inspired proxy, not the authors\' official implementation. '
           'Real-data results, if any, should not be compared directly with published paper metrics. '
           'Synthetic-data runs only show that the experiment pipeline works.',
           '', '## Screening result',
           f'**Decision:** {status}.',
           f"The **validation-selected** candidate is `{winner}` (validation 0.6 ROC-AUC + 0.4 PR-AUC).",
           f"Its held-out test ROC-AUC is **{best.test_auc:.3f}** vs **{baseline.test_auc:.3f}** for the proxy baseline; "
           f"PR-AUC is **{best.test_pr_auc:.3f}** vs **{baseline.test_pr_auc:.3f}**.",
           '', 'A better result on a small pilot is a **hypothesis for replication**, not evidence of a guaranteed improvement.',
           '', '## Full comparison (mean across seeds)', '',
           '| Variant | Val AUC | Val PR-AUC | Test AUC | Test PR-AUC | Test F1 | Test Sensitivity | Test Brier |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in summary.iterrows():
        lines.append(f'| {name} | {row.val_auc:.3f} | {row.val_pr_auc:.3f} | {row.test_auc:.3f} | '
                     f'{row.test_pr_auc:.3f} | {row.test_f1:.3f} | {row.test_sensitivity:.3f} | {row.test_brier:.3f} |')
    lines += ['', '## Recommended next steps',
              '1. Repeat this screening with 5-20 seeds on real Illinois data and multiple forecast horizons.',
              '2. Replace the proxy with the published authors\' GraphMAGE code when available, preserving identical data and splits.',
              '3. Add actual weather and land-cover covariates; Chicago-only mode does not test climate awareness.',
              '4. Test unseen rural traps and uncertainty intervals before claiming a real improvement.',
              '5. Keep the test years untouched during architecture/hyperparameter selection.', '',
              '## Leakage safeguards',
              '- Future labels are never placed into input features.',
              '- Earlier observed outcomes may appear as lagged history, but the target week outcome is masked.',
              '- Locations are selected from training years only in Chicago mode.',
              '- Model checkpoints, thresholds, and preferred variant use validation observations only.',
              '- Test data are used for reporting after selection, not for threshold tuning.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',choices=['synthetic','chicago'],default='synthetic')
    parser.add_argument('--csv',help='local Chicago export CSV, avoids download')
    parser.add_argument('--cache',default='data/chicago_wnv.csv')
    parser.add_argument('--max-traps',type=int,default=48)
    parser.add_argument('--horizon',type=int,default=3)
    parser.add_argument('--history',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=18)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    parser.add_argument('--out',default='results/pilot')
    args=parser.parse_args()
    if args.seeds < 1 or args.horizon < 1 or args.history < 1:
        parser.error('seeds, horizon, and history must be >=1')
    set_seed(7)
    if args.data=='synthetic':
        data=generate_synthetic(n_nodes=args.max_traps)
    else:
        path = Path(args.csv) if args.csv else acquire_chicago(args.cache)
        data=load_chicago(path, args.max_traps)
    xs,ys,years,dates=as_windows(data, args.horizon, args.history)
    tr,va,te,split=year_split(years)
    graph=make_knn(data.coords)
    metadata={'source':data.source,'horizon':args.horizon,'history':args.history,
              'n_traps':len(data.coords),'seeds':args.seeds,'split':split,
              'n_windows':len(years),'examples_train':int(np.isfinite(ys[tr]).sum()),
              'examples_validation':int(np.isfinite(ys[va]).sum()),
              'examples_test':int(np.isfinite(ys[te]).sum()),
              'variants':args.variants,'dataset_citation':'https://data.cityofchicago.org/d/jqe8-8r6s' if args.data=='chicago' else 'synthetic data',
              'paper':'https://arxiv.org/abs/2509.22657'}
    print('Dataset:',metadata['source'])
    print('Splits:',split,'/ labeled observations:',metadata['examples_train'],metadata['examples_validation'],metadata['examples_test'],flush=True)
    results=[]
    for name in args.variants:
        for seed in range(42,42+args.seeds):
            run=run_variant(name,seed,xs,ys,tr,va,te,graph,epochs=args.epochs)
            results.append(run)
            print(f"{name} seed={seed}: val_auc={run['val_auc']:.3f} val_pr_auc={run['val_pr_auc']:.3f} test_auc={run['test_auc']:.3f}",flush=True)
    df=pd.DataFrame(results)
    if 'baseline' not in df.variant.to_list():
        raise ValueError('Include baseline in --variants for valid improvement comparisons.')
    summary=save_report(df,metadata,args.out)
    print('\nValidation-ranked comparisons:\n',summary[['val_auc','val_pr_auc','test_auc','test_pr_auc','test_f1']].round(3))
    print('\nWritten to',Path(args.out).resolve())

if __name__=='__main__':
    main()