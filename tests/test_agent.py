import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from agent import (generate_synthetic, as_windows, year_split, make_knn,
                   TinyGraphMAGE, validation_threshold)
import torch


def test_history_lags_and_target_separation():
    d = generate_synthetic(n_nodes=12)
    x,y,years,dates=as_windows(d,horizon=3,history=4)
    assert x.shape[1:] == (12,4,10)
    assert not np.any(np.isnan(x))
    assert len(y)==len(x)
    assert dates[0] == str((pd.Timestamp(d.dates[0])+pd.Timedelta(weeks=3)).date())


def test_year_split_disjoint():
    d=generate_synthetic(n_nodes=12)
    x,y,years,dates=as_windows(d)
    tr,va,te,sp=year_split(years)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    assert max(sp['train']) < min(sp['validation']) < min(sp['test'])


def test_graph_and_models():
    d=generate_synthetic(n_nodes=10)
    graph=make_knn(d.coords)
    assert graph.shape == (10,10)
    assert torch.allclose(graph.sum(dim=1),torch.ones(10),atol=1e-5)
    x=torch.randn(2,10,4,10)
    for a,t in ((False,False),(True,False),(False,True),(True,True)):
        m=TinyGraphMAGE(10,adaptive=a,temporal=t)
        out=m(x,graph)
        assert out.shape==(2,10)
        out.mean().backward()
        assert all(p.grad is not None for p in m.parameters())


def test_threshold():
    y=np.array([[0,1,np.nan],[1,0,1.]])
    p=np.array([[.1,.8,.9],[.6,.2,.7]])
    assert 0 <= validation_threshold(y,p) <= 1


def test_city_ingestion_and_unobserved_mask(tmp_path):
    from agent import load_chicago
    rows=[]
    # Six years, enough for chronological train / validation / test split.
    for year in range(2018,2024):
        for wk in range(12):
            for j in range(8):
                if j == 0 and wk == 5:
                    continue  # A missing trap-week is unknown, NOT negative.
                rows.append({'TRAP': f'T{j}', 'LATITUDE':41.7+j*.01,
                             'LONGITUDE':-87.8+j*.01, 'TEST DATE':str(
                    pd.Timestamp(year=year,month=5,day=8)+pd.Timedelta(weeks=wk)),
                             'RESULT':'Positive' if (wk+j)%9 == 0 else 'Negative'})
    p=tmp_path/'traps.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    data=load_chicago(p, max_traps=8,through_year=2023)
    assert data.x.shape[1:] == (8,10)
    assert len(data.dates)>70
    assert np.isnan(data.y).sum()>0
    assert np.isfinite(data.y).sum()>0
    x,y,years,_ = as_windows(data,horizon=3,history=4)
    tr,va,te,split = year_split(years)
    assert len(tr)>0 and len(va)>0 and len(te)>0