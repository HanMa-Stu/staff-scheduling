"""Daily five-model ablation. Train on observations 0:86 and evaluate 86:122 using observed-history one-step forecasting. Seeds are 42, 43 and 44. Decomposition branches use the final frequency-sorted channel and the preceding observed level. SARIMA hybrids add the preceding residual plus its predicted change to the SARIMA forecast. No model training is performed merely by importing this module."""
from __future__ import annotations
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
import argparse
import concurrent.futures as cf
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, _HERE)
import model_core as R
from decomp_feature import one_feature, train_matrix
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[2] / 'data' / 'daily'
OUT = ROOT.parents[2] / 'results' / 'forecast' / 'daily'
N = R.CONFIG['train_n']
ORDER = ['SARIMA', 'LSTM', 'SARIMA-LSTM', 'CEEMDAN-LSTM', 'SARIMA-CEEMDAN-LSTM']
RECIPE = 'reslvl'

def metrics(actual, predicted):
    """Compute MAE, RMSE and R2 on each site-seed test segment."""
    e = np.asarray(predicted, float) - np.asarray(actual, float)
    act = np.asarray(actual, float)
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), R2=float(1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum()))

def features(series, seed):
    """Construct training and test feature matrices from the observed history available at each origin."""
    (X, k) = train_matrix(series, seed)
    te = np.asarray([one_feature(series, s, k, seed) for s in range(N, len(series))])
    return (X, te, k)

def neural_pred(X, te, series, seed, base_add=None):
    """Train a decomposition branch on delta targets. Add the previous observed level/residual, then the SARIMA forecast if provided."""
    (tr, tc) = last_channel_and_level(X, te, series)
    deltas = series[1:N] - series[:N - 1]
    if len(tr) != len(deltas[-len(tr):]):
        raise ValueError('feature/label misalignment')
    previous = series[N - 1:len(series) - 1]
    if len(tc) != len(previous):
        raise ValueError('test alignment')
    fitted = R.train_delta(tr, deltas[-len(tr):], seed)
    pred = R.predict_delta(fitted, tc, previous)
    return pred if base_add is None else pred + base_add

def last_channel_and_level(X, te, series):
    """Concatenate the last frequency-sorted component and the preceding observed level, using identical alignment rules in training and testing."""
    tr_lvl = series[N - 1 - len(X):N - 1, None]
    te_lvl = series[N - 1:len(series) - 1, None]
    return (np.hstack([X[:, -1:], tr_lvl]), np.hstack([te[:, -1:], te_lvl]))

def station_job(site):
    y = pd.read_csv(DATA / f'{site}.csv', header=None).iloc[:, 0].to_numpy(float)
    if len(y) != 122 or not np.isfinite(y).all():
        raise ValueError(f'Invalid data {site}')
    act = y[N:]
    rows = []

    def add(seed, name, pred):
        row = {'site': site, 'seed': seed, 'model': name}
        row.update(metrics(act, pred))
        rows.append(row)
    (sarima_fit, sarima_pred, _) = R.sarima_path(y)
    residual = np.concatenate([y[:N] - sarima_fit, y[N:] - sarima_pred])
    for seed in R.CONFIG['seeds']:
        add(seed, 'SARIMA', sarima_pred)
        f = R.train_delta(y[:N - 1, None], y[1:N] - y[:N - 1], seed)
        pred_lstm = R.predict_delta(f, y[N - 1:-1, None], y[N - 1:-1])
        add(seed, 'LSTM', pred_lstm)
        f = R.train_delta(residual[:N - 1, None], residual[1:N] - residual[:N - 1], seed)
        add(seed, 'SARIMA-LSTM', R.predict_delta(f, residual[N - 1:-1, None], residual[N - 1:-1]) + sarima_pred)
        line = []
        for (tag, target, add_back) in (('CEEMDAN-LSTM', y, None), ('SARIMA-CEEMDAN-LSTM', residual, sarima_pred)):
            (X, te, k) = features(target, seed)
            pred = neural_pred(X, te, target, seed, add_back)
            add(seed, tag, pred)
            line.append('%s %.0f' % (tag, metrics(act, pred)['MAE']))
        print('%s seed%d LSTM %.0f SARIMA %.0f | %s' % (site, seed, metrics(act, pred_lstm)['MAE'], metrics(act, sarima_pred)['MAE'], ' '.join(line)), flush=True)
    return rows

def summarize(d):
    """Summarize site-seed metrics, ratios of mean MAE, win counts and the supplied paired Wilcoxon tests."""
    from scipy.stats import wilcoxon
    g = d.groupby('model').agg(MAE=('MAE', 'mean'), RMSE=('RMSE', 'mean'), R2=('R2', 'mean'), n_runs=('MAE', 'size')).reindex(ORDER)
    (base_l, base_s) = (g.loc['LSTM'], g.loc['SARIMA'])
    g['MAE_change_vs_LSTM_pct'] = (g.MAE / base_l.MAE - 1) * 100
    g['MAE_change_vs_SARIMA_pct'] = (g.MAE / base_s.MAE - 1) * 100
    ref = d[d.model == 'LSTM'].set_index(['site', 'seed']).sort_index()
    w = {m: x.set_index(['site', 'seed']).sort_index() for (m, x) in d.groupby('model')}
    for (col, key) in (('runs_better_than_LSTM', 'MAE'), ('p_MAE_vs_LSTM', 'MAE'), ('p_RMSE_vs_LSTM', 'RMSE'), ('p_R2_vs_LSTM', 'R2')):
        vals = []
        for m in ORDER:
            v = w[m][key].values if m != 'LSTM' else ref[key].values
            r = ref[key].values
            vals.append(0 if m == 'LSTM' and col == 'runs_better_than_LSTM' else int((v < r).sum()) if col == 'runs_better_than_LSTM' else 1.0 if m == 'LSTM' else float(wilcoxon(v, r).pvalue))
        g[col] = vals
    out = g.reset_index().rename(columns={'model': 'model'})
    out[['MAE', 'RMSE', 'R2', 'MAE_change_vs_LSTM_pct', 'MAE_change_vs_SARIMA_pct']] = out[['MAE', 'RMSE', 'R2', 'MAE_change_vs_LSTM_pct', 'MAE_change_vs_SARIMA_pct']].round(6)
    return out

def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n', type=int, default=57)
    ap.add_argument('--outdir', type=str, default=str(OUT))
    a = ap.parse_args()
    sites = sorted((p.stem for p in DATA.glob('*.csv')), key=lambda s: int(s[2:]))[:a.n]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init) as pool:
        for chunk in pool.map(station_job, sites):
            rows.extend(chunk)
    d = pd.DataFrame(rows)
    d['model'] = pd.Categorical(d.model, ORDER, ordered=True)
    d = d.sort_values(['model', 'site', 'seed']).reset_index(drop=True)
    d['model'] = d.model.astype(str)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d.to_csv(outdir / 'station_metrics.csv', index=False, encoding='utf-8-sig')
    t = summarize(d)
    t.to_csv(outdir / 'model_summary.csv', index=False, encoding='utf-8-sig')
    print('\n%d sites x %d seeds = %d model evaluations' % (len(sites), len(R.CONFIG['seeds']), len(d)))
    print('%-22s %9s %9s %8s %9s %9s %8s %9s' % ('model', 'MAE', 'RMSE', 'R2', 'relative_to_LSTM', 'relative_to_SARIMA', 'wins', 'p(MAE)'))
    for (_, r) in t.iterrows():
        print('%-22s %9.1f %9.1f %8.4f %+8.2f%% %+9.2f%% %4d/%-3d %9.2e' % (r['model'], r['MAE'], r['RMSE'], r['R2'], r['MAE_change_vs_LSTM_pct'], r['MAE_change_vs_SARIMA_pct'], r['runs_better_than_LSTM'], r['n_runs'], r['p_MAE_vs_LSTM']))
    print('\nSaved %s' % (outdir / 'station_metrics.csv'))
    print('Saved %s' % (outdir / 'model_summary.csv'))
if __name__ == '__main__':
    main()
