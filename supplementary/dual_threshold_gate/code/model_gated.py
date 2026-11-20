"""Supplementary branch-training script, excluded from the correction analysis. Train LSTM and CEEMDAN-LSTM branches once and save their daily forecasts. This script implements the older cooldown comparison; the reported 6/1 hysteresis rule is in gate_hyst_agg.py. Gate inputs use preceding observed test values and a training-only scale, never the current or future target."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import argparse
import concurrent.futures as cf
import numpy as np
import pandas as pd
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_core as R
from decomp_feature import one_feature, train_matrix
DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'daily'))
OUTDIR = os.path.join(HERE, '..', 'results')
(N, W, MINW) = (86, 32, 16)
SEEDS = [42, 43, 44]
(THETA, K) = (2.0, 1)
MODELS = ['naive', 'LSTM', 'CEEMDAN-LSTM', 'Gated-CEEMDAN-LSTM']
CONFIG = {'train_n': N, 'test_n': 36, 'seeds': SEEDS, 'feature_window': W, 'feature_min_window': MINW, 'lstm': {'lookback': 1, 'hidden': 64, 'layers': 2, 'dropout': 0.2, 'epochs': 400, 'batch': 32, 'lr': 0.001, 'weight_decay': 0.0001}, 'ceemdan': {'trials': 50, 'epsilon': 0.005, 'noise_scale': 1.0}, 'ceemdan_branch': {'endpoint_extension': 'extR', 'extension_points': 1, 'channel': 'residual', 'with_latest_observation': True}, 'gate': {'signal': 'abs(delta[t-1])/sigma_delta_train', 'theta': THETA, 'cooldown_days': K, 'branch_on_trigger': 'LSTM', 'estimated_on': 'training segment only'}, 'target': 'next_level_minus_last_observed_level', 'protocol': 'observed_history_rolling_one_step_frozen_weights'}

def metrics(pred, act):
    e = np.asarray(pred) - np.asarray(act)
    act = np.asarray(act)
    return (float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), float(1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum()))

def gate_mask(y, theta=THETA, k=K):
    """Construct the cooldown comparison mask from lagged observed changes normalized by the training-difference standard deviation."""
    sd = float(np.diff(y[:N]).std())
    T = len(y) - N
    out = np.zeros(T, bool)
    cool = 0
    for i in range(T):
        t = N + i
        z = abs(y[t - 1] - y[t - 2]) / sd
        if cool > 0:
            out[i] = True
            cool -= 1
        elif z >= theta:
            out[i] = True
            cool = k
    return out

def job(payload):
    (site, seed, theta, kk) = payload
    y = pd.read_csv(os.path.join(DATA, f'{site}.csv'), header=None).iloc[:, 0].to_numpy(float)
    (prev, act) = (y[N - 1:-1], y[N:])
    dtr = y[1:N] - y[:N - 1]
    sd = float(np.diff(y[:N]).std())
    f = R.train_delta(y[:N - 1, None], dtr, seed)
    p_lstm = R.predict_delta(f, prev[:, None], prev)
    (X, k) = train_matrix(y, seed)
    te = np.asarray([one_feature(y, s, k, seed) for s in range(N, len(y))])
    (tr_lvl, te_lvl) = (y[15:15 + len(X), None], prev[:, None])
    f = R.train_delta(np.hstack([X[:, -1:], tr_lvl]), dtr[-len(X):], seed)
    p_ce = R.predict_delta(f, np.hstack([te[:, -1:], te_lvl]), prev)
    g = gate_mask(y, theta, kk)
    p_gate = np.where(g, p_lstm, p_ce)
    days = [{'site': site, 'seed': seed, 'day': i + 1, 'y': float(act[i]), 'prev': float(prev[i]), 'z_lag': abs(float(y[N + i - 1]) - float(y[N + i - 2])) / sd, 'p_LSTM': float(p_lstm[i]), 'p_CE': float(p_ce[i])} for i in range(len(act))]
    out = []
    for (name, p) in (('naive', prev), ('LSTM', p_lstm), ('CEEMDAN-LSTM', p_ce), ('Gated-CEEMDAN-LSTM', p_gate)):
        (mae, rmse, r2) = metrics(p, act)
        out.append({'site': site, 'seed': seed, 'model': name, 'k': k, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'lstm_share': float(g.mean())})
    print('%s seed%d LSTM %.0f CE %.0f Gate %.0f (%.0f%% LSTM)' % (site, seed, out[1]['MAE'], out[2]['MAE'], out[3]['MAE'], g.mean() * 100), flush=True)
    return {'metrics': out, 'days': days}

def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n', type=int, default=57)
    ap.add_argument('--theta', type=float, default=THETA)
    ap.add_argument('--cooldown', type=int, default=K)
    ap.add_argument('--out', default=os.path.join(OUTDIR, 'branch_metrics.csv'))
    ap.add_argument('--dump', default=os.path.join(OUTDIR, 'daily_predictions.csv'))
    a = ap.parse_args()
    (THETA, K) = (a.theta, a.cooldown)
    CONFIG['gate'].update(theta=THETA, cooldown_days=K)
    sites = sorted((f[:-4] for f in os.listdir(DATA) if f.endswith('.csv')), key=lambda s: int(s[2:]))[:a.n]
    (rows, days) = ([], [])
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=init) as pool:
        for chunk in pool.map(job, [(s, sd, a.theta, a.cooldown) for s in sites for sd in SEEDS]):
            rows.extend(chunk['metrics'])
            days.extend(chunk['days'])
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False, encoding='utf-8-sig')
    pd.DataFrame(days).to_csv(a.dump, index=False, encoding='utf-8-sig')
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), 'gate_config.json'), 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
    from scipy.stats import wilcoxon
    print('\n%d sites x %d seeds = %d runs; seeds=%s' % (len(sites), len(SEEDS), len(sites) * len(SEEDS), SEEDS))
    print('%-24s %9s %9s %9s %10s %10s' % ('model', 'MAE', 'RMSE', 'R2', 'relative_to_LSTM', 'relative_to_naive'))
    base = d[d.model == 'LSTM']
    nb = d[d.model == 'naive']
    for m in MODELS:
        g = d[d.model == m]
        print('%-24s %9.1f %9.1f %9.4f %+9.2f%% %+9.2f%%' % (m, g.MAE.mean(), g.RMSE.mean(), g.R2.mean(), (g.MAE.mean() / base.MAE.mean() - 1) * 100, (g.MAE.mean() / nb.MAE.mean() - 1) * 100))
    print('\nPaired tests by site and seed (n=%d)' % len(base))
    piv = d.pivot_table(index=['site', 'seed'], columns='model', values=['MAE', 'RMSE', 'R2'])
    for m in ['CEEMDAN-LSTM', 'Gated-CEEMDAN-LSTM']:
        print('  %-22s vs LSTM: MAE %+.2f%%  RMSE %+.2f%%  R2 %+.4f  wins %d/%d  p=%.4f' % (m, (piv['MAE'][m] / piv['MAE']['LSTM'] - 1).mean() * 100, (piv['RMSE'][m] / piv['RMSE']['LSTM'] - 1).mean() * 100, (piv['R2'][m] - piv['R2']['LSTM']).mean(), (piv['MAE'][m] < piv['MAE']['LSTM']).sum(), len(piv), wilcoxon(piv['MAE'][m], piv['MAE']['LSTM']).pvalue))
    print('\nLSTM branch share: mean %.0f%%, range %.0f%% to %.0f%%' % (d[d.model == 'Gated-CEEMDAN-LSTM'].lstm_share.mean() * 100, d[d.model == 'Gated-CEEMDAN-LSTM'].lstm_share.min() * 100, d[d.model == 'Gated-CEEMDAN-LSTM'].lstm_share.max() * 100))
