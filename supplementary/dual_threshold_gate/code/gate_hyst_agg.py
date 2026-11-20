"""Supplementary hysteresis analysis, excluded from the correction. Read saved branch predictions without retraining. Enter the LSTM branch at z >= 6; return to CEEMDAN-LSTM at z < 1; otherwise retain the previous state. Both branch predictions are already computed: gating selects the output and does not skip decomposition computation. Threshold-scan and comparison tables are retained as supplementary records."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = [os.path.join(HERE, '..', 'results'), os.path.join(HERE, 'results'), HERE]
RES = next((c for c in CANDS if os.path.exists(os.path.join(c, 'daily_predictions.csv'))), CANDS[0])
SRC = os.path.join(RES, 'daily_predictions.csv')
OUT13 = os.path.join(RES, 'rule_comparison.csv')
OUT15 = os.path.join(RES, 'threshold_scan.csv')
OUT16 = os.path.join(RES, 'rule_station_details.csv')
OUT19 = os.path.join(RES, 'station_metrics_6_1.csv')
(HI_FORMAL, LO_FORMAL) = (6.0, 1.0)
HI = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
LO = [0.5, 1.0, 2.0, 3.0, 4.0]
KS = [3, 4, 5, 6]

def metrics(p, a):
    e = np.asarray(p, float) - np.asarray(a, float)
    a = np.asarray(a, float)
    return (float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), float(1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum()))

def mask_A(z, hi):
    return z >= hi

def mask_B(z, hi, lo):
    s = np.zeros(len(z), bool)
    cur = bool(z[0] >= hi)
    for (i, v) in enumerate(z):
        if v >= hi:
            cur = True
        elif v < lo:
            cur = False
        s[i] = cur
    return s

def mask_K(z, theta, k):
    """Cooldown comparison: select LSTM on the trigger day and for K subsequent days; triggers during cooldown do not extend it."""
    s = np.zeros(len(z), bool)
    cool = 0
    for i in range(len(z)):
        if cool > 0:
            s[i] = True
            cool -= 1
        elif z[i] >= theta:
            s[i] = True
            cool = k
    return s

def runs_of(mask, z, hi):
    """Return lengths of selected-branch spells whose start satisfies the high threshold."""
    (out, i, n) = ([], 0, len(mask))
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if z[i] >= hi:
                out.append(j - i)
            i = j
        else:
            i += 1
    return out

def main():
    d = pd.read_csv(SRC).sort_values(['site', 'seed', 'day'])
    keys = ['site', 'seed']
    (recs, dwell) = ([], {'A': [], 'B': [], 'K': []})
    for ((s, sd), g) in d.groupby(keys):
        y = g.y.to_numpy()
        z = g.z_lag.to_numpy()
        pl = g.p_LSTM.to_numpy()
        pc = g.p_CE.to_numpy()
        recs.append({'site': s, 'seed': sd, 'rule': 'LSTM', 'parameters': '-', **dict(zip(['MAE', 'RMSE', 'R2'], metrics(pl, y))), 'LSTM_share': 1.0, 'mean_spell_length': np.nan, 'max_spell_length': np.nan, 'single_day_spell_share': np.nan})
        recs.append({'site': s, 'seed': sd, 'rule': 'CEEMDAN-LSTM', 'parameters': '-', **dict(zip(['MAE', 'RMSE', 'R2'], metrics(pc, y))), 'LSTM_share': 0.0, 'mean_spell_length': np.nan, 'max_spell_length': np.nan, 'single_day_spell_share': np.nan})
        for th in HI:
            m = mask_A(z, th)
            r = runs_of(m, z, th)
            if th == HI_FORMAL:
                dwell['A'] += r
            recs.append({'site': s, 'seed': sd, 'rule': 'A_single_threshold', 'parameters': f'{th:g}', **dict(zip(['MAE', 'RMSE', 'R2'], metrics(np.where(m, pl, pc), y))), 'LSTM_share': m.mean(), 'mean_spell_length': np.mean(r) if r else 0.0, 'max_spell_length': max(r) if r else 0, 'single_day_spell_share': np.mean([x == 1 for x in r]) if r else np.nan})
        for th in HI:
            for lo in LO:
                if lo >= th:
                    continue
                m = mask_B(z, th, lo)
                r = runs_of(m, z, th)
                if th == HI_FORMAL and lo == LO_FORMAL:
                    dwell['B'] += r
                recs.append({'site': s, 'seed': sd, 'rule': 'B_dual_threshold', 'parameters': f'{th:g}/{lo:g}', **dict(zip(['MAE', 'RMSE', 'R2'], metrics(np.where(m, pl, pc), y))), 'LSTM_share': m.mean(), 'mean_spell_length': np.mean(r) if r else 0.0, 'max_spell_length': max(r) if r else 0, 'single_day_spell_share': np.mean([x == 1 for x in r]) if r else np.nan})
        for th in [8.0]:
            for k in KS:
                m = mask_K(z, th, k)
                r = runs_of(m, z, th)
                if k == 5:
                    dwell['K'] += r
                recs.append({'site': s, 'seed': sd, 'rule': 'K_cooldown', 'parameters': f'{th:g}/K{k}', **dict(zip(['MAE', 'RMSE', 'R2'], metrics(np.where(m, pl, pc), y))), 'LSTM_share': m.mean(), 'mean_spell_length': np.mean(r) if r else 0.0, 'max_spell_length': max(r) if r else 0, 'single_day_spell_share': np.mean([x == 1 for x in r]) if r else np.nan})
    D = pd.DataFrame(recs)
    D.to_csv(OUT16, index=False, encoding='utf-8-sig')
    base = D[D.rule == 'LSTM'].set_index(keys)
    (agg, scan) = ([], [])
    for ((rule, par), g) in D[~D.rule.isin(['LSTM', 'CEEMDAN-LSTM'])].groupby(['rule', 'parameters']):
        g = g.set_index(keys).join(base[['MAE', 'RMSE', 'R2']], rsuffix='_L')
        rel = (g.MAE / g.MAE_L - 1).mean()
        p = wilcoxon(g.MAE, g.MAE_L).pvalue if (g.MAE - g.MAE_L).abs().max() > 0 else 1.0
        row = {'rule': rule, 'parameters': par, 'MAE': g.MAE.mean(), 'RMSE': g.RMSE.mean(), 'R2': g.R2.mean(), 'relative_to_LSTM': rel, 'p(MAE)': p, 'MAE_wins': int((g.MAE < g.MAE_L).sum()), 'R2_wins': int((g.R2 > g.R2_L).sum()), 'LSTM_share': g['LSTM_share'].mean(), 'mean_spell_length': g['mean_spell_length'].mean(), 'max_spell_length': g['max_spell_length'].max(), 'single_day_spell_share': g['single_day_spell_share'].mean()}
        agg.append(row)
        if rule == 'B_dual_threshold':
            (hi, lo) = par.split('/')
            scan.append({'theta_hi': float(hi), 'theta_lo': float(lo), **row})
    A = pd.DataFrame(agg)
    A.to_csv(OUT13, index=False, encoding='utf-8-sig')
    pd.DataFrame(scan).to_csv(OUT15, index=False, encoding='utf-8-sig')
    lb = base.MAE.mean()
    print('== Rule comparison (171 runs; reference LSTM MAE %.1f, RMSE %.1f, R2 %.4f) ==' % (base.MAE.mean(), base.RMSE.mean(), base.R2.mean()))
    A = A.sort_values(['rule', 'parameters'], key=lambda s: s.map({'A_single_threshold': 0, 'B_dual_threshold': 1, 'K_cooldown': 2}) if s.name == 'rule' else s)
    print(f"{'rule':8s}{'parameters':10s}{'MAE':>9s}{'RMSE':>9s}{'R2':>8s}{'relative_to_LSTM':>10s}{'p':>8s}{'MAE_wins':>7s}{'coverage':>7s}{'spell_length':>7s}{'longest':>6s}{'single_day_spell':>8s}")
    for (_, r) in A.iterrows():
        print(f"{r['rule']:8s}{r['parameters']:10s}{r.MAE:9.1f}{r.RMSE:9.1f}{r.R2:8.4f}{r['relative_to_LSTM'] * 100:+9.2f}%{r['p(MAE)']:8.4f}{r['MAE_wins']:7d}{r['LSTM_share'] * 100:6.1f}%{r['mean_spell_length']:7.2f}{r['max_spell_length']:6.0f}{r['single_day_spell_share'] * 100:7.1f}%")
    print()
    nspike = int((d.z_lag >= 8).sum())
    print(f'== Triggered spell lengths (171 runs; station-days with z >= 8: {nspike}; share {100 * nspike / len(d):.2f}%） ==')
    for (k, name) in [('A', 'A_single_threshold θ=6'), ('B', 'B_dual_threshold 6/1'), ('K', 'K_cooldown 8/K5')]:
        v = np.asarray(dwell[k], float)
        if len(v) == 0:
            continue
        print(f'  {name:16s} spells {len(v):3d} mean {v.mean():5.2f} days; median {np.median(v):4.1f} longest {int(v.max()):2d} one-day share {100 * (v == 1).mean():5.1f}% quantiles: {np.round(np.percentile(v, [25, 50, 75, 90]), 1)}')
    formal = D[(D.rule == 'B_dual_threshold') & (D.parameters == f'{HI_FORMAL:g}/{LO_FORMAL:g}')]
    formal = formal[['site', 'seed', 'MAE', 'RMSE', 'R2', 'LSTM_share']]
    formal.to_csv(OUT19, index=False, encoding='utf-8-sig')
    print()
    print('== Reported configuration theta_hi=%g, theta_lo=%g: %d runs ==' % (HI_FORMAL, LO_FORMAL, len(formal)))
    print('   MAE %.3f   RMSE %.3f   R2 %.4f   LSTM share %.1f%%' % (formal.MAE.mean(), formal.RMSE.mean(), formal.R2.mean(), 100 * formal['LSTM_share'].mean()))
    print()
    for out in (OUT13, OUT15, OUT16, OUT19):
        print('Saved ' + out)
if __name__ == '__main__':
    main()
