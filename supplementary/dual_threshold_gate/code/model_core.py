"""Shared delta-LSTM training, prediction, decomposition, and SARIMA state updates. Two LSTM layers are used. Neural weights and scalers are frozen during rolling one-step evaluation. New observations update inputs and SARIMA state only."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PyEMD import CEEMDAN
from sarima_core import fit_sarima
DEVICE = os.environ.get('TORCH_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = {'train_n': 86, 'test_n': 36, 'seeds': [42, 43, 44], 'neural': {'lookback': 1, 'hidden': 64, 'layers': 2, 'dropout': 0.2, 'epochs': 400, 'batch': 32, 'lr': 0.001, 'weight_decay': 0.0001}, 'ceemdan': {'trials': 50, 'epsilon': 0.005, 'noise_scale': 1.0, 'parallel': False}, 'seasonal_period': 7, 'feature_window': 32, 'feature_min_window': 16, 'compute_device': DEVICE, 'protocol': 'observed_history_rolling_one_step_frozen_weights|shared_scale|channel_zcr_sorted|consistent_window_decomposition', 'target': 'next_level_minus_last_observed_level'}

class DeltaLSTM(nn.Module):

    def __init__(self, channels, cfg):
        super().__init__()
        self.lstm = nn.LSTM(channels, cfg['hidden'], cfg['layers'], batch_first=True, dropout=cfg['dropout'] if cfg['layers'] > 1 else 0.0)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.fc = nn.Linear(cfg['hidden'], 1)

    def forward(self, x):
        (_, (hidden, _)) = self.lstm(x)
        return self.fc(self.dropout(hidden[-1]))

def train_delta(features, deltas, seed):
    """Fit one delta regressor. Input channels share training-only min/max; delta targets use a separate training-only scale. Optimize MSE with Adam, then freeze weights."""
    c = CONFIG['neural']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=DEVICE == 'cuda')
    x = np.asarray(features, dtype=np.float32)[:, None, :]
    y = np.asarray(deltas, dtype=np.float32)[:, None]
    (xmin, xmax) = (float(x.min()), float(x.max()))
    (ymin, ymax) = (float(y.min()), float(y.max()))
    (xs, ys) = (xmax - xmin + 1e-08, ymax - ymin + 1e-08)
    loader = DataLoader(TensorDataset(torch.tensor((x - xmin) / xs, device=DEVICE), torch.tensor((y - ymin) / ys, device=DEVICE)), batch_size=c['batch'], shuffle=True)
    model = DeltaLSTM(x.shape[2], c).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[200, 300], gamma=0.1)
    model.train()
    for _ in range(c['epochs']):
        for (bx, by) in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(bx), by)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
    model.eval()
    return (model, xmin, xs, ymin, ys)

def predict_delta(fitted, features, previous):
    (model, xmin, xs, ymin, ys) = fitted
    windows = np.asarray(features, dtype=np.float32)[:, None, :]
    with torch.no_grad():
        change = model(torch.tensor((windows - xmin) / xs, device=DEVICE)).detach().cpu().numpy() * ys + ymin
    return (change[:, 0] + np.asarray(previous, dtype=np.float32)).astype(float)

def decompose(history, seed):
    values = np.asarray(history, dtype=float)
    if np.std(values) == 0:
        return values[:, None]
    decomposer = CEEMDAN(**CONFIG['ceemdan'])
    decomposer.noise_seed(seed)
    return decomposer.ceemdan(values).T

def align_channels(raw, channels):
    """Keep up to K-1 leading components, merge all remaining components into the last channel, and pad missing intermediate channels with zeros. Check reconstruction."""
    result = np.zeros((len(raw), channels))
    head = min(channels - 1, raw.shape[1] - 1)
    result[:, :head] = raw[:, :head]
    result[:, -1] = raw[:, head:].sum(axis=1)
    if not np.allclose(result.sum(axis=1), raw.sum(axis=1), rtol=1e-08, atol=1e-08):
        raise ValueError('Component reconstruction failed')
    return result

def zero_cross_rate(series):
    sign = np.sign(series - series.mean())
    return float(np.mean(np.diff(sign) != 0)) if len(sign) > 1 else 0.0

def order_by_frequency(matrix):
    """Sort channels by descending empirical zero-crossing rate. This ordering does not guarantee invariant physical frequency bands."""
    rates = np.array([zero_cross_rate(matrix[:, j]) for j in range(matrix.shape[1])])
    return matrix[:, np.argsort(-rates)]

def sarima_path(values):
    """Fit/select SARIMA on the initial training segment only. Forecast before incorporating each actual observation with state.extend; parameters remain fixed."""
    n = CONFIG['train_n']
    (fitted, _, metadata, state) = fit_sarima(values[:n], 1, CONFIG['seasonal_period'])
    (location, scale) = (metadata['train_mean'], metadata['standardization_scale'])
    predictions = []
    for actual in values[n:]:
        predictions.append(float(np.asarray(state.forecast(1))[0] * scale + location))
        state = state.extend([(float(actual) - location) / scale])
    return (np.asarray(fitted), np.asarray(predictions), metadata)
