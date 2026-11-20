"""Causal decomposition features. Use at most 32 observed values, with at least 16 values. Extend one point using (last - third_last)/2, decompose, align to training-defined K, sort by descending zero-crossing rate, and select the penultimate row. The extrapolated point is synthetic, never an observed future target."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import numpy as np
sys.path.insert(0, _HERE)
import model_core as R
N = R.CONFIG['train_n']
W = R.CONFIG['feature_window']
MINW = R.CONFIG['feature_min_window']

def _extend(window):
    """Append one synthetic point with slope (window[-1] - window[-3]) / 2."""
    slope = (window[-1] - window[-3]) / 2.0
    return np.append(window, window[-1] + slope)

def one_feature(series, s, k, seed):
    """Use only series[max(0,s-W):s]. Align components before frequency sorting and return the penultimate row of the extended decomposition."""
    lo = max(0, s - W)
    extended = _extend(series[lo:s])
    matrix = R.order_by_frequency(R.align_channels(R.decompose(extended, seed), k))
    return matrix[-2]

def train_matrix(series, seed):
    """Determine K from the extended training segment; construct features for origins 16 through 85."""
    k = R.decompose(_extend(series[:N]), seed).shape[1]
    X = np.asarray([one_feature(series, s, k, seed) for s in range(MINW, N)])
    return (X, k)
