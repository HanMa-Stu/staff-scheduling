"""Training-only SARIMA selection from the supplied Package candidate grid. The fitted state is returned for subsequent fixed-parameter state updates."""
import math
import warnings
from typing import Any
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def _finite(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be nonempty and finite')
    return array

def _series(values: Any, name: str='training series') -> np.ndarray:
    array = _finite(values, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f'{name} must have one channel')
    return array

def fit_sarima(train: Any, steps: int, seasonal_period: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a converged stable SARIMA using only the provided training data.

    Identical deterministic results can be shared between all three SARIMA
    families and all repeated neural-network seeds at the same training stage.
    Rejected candidates are retained in the JSON-compatible diagnostics.
    """
    series = _series(train)
    if int(steps) < 1 or len(series) < 3:
        raise ValueError('SARIMA requires a positive forecast horizon and at least three points')
    period = int(seasonal_period)
    location = float(series.mean())
    observed_std = float(series.std())
    scale = observed_std if observed_std > 0.0 else 1.0
    normalized = _finite((series - location) / scale, 'standardized SARIMA training series')
    orders = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
    seasonal_orders = [(0, 0, 0, 0)]
    if period > 1 and len(series) >= 2 * period:
        seasonal_orders.extend([(1, 0, 0, period), (0, 0, 1, period), (1, 0, 1, period)])
    diagnostics = []
    best = None
    best_score = math.inf
    for order in orders:
        for seasonal_order in seasonal_orders:
            diagnostic: dict[str, Any] = {'order': list(order), 'seasonal_order': list(seasonal_order), 'accepted': False}
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    result = SARIMAX(normalized, order=order, seasonal_order=seasonal_order, trend='n', enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=250)
                    converged = bool(result.mle_retvals.get('converged', False))
                    aic = float(result.aic)
                    bic = float(result.bic)
                    diagnostic['converged'] = converged
                    diagnostic['aic'] = aic if np.isfinite(aic) else None
                    diagnostic['bic'] = bic if np.isfinite(bic) else None
                    diagnostic['optimizer_iterations'] = int(result.mle_retvals.get('iterations', 0))
                    if not converged:
                        raise RuntimeError('optimizer did not converge')
                    if not np.isfinite(aic + bic):
                        raise FloatingPointError('AIC plus BIC is not finite')
                    _finite(result.params, 'SARIMA parameters')
                    fitted = _series(result.fittedvalues, 'SARIMA in-sample fitted values') * scale + location
                    prediction = _series(result.get_forecast(steps=int(steps)).predicted_mean, 'SARIMA forecast') * scale + location
                    _finite(fitted, 'unscaled SARIMA fitted values')
                    _finite(prediction, 'unscaled SARIMA forecast')
                    if len(fitted) != len(series) or len(prediction) != int(steps):
                        raise ValueError('SARIMA produced an incorrect output length')
                    diagnostic['accepted'] = True
                    diagnostic['score'] = aic + bic
                    if aic + bic < best_score:
                        best_score = aic + bic
                        best = (fitted, prediction, list(order), list(seasonal_order))
                        best_state = result
            except Exception as exc:
                diagnostic['rejection'] = f'{type(exc).__name__}: {exc}'
            diagnostic['warnings'] = list(dict.fromkeys((str(item.message) for item in captured)))
            diagnostics.append(diagnostic)
    if best is None:
        raise RuntimeError(f'All SARIMA candidates failed; diagnostics={diagnostics!r}')
    (fitted, prediction, order, seasonal_order) = best
    metadata = {'order': order, 'seasonal_order': seasonal_order, 'selection': 'minimum training AIC+BIC among finite converged candidates', 'selected_score': float(best_score), 'train_mean': location, 'train_std': observed_std, 'standardization_scale': scale, 'train_length': int(len(series)), 'forecast_steps': int(steps), 'seasonal_period': period, 'enforce_stationarity': True, 'enforce_invertibility': True, 'maxiter': 250, 'candidate_diagnostics': diagnostics}
    return (fitted, prediction, metadata, best_state)
