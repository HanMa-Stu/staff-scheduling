"""Shared, train-only forecasting implementations for the complete rerun.

The five primary models use a single recursive LSTM training protocol.  This
module uses training data only, a common seed rule, and one fixed recursive
forecasting procedure. Residual forecasts are added directly to SARIMA forecasts.
"""

import os

# Must be set before CUDA is initialized, including when imported by a worker.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PyEMD.CEEMDAN import CEEMDAN
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODELS = ["SARIMA", "LSTM", "SARIMA-LSTM", "CEEMDAN-LSTM", "SARIMA-CEEMDAN-LSTM"]

DEFAULT_CONFIG = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "ceemdan_trials": 30,
    "ceemdan_epsilon": 0.005,
    "ceemdan_noise_scale": 1.0,
}


def set_seed(seed: int) -> None:
    """Reset initialization, dropout, and deterministic execution per fit."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _finite(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _series(values: Any, name: str = "training series") -> np.ndarray:
    array = _finite(values, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have one channel")
    return array


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Compute unrounded metrics; a constant target has undefined R squared."""
    actual = _series(y_true, "metric target")
    predicted = _series(y_pred, "metric prediction")
    if actual.shape != predicted.shape:
        raise ValueError(f"metric length mismatch: {actual.shape} versus {predicted.shape}")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        error = predicted - actual
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(np.square(error)))
        ss_res = float(np.dot(error, error))
        centered = actual - float(np.mean(actual))
        ss_tot = float(np.dot(centered, centered))
        defined = ss_tot > 0.0
        r2 = float(1.0 - ss_res / ss_tot) if defined else None
    if not np.isfinite(mae) or not np.isfinite(mse) or (defined and not np.isfinite(r2)):
        raise FloatingPointError("metrics are not finite")
    return {"MAE": mae, "RMSE": math.sqrt(mse), "R2": r2, "r2_defined": bool(defined)}


def fit_sarima(
    train: Any, steps: int, seasonal_period: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a converged stable SARIMA using only the provided training data.

    Identical deterministic results can be shared between all three SARIMA
    families and all repeated neural-network seeds at the same training stage.
    Rejected candidates are retained in the JSON-compatible diagnostics.
    """
    series = _series(train)
    if int(steps) < 1 or len(series) < 3:
        raise ValueError("SARIMA requires a positive forecast horizon and at least three points")
    period = int(seasonal_period)
    location = float(series.mean())
    observed_std = float(series.std())
    scale = observed_std if observed_std > 0.0 else 1.0
    normalized = _finite((series - location) / scale, "standardized SARIMA training series")
    orders = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
    seasonal_orders = [(0, 0, 0, 0)]
    if period > 1 and len(series) >= 2 * period:
        seasonal_orders.extend([(1, 0, 0, period), (0, 0, 1, period), (1, 0, 1, period)])
    diagnostics = []
    best = None
    best_score = math.inf
    for order in orders:
        for seasonal_order in seasonal_orders:
            diagnostic: dict[str, Any] = {
                "order": list(order), "seasonal_order": list(seasonal_order), "accepted": False
            }
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    result = SARIMAX(
                        normalized,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="n",
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).fit(disp=False, maxiter=250)
                    converged = bool(result.mle_retvals.get("converged", False))
                    aic = float(result.aic)
                    bic = float(result.bic)
                    diagnostic["converged"] = converged
                    diagnostic["aic"] = aic if np.isfinite(aic) else None
                    diagnostic["bic"] = bic if np.isfinite(bic) else None
                    diagnostic["optimizer_iterations"] = int(result.mle_retvals.get("iterations", 0))
                    if not converged:
                        raise RuntimeError("optimizer did not converge")
                    if not np.isfinite(aic + bic):
                        raise FloatingPointError("AIC plus BIC is not finite")
                    _finite(result.params, "SARIMA parameters")
                    fitted = _series(result.fittedvalues, "SARIMA in-sample fitted values") * scale + location
                    prediction = _series(
                        result.get_forecast(steps=int(steps)).predicted_mean, "SARIMA forecast"
                    ) * scale + location
                    _finite(fitted, "unscaled SARIMA fitted values")
                    _finite(prediction, "unscaled SARIMA forecast")
                    if len(fitted) != len(series) or len(prediction) != int(steps):
                        raise ValueError("SARIMA produced an incorrect output length")
                    diagnostic["accepted"] = True
                    diagnostic["score"] = aic + bic
                    if aic + bic < best_score:
                        best_score = aic + bic
                        best = (fitted, prediction, list(order), list(seasonal_order))
            except Exception as exc:
                diagnostic["rejection"] = f"{type(exc).__name__}: {exc}"
            diagnostic["warnings"] = list(dict.fromkeys(str(item.message) for item in captured))
            diagnostics.append(diagnostic)
    if best is None:
        raise RuntimeError(f"All SARIMA candidates failed; diagnostics={diagnostics!r}")
    fitted, prediction, order, seasonal_order = best
    metadata = {
        "order": order,
        "seasonal_order": seasonal_order,
        "selection": "minimum training AIC+BIC among finite converged candidates",
        "selected_score": float(best_score),
        "train_mean": location,
        "train_std": observed_std,
        "standardization_scale": scale,
        "train_length": int(len(series)),
        "forecast_steps": int(steps),
        "seasonal_period": period,
        "enforce_stationarity": True,
        "enforce_invertibility": True,
        "maxiter": 250,
        "candidate_diagnostics": diagnostics,
    }
    return fitted, prediction, metadata


class RecursiveLSTM(nn.Module):
    def __init__(self, channels: int, config: dict[str, Any]) -> None:
        super().__init__()
        layers = int(config["num_layers"])
        dropout = float(config["dropout"])
        hidden = int(config["hidden_size"])
        self.lstm = nn.LSTM(
            input_size=channels, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output(self.dropout(hidden[-1]))


def _recursive_lstm(
    train: np.ndarray, steps: int, config: dict[str, Any], seed: int, device: torch.device
) -> tuple[np.ndarray, dict[str, Any]]:
    values = _finite(train, "LSTM training channels")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("LSTM training input must be a time by channel matrix")
    lookback = int(config["lookback"])
    epochs = int(config["epochs"])
    if lookback < 1 or len(values) <= lookback or epochs < 1:
        raise ValueError("not enough training points or invalid LSTM lookback/epochs")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    span = maximum - minimum
    normalization_scale = np.where(span > 0.0, span, 1.0)
    normalized = _finite((values - minimum) / normalization_scale, "normalized LSTM channels")
    x_numpy = np.stack([normalized[i:i + lookback] for i in range(len(values) - lookback)])
    y_numpy = normalized[lookback:]
    # Preload all tensors, including on CUDA, so batches do not cross the bus.
    x = torch.as_tensor(x_numpy, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_numpy, dtype=torch.float32, device=device)
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise FloatingPointError("LSTM training values overflow float32")
    set_seed(seed)  # Crucially before nn.Module constructs its initial weights.
    model = RecursiveLSTM(int(values.shape[1]), config).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=min(int(config["batch_size"]), len(x)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    milestones = [max(1, int(epochs * 0.5)), max(1, int(epochs * 0.75))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    criterion = nn.MSELoss()
    loss_history = []
    model.train()
    for _ in range(epochs):
        epoch_loss = torch.zeros((), dtype=torch.float32, device=device)
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config["gradient_clip"]), error_if_nonfinite=True
            )
            optimizer.step()
            epoch_loss += loss.detach() * len(batch_x)
        epoch_mean = float((epoch_loss / len(x)).cpu().item())
        if not np.isfinite(epoch_mean):
            raise FloatingPointError("LSTM training loss is not finite")
        loss_history.append(epoch_mean)
        scheduler.step()
    # Keep generated values on the same device and feed only forecasts forward.
    history = torch.as_tensor(normalized[-lookback:], dtype=torch.float32, device=device)
    predicted_rows = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(steps)):
            next_row = model(history[None, :, :])[0]
            predicted_rows.append(next_row)
            history = torch.cat((history[1:], next_row[None, :]), dim=0)
    predictions_norm = torch.stack(predicted_rows).cpu().numpy().astype(np.float64)
    predictions = _finite(
        predictions_norm * normalization_scale + minimum, "recursive LSTM forecast"
    )
    metadata = {
        "stage_seed": int(seed), "initialization_seed": int(seed), "loader_seed": int(seed),
        "forecast_mode": "recursive_one_step_fixed_origin",
        "channels": int(values.shape[1]), "training_windows": int(len(x)),
        "lookback": lookback, "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]), "dropout": float(config["dropout"]),
        "epochs": epochs, "batch_size": min(int(config["batch_size"]), len(x)),
        "optimizer": "Adam", "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "gradient_clip": float(config["gradient_clip"]),
        "scheduler": {"type": "MultiStepLR", "milestones": milestones, "gamma": 0.2},
        "normalization": "training-only channel-wise min-max",
        "normalization_min": minimum.tolist(), "normalization_max": maximum.tolist(),
        "normalization_scale": normalization_scale.tolist(),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epoch_mean_training_losses": loss_history,
        "last_epoch_mean_training_loss": loss_history[-1],
        "device": str(device), "deterministic_algorithms": True,
    }
    return predictions, metadata


def _ceemdan(
    train: np.ndarray, config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    values = _series(train, "CEEMDAN training series")
    trials = int(config["ceemdan_trials"])
    epsilon = float(config["ceemdan_epsilon"])
    noise_scale = float(config["ceemdan_noise_scale"])
    constant = bool(np.std(values) == 0.0)
    if constant:
        # CEEMDAN divides by signal std. Its degenerate constant input consists
        # only of the complete residue; no stochastic IMFs can be extracted.
        components = values[:, None].copy()
    else:
        decomposer = CEEMDAN(
            trials=trials, epsilon=epsilon, noise_scale=noise_scale, parallel=False
        )
        decomposer.noise_seed(int(seed))
        # PyEMD's returned last channel is already the final residue. Keep it.
        components = _finite(decomposer.ceemdan(values).T, "CEEMDAN components")
    if components.ndim != 2 or components.shape[0] != len(values):
        raise ValueError("CEEMDAN returned an incorrect time dimension")
    reconstruction = components.sum(axis=1)
    reconstruction_error = float(np.max(np.abs(reconstruction - values)))
    if not np.allclose(reconstruction, values, rtol=1e-8, atol=1e-8):
        raise FloatingPointError(f"CEEMDAN reconstruction failed: max error {reconstruction_error}")
    metadata = {
        "seed": int(seed), "trials": trials, "epsilon": epsilon,
        "noise_scale": noise_scale, "parallel": False,
        "channels_including_final_residue": int(components.shape[1]),
        "final_residue_included": True,
        "constant_input_residue_only": constant,
        "maximum_reconstruction_error": reconstruction_error,
    }
    return components, metadata


def forecast_model(
    model_name: str,
    train: Any,
    steps: int,
    configdict: dict[str, Any],
    seed: int,
    device: Any,
    sarima_tuple: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Forecast a pure model, without access to external or held-out labels."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    series = _series(train)
    if int(steps) < 1:
        raise ValueError("forecast horizon must be positive")
    config = {**DEFAULT_CONFIG, **configdict}
    if "lookback" not in config:
        period = int(config.get("seasonal_period", 7))
        config["lookback"] = 48 if period == 24 else 7
    device = torch.device(device)
    metadata: dict[str, Any] = {
        "model": model_name, "primary_pipeline": "pure_fixed_recursive",
        "stage_seed": int(seed), "train_length": int(len(series)),
        "forecast_steps": int(steps),
    }
    if model_name.startswith("SARIMA"):
        if sarima_tuple is None:
            sarima_tuple = fit_sarima(series, int(steps), int(config.get("seasonal_period", 7)))
        fitted, sarima_prediction, sarima_metadata = sarima_tuple
        fitted = _series(fitted, "cached SARIMA fitted values")
        sarima_prediction = _series(sarima_prediction, "cached SARIMA forecast")
        if len(fitted) != len(series) or len(sarima_prediction) != int(steps):
            raise ValueError("cached SARIMA tuple has inconsistent dimensions")
        metadata["sarima"] = sarima_metadata
        if model_name == "SARIMA":
            metadata["stochastic_seed_used"] = False
            return sarima_prediction.copy(), metadata
        neural_train = _finite(series - fitted, "SARIMA training residuals")
        baseline = sarima_prediction
        metadata["combination"] = "SARIMA forecast + sum of neural residual forecasts"
    else:
        neural_train = series
        baseline = np.zeros(int(steps), dtype=np.float64)
        metadata["combination"] = "sum of neural channel forecasts"
    if "CEEMDAN" in model_name:
        neural_train, decomposition_metadata = _ceemdan(neural_train, config, int(seed))
        metadata["ceemdan"] = decomposition_metadata
    neural_prediction, neural_metadata = _recursive_lstm(neural_train, int(steps), config, int(seed), device)
    metadata["lstm"] = neural_metadata
    metadata["stochastic_seed_used"] = True
    prediction = _finite(baseline + neural_prediction.sum(axis=1), f"{model_name} forecast")
    return prediction, metadata
