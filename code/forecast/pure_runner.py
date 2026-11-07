"""Re-run the pure forecasting protocol for selected stations.

Uses the fixed split in ``config.json``: the series is cut once, the first part is
used for training, the last part is held out for evaluation. Every model sees only
the training part. One row per station, seed and model is written to CSV.

Run: python pure_runner.py --datasets daily --sites SC1 --models SARIMA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unified_models import MODELS, compute_metrics, fit_sarima, forecast_model, set_seed

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["prediction_level", "site", "model_name", "seed",
           "parameter_settings", "MAE", "RMSE", "R2", "status"]


def station_key(path: Path) -> int:
    return int(path.stem[2:])


def settings_of(metadata: dict) -> dict:
    """Keep the settings that identify what was fitted; drop the loss history."""
    kept = {"protocol": "pure", "train_n": metadata["train_length"], "test_n": metadata["forecast_steps"]}
    if "sarima" in metadata:
        sarima = metadata["sarima"]
        kept["SARIMA"] = {"order": sarima["order"],
                          "seasonal_order": sarima["seasonal_order"],
                          "seasonal_period": sarima["seasonal_period"]}
    if "lstm" in metadata:
        lstm = metadata["lstm"]
        kept["LSTM"] = {key: lstm[key] for key in
                        ("lookback", "hidden_size", "num_layers", "epochs", "batch_size", "channels")}
    if "ceemdan" in metadata:
        ceemdan = metadata["ceemdan"]
        kept["CEEMDAN"] = {"trials": ceemdan["trials"],
                           "channels": ceemdan["channels_including_final_residue"]}
    return kept


def run_station(dataset: str, site: str, config: dict, models: list[str], device: str) -> pd.DataFrame:
    values = pd.read_csv(ROOT / "data" / dataset / f"{site}.csv", header=None).iloc[:, 0].to_numpy(dtype=float)
    split = config["split"]
    train_n = split["offset"] + int(split["ratio"] * (len(values) - split["offset"]))
    train, test = values[:train_n], values[train_n:]
    model_config = {**config["neural"], **config["datasets"][dataset]}
    sarima = fit_sarima(train, len(test), model_config["seasonal_period"])

    rows = []
    for seed in config["seeds"]:
        for model in models:
            set_seed(int(seed))
            if model == "SARIMA":
                predicted = sarima[1]
                settings = {"protocol": "pure", "train_n": train_n, "test_n": len(test),
                            "SARIMA": {"order": sarima[2]["order"],
                                       "seasonal_order": sarima[2]["seasonal_order"],
                                       "seasonal_period": sarima[2]["seasonal_period"]}}
            else:
                stage_seed = int(seed + config["seed_derivation"]["outer_offset"])
                predicted, metadata = forecast_model(
                    model, train, len(test), model_config, stage_seed, device, sarima)
                settings = settings_of(metadata)
            metric = compute_metrics(test, predicted)
            rows.append({
                "prediction_level": dataset, "site": site, "model_name": model, "seed": seed,
                "parameter_settings": json.dumps(settings, separators=(",", ":"), ensure_ascii=False),
                "MAE": metric["MAE"], "RMSE": metric["RMSE"], "R2": metric["R2"], "status": "ok",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["daily", "hourly"], choices=["daily", "hourly"])
    parser.add_argument("--sites", nargs="+", help="station ids such as SC1 SC17; default is all")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "forecast" / "rerun")
    args = parser.parse_args()

    config = json.loads((Path(__file__).with_name("config.json")).read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    frames = []
    for dataset in args.datasets:
        sites = args.sites or [path.stem for path in
                               sorted((ROOT / "data" / dataset).glob("SC*.csv"), key=station_key)]
        for site in sites:
            frame = run_station(dataset, site, config, args.models, args.device)
            frames.append(frame)
            print(f"{dataset}/{site}: MAE {frame.MAE.mean():.4f} over {len(frame)} rows", flush=True)

    result = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "pure_metrics.csv"
    result.to_csv(target, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(result), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
