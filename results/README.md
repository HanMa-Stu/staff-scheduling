# Results

`SC1/` holds the staffing results and `forecast/` holds the forecasting results. The SC1 rerun requires `SC1/parameters.json` and `SC1/forecast_hourly.csv`. `SC1_rerun/` and `demo_out/` appear only after running and are not shipped with the package.

## SC1/

| File | Content |
|---|---|
| parameters.json | Site parameters m, n, N, O, K and the two GA configurations |
| forecast_hourly.csv | Saved 168-hour forecast. Scheduling uses the `y_pred_for_schedule` column. |
| stage1_daily.csv | Daily Pij with regular and temporary totals |
| stage1_shift_counts.csv | Stage-1 day × shift counts |
| stage2_shift_counts.csv | Stage-2 day × shift counts |
| stage2_assignments.csv | Shift label per employee and day, `assigned=0` for a rest day |
| employee_weekly_schedule.csv | The same roster in wide form, `OFF` for a rest day |
| summary.json | Objectives, daily quotas, score components, status |

## forecast/

| File | Content |
|---|---|
| pure.csv | MAE, RMSE and R² per station × seed × model |
| pure_summary.csv | Mean and standard deviation over the three seeds, by station and model |
| seed_model_means.csv | Mean over the 57 stations, by prediction level and seed |
| seed_model_comparisons.csv | Per-station wins, losses and differences against the four baselines |
| pure_main_predictions.csv.gz | Point forecasts and actual values of the main model |
| figures/ | Forecast comparison plots for the 57 stations |

## How results are produced

`code/optimization/run_SC1_schedule.py` reads the two saved inputs from `results/SC1/` and writes the reproduced schedule to `results/SC1_rerun/` by default.

`code/forecast/pure_runner.py` uses `data/` and `code/forecast/config.json` to rerun the forecast and writes `pure_metrics.csv` to `results/forecast/rerun/` by default. The [model comparison summary by seed](seed_model_comparison_summary.md) describes the saved forecast results.
