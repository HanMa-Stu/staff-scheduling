# SC1 two-stage staffing reproducibility package

Companion code for [Research on Staff Scheduling in Sorting Centers Based on SARIMA-CEEMD-LSTM Model and Two-Stage Optimization Model](https://doi.org/10.1109/ACCESS.2025.3621615).

This package contains the two-stage staffing model and results for station SC1, and the evaluation results of five forecasting models on 57 stations. The forecasting rerun uses the series in `data/`. The SC1 scheduling rerun uses the saved forecast and parameters in `results/SC1/`. Both write to separate output directories by default.

## Layout

```
code/optimization/   staffing model, SC1 entry point, LINGO reference models, example input
code/forecast/       forecasting models, rerun entry point, config.json
data/daily/          daily series of 57 stations, one column without header
data/hourly/         hourly series of 57 stations, one column without header
results/SC1/         SC1 parameters, saved forecast, two-stage results, weekly roster
results/forecast/    57-station metrics, summaries, figures
```

## Install

```
pip install -r requirements.txt
```

Python 3.10 or newer. The examples run on CPU by default. If you use CUDA, install a compatible PyTorch build.

## Running

Reproduce the SC1 schedule. It uses the saved 168-hour forecast, takes about four minutes, and writes to `results/SC1_rerun/`.

```
cd code/optimization
python run_SC1_schedule.py
```

Run the example input. It covers seven days at one station, runs in a few seconds, and writes to `results/demo_out/`.

```
cd code/optimization
python ga_two_stage_schedule.py --input demo_input.json --out ../../results/demo_out
```

Rerun a forecast. The example uses one station and one model. The full run of 57 stations × 2 frequencies × 5 models × 3 seeds takes several hours.

```
cd code/forecast
python pure_runner.py --datasets daily --sites SC1 --models SARIMA
```

Omit `--sites` to run all stations, omit `--models` to run all five models.

## Model

The six shifts are 00-08, 05-13, 08-16, 12-20, 14-22 and 16-24. Hour h means (h-1):00 to h:00. The coefficient matrix of Eq. (25) is at the top of `ga_two_stage_schedule.py`.

Stage 1 finds daily shift counts of regular workers x and temporary workers y subject to Eq. (25), at most N regular workers per day, and Σx ≥ O·Nij whenever temporary workers are used. A chromosome that violates a constraint is rejected and never scored.

Stage 2 receives Pij and the Stage-1 x and y references, fixes a weekly regular budget of N·floor(O·7), then assigns named employees and may relabel shifts. This stage uses penalties to search a 200×7×6 assignment matrix, where rejecting every infeasible assignment would make the search impractical. The `status` field of `summary.json` reports `feasible` when all constraints are satisfied.

O=0.85 has different meanings in the two stages. In Stage 1 it is the site-day threshold for enabling temporary workers. In Stage 2 it is the attendance-rate cap per employee. Attendance is counted in whole days, so the implemented cap is floor(0.85×7)=5 days, an effective weekly cap of 71.4%.

## Results

`results/SC1/summary.json` records the two-stage objectives, daily quotas, shift counts, score components and status.

`stage1_daily.csv` holds daily Pij with regular and temporary totals. `stage1_shift_counts.csv` and `stage2_shift_counts.csv` hold day × shift counts. `stage2_assignments.csv` holds the shift label per employee and day, with `assigned=0` for a rest day. `employee_weekly_schedule.csv` is the same roster in wide form, where `OFF` marks a rest day.

Each row of `results/forecast/pure.csv` is one station × seed × model, with MAE, RMSE, R² and the fitted settings of that run. See the [model comparison summary by seed](results/seed_model_comparison_summary.md) for aggregate metrics and station-level win counts.

## Notes

The two `*.lng` files are LINGO reference models. They need A, Pij, Q, XREF and YREF supplied by the user and are not used by the commands above.

SARIMA is reproducible bit for bit under the same dependency versions. The neural models differ by about 1% between CPU and GPU because of floating-point backends.
