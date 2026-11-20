# Review package

This package replaces only the daily forecasting results with the user-supplied final delivery. Hourly numerical results are retained from the original Package. SC1 scheduling code and results are replaced by the supplied six-day-attendance delivery. No training was performed while preparing this package.

## Layout

- `code/forecast/daily/`: supplied daily five-model implementation, English documentation and configuration.
- `code/forecast/pure_runner.py`, `unified_models.py`, `config.json`: original hourly implementation; the runner now accepts hourly data only.
- `data/daily/`, `data/hourly/`: station data.
- `results/forecast/daily/`: supplied five-model metrics, aggregate and seed summaries, and one five-model CSV per station and seed.
- `results/forecast/hourly/`: original hourly metrics and main-model forecasts.
- `results/forecast/pure.csv`: combined correction metrics, explicitly labeled by task.
- `code/optimization/`, `results/SC1/`: updated six-day-attendance scheduling code and results.
- `supplementary/dual_threshold_gate/`: additional gating experiment, excluded from the correction results and analysis.

Daily evaluation uses 86 training days and 36 one-step forecasts with newly observed history. Hourly evaluation uses 505 training hours and a fixed-origin 215-hour test interval. These are different tasks. Hourly runs labeled 42/43/44 use the original effective neural/decomposition seeds 20042/20043/20044. Daily seeds are directly 42/43/44.

## Commands

Install the original pinned dependencies with `pip install -r requirements.txt`.

Daily training (CPU example):
```bash
TORCH_DEVICE=cpu python code/forecast/daily/run_five_models.py --workers 6
```
On PowerShell, set `$env:TORCH_DEVICE='cpu'` before the Python command. The supplied implementation also supports CUDA via its existing device selection. Exact cross-device reproduction is not established by this packaging review.

Hourly training:
```bash
python code/forecast/pure_runner.py --datasets hourly --device cpu
```
Use separate output directories when rerunning; the scripts' native outputs and postpackaging seed exports are described in METHODS.md. The JSON daily configuration documents constants in the source; it is not a runtime configuration loader.

## Result interpretation

Daily LSTM has the highest supplied mean R2 (approximately 0.375706). The main hybrid has mean R2 approximately 0.270187. All five models and stations are retained. The supplied summary includes paired Wilcoxon results across station-seed runs; these are preserved source statistics, not a new confirmation of significance or independence across seeds.

The daily source did not include per-day five-model predictions, so obsolete daily prediction files and mixed daily/hourly figures have been removed. Supplementary forecasts are not used to fill this gap. See `code/forecast/daily/METHODS.md` for the actual implemented formulas and corrections to inconsistent source descriptions.

## Updated SC1 schedule

See `code/optimization/README.md`. Both stages have identical regular-staff shift counts. All 200 regular employees work six days: 1200 regular worker-days and 445 temporary worker-days. Day-1 regular counts are 23, 20, 21, 24, 19, 67. Individual assignments are in `results/SC1/employee_schedule.csv` and `results/SC1/day1_staff_ids.csv`. Run `python code/optimization/validate_results.py` to check the saved result without rerunning optimization.
