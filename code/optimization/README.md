# SC1 staff scheduling with two-stage genetic algorithms

Reproducible seven-day staff scheduling for sorting center SC1 using saved hourly forecasts. Stage 2 assigns individual employees to the **exact regular-staff counts selected in Stage 1**. Each regular employee works at most **six days** in the seven-day horizon.

This SC1 experiment fixes identical regular-staff counts across both stages. Each of the 200 employees works six days. The included data and code are sufficient to run and validate the schedule.

## Verified result

| Day | Regular staff | Temporary staff |
|---|---:|---:|
| 1 | 174 | 62 |
| 2 | 170 | 71 |
| 3 | 171 | 67 |
| 4 | 170 | 66 |
| 5 | 171 | 63 |
| 6 | 172 | 59 |
| 7 | 172 | 57 |
| Total worker-days | **1200** | **445** |

- All 42 regular-staff shift counts agree between the stages; temporary counts are retained as well.
- Each of the 200 regular employees works six days and rests one day. No employee works seven consecutive days within this horizon.
- Day 1 regular-staff counts are **23, 20, 21, 24, 19, 67** for shifts S1–S6.
- Relative to the previous Stage-1 result, only Day 4 changes. Its selected regular counts are **25, 20, 20, 21, 19, 65**, with 66 temporary staff in S6.

The six shifts are S1 00:00–08:00, S2 05:00–13:00, S3 08:00–16:00, S4 12:00–20:00, S5 14:00–22:00, and S6 16:00–24:00. Employee numbers are model identifiers 1–200; `OFF` denotes rest.

## Package paths and commands

- Code and parameters: `code/optimization/`.
- Saved input: `data/SC1_forecast.csv` (168 hourly forecasts).
- Selected results and individual rosters: `results/SC1/`.
- Reproduction output: `results/SC1_reproduced/`.

Run these commands from the package root:

```bash
python code/optimization/validate_results.py
python code/optimization/run_SC1.py
python code/optimization/validate_results.py --results results/SC1_reproduced --compare results/SC1
```

The validator does not train or optimize a model. The reproduction command runs the scheduling GA using saved forecasts, without retraining any forecasting model. The included CSV and JSON results are copied from the supplied SC1 scheduling delivery. The script changes in this package are limited to default input/output paths.

`shift_counts.csv` contains both stages side by side; `employee_schedule.csv` contains all 200 weekly rosters; `day1_staff_ids.csv` lists individual staff IDs for each Day-1 shift.

## Model settings and candidate selection

`parameters.json` is the single configuration file. `O_threshold=0.85` is the Stage-1 temporary-worker activation threshold: temporary staff may be used only after regular staffing reaches 170 that day. It is **not** a daily regular-staff maximum. The explicit integer `max_attendance_days=6` controls individual attendance in Stage 2; it is equivalent to a six-sevenths individual attendance-rate bound for this horizon. An unused legacy individual rate is omitted to prevent ambiguity.

Stage 1 retains its original constraints and weighted score: total worker-days plus 5000 times regular-shift imbalance. No weekly attendance-budget constraint is added to Stage 1. Its GA uses population 20, generation limit 100, patience 100, and seed 20042. Regular/temporary processing capacities are 25/20 items per hour.

The search retains 1402 distinct feasible GA candidates. Single-day crossovers between the search-best candidate and visited candidates expand this to 4713 candidates, without consulting the Stage-2 attendance cap. Candidates are ranked using the original Stage-1 score. Stage 2 rejects inputs it cannot assign and never clips, rescales, or reallocates the fixed shift counts. The accepted candidate is ranked 1203 and replaces Day 4 with a day from another GA chromosome. The unselected Stage-1 search-best result requires 1204 regular worker-days and is not the reported schedule.

Stage 2 uses seed 42, population 30, generation limit 100, and patience 30. Whole-day crossover and within-day employee swaps preserve Eq. (29); attendance repair also preserves each shift count. It stops after 31 generations without further score improvement. With 1200 assignments and a six-day cap, all 200 employees necessarily work six days, so zero attendance variance follows from these counts and is not an independent claim of GA superiority.

The result is a feasible heuristic schedule, not a proof of global optimality. Temporary staff remain in S6 under the existing capacity-repair rule. Individual cross-day minimum rest hours and employee skills are outside this experiment's constraints.

## Forecast input and verification

`data/SC1_forecast.csv` contains the 168 saved hourly forecasts from the SARIMA–CEEMDAN–LSTM SC1 run with seed 42. Scheduling uses `y_pred_for_schedule`; `y_pred_raw` is retained for traceability. The input hash is recorded in `summary.json`. The forecasting model itself is outside this scheduling-only repository.

The independent validator checks Eq. (25), the temporary-worker gate, integer staffing, daily regular-staff limits, exact Eq. (29) counts reconstructed from individual schedules, the six-day attendance cap, consecutive working days, staff-ID lists, summary totals, and the forecast hash. Runtime logs, rejected-candidate listings, historical result folders, and obsolete flexible-count implementations are excluded.

Related article: H. Ma, S. Wang, and X. Gu, “Research on Staff Scheduling in Sorting Centers Based on SARIMA-CEEMD-LSTM Model and Two-Stage Optimization Model,” IEEE Access, vol. 13, pp. 179419–179437, 2025, doi: 10.1109/ACCESS.2025.3621615. This repository records the corrected SC1 scheduling experiment described above.
