"""Re-run the SC1 two-stage schedule from the saved forecast.

Reads ``parameters.json`` and ``forecast_hourly.csv`` from --input, runs Stage 1
and Stage 2, and writes the shift counts, the individual assignments, the weekly
roster and ``summary.json`` to --out. The forecast model is never imported or
refitted, so the scheduling result depends only on the saved hourly forecast.

Run: python run_SC1_schedule.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ga_two_stage_schedule import run_two_stage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "SC1"
DEFAULT_OUT = ROOT / "results" / "SC1_rerun"


def sum_pij(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return int(sum(int(row["Pij"]) for row in csv.DictReader(handle)))


def sum_counts(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return int(sum(int(row["x_regular"]) + int(row["y_temporary"]) for row in csv.DictReader(handle)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="directory holding parameters.json and forecast_hourly.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    params = json.loads((args.input / "parameters.json").read_text(encoding="utf-8"))
    with (args.input / "forecast_hourly.csv").open(encoding="utf-8-sig", newline="") as handle:
        hourly = np.array([float(row["y_pred_for_schedule"]) for row in csv.DictReader(handle)])
    days, hours = int(params["days"]), int(params["hours"])
    if hourly.size != days * hours:
        raise ValueError(f"expected {days * hours} forecast rows, got {hourly.size}")
    if not np.isfinite(hourly).all() or (hourly < 0).any():
        raise ValueError("forecast contains negative or non-finite values")

    site = {**params["site"], "demand": np.cumsum(hourly.reshape(days, hours), axis=1).tolist()}
    config = {k: params[k] for k in ("days", "hours", "m", "n", "temporary_big_m", "stage1_ga", "ga")}
    config.update(stage1_method=params.get("stage1_method", "ga"), sites=[site])

    manifest = run_two_stage(config, args.out)
    result = manifest["sites"][0]
    if result["stage1_status"] not in {"optimal", "feasible"}:
        raise SystemExit(f"stage 1 failed: {result['stage1_status']} — {result['stage1_message']}")
    stage2 = result["stage2"]
    if stage2["status"] != "feasible":
        raise SystemExit(f"stage 2 did not reach a feasible roster: {stage2['status']}")

    with (args.out / "stage2_assignments.csv").open(encoding="utf-8-sig", newline="") as handle:
        lookup = {(int(r["employee"]), int(r["day"])): r for r in csv.DictReader(handle)}
    attendance = []
    with (args.out / "employee_weekly_schedule.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site", "employee", *[f"day_{day}" for day in range(1, days + 1)]])
        for employee in range(1, int(site["N"]) + 1):
            labels = []
            for day in range(1, days + 1):
                row = lookup[employee, day]
                on_duty = int(row["assigned"])
                labels.append(f"S{row['shift']}" if on_duty else "OFF")
            attendance.append(sum(label != "OFF" for label in labels))
            writer.writerow([site["name"], employee, *labels])

    stage1_total = sum_pij(args.out / "stage1_daily.csv")
    stage2_total = sum_counts(args.out / "stage2_shift_counts.csv")
    print(json.dumps({
        "stage1_total": stage1_total,
        "stage2_total": stage2_total,
        "stage2_score": round(float(stage2["score"]), 6),
        "attendance_days": [min(attendance), max(attendance)],
        "out": str(args.out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
