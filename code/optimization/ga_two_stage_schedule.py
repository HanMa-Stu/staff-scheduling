"""Two-stage staffing model.

Stage 1 searches daily regular-worker counts x[j,k] with a genetic algorithm and
fills temporary counts y[j,k] by a bounded repair. Its constraints are checked
directly: a candidate that violates Eq. (25), the daily upper limit N, or the
temporary-staff gate is rejected rather than penalised.

Stage 2 takes P, X and Y from Stage 1, builds a weekly regular-worker budget,
assigns individual workers to site-days, and may relabel shifts. Its constraints
are handled in the score: the hard part (Eq. (25), daily quota, attendance cap)
is repaired where possible and otherwise returned as a violation, so a run always
ends with a usable answer plus an explicit feasibility status.

Six shifts: 00-08, 05-13, 08-16, 12-20, 14-22, 16-24.
Run: python ga_two_stage_schedule.py --input demo_input.json --out demo_out
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except ImportError:  # reported by solve_stage1 and main()
    Bounds = LinearConstraint = milp = coo_matrix = None


CHECK_HOURS = (8, 13, 16, 20, 22, 24)
# Rows are the coefficients of shifts 1..6 at the six check hours.
COVERAGE_COEFFICIENTS = np.asarray(
    [
        [8, 3, 0, 0, 0, 0],
        [8, 8, 5, 1, 0, 0],
        [8, 8, 8, 4, 2, 0],
        [8, 8, 8, 8, 6, 4],
        [8, 8, 8, 8, 8, 6],
        [8, 8, 8, 8, 8, 8],
    ],
    dtype=float,
)


@dataclass
class Stage1Result:
    status: str
    message: str
    objective: float | None
    x: np.ndarray | None
    y: np.ndarray | None
    p: np.ndarray | None
    metadata: dict[str, Any] | None = None


@dataclass
class Stage1GAConfig:
    """Controls the shift-count GA used before individual assignment."""

    population_size: int = 80
    generations: int = 500
    elite_count: int = 6
    mutation_transfers: int = 5
    balance_weight: float = 5000.0
    violation_penalty: float = 1.0e7
    patience: int = 100
    seed: int = 42


@dataclass
class GAConfig:
    population_size: int = 120
    generations: int = 2000
    elite_count: int = 4
    mutation_swaps: int = 2
    violation_penalty: float = 1000.0
    patience: int = 250
    shift_balance_weight: float = 5.0
    reference_shift_weight: float = 0.5
    temporary_weight: float = 0.001
    seed: int = 42


def _site_nij(site: dict[str, Any], days: int, n_regular: int) -> np.ndarray:
    """Return the day-specific threshold capacity Nij[i,j]."""

    raw = site.get("Nij", n_regular)
    if np.isscalar(raw):
        return np.full(days, float(raw), dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (days,):
        raise ValueError(f"Nij for {site.get('name')} must be scalar or length {days}")
    return arr


def _demand(site: dict[str, Any], days: int, hours: int) -> np.ndarray:
    arr = np.asarray(site["demand"], dtype=float)
    if arr.shape != (days, hours):
        raise ValueError(
            f"demand for {site.get('name')} must have shape {(days, hours)}, got {arr.shape}"
        )
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"demand for {site.get('name')} contains invalid values")
    return arr


def solve_stage1(
    site: dict[str, Any],
    *,
    days: int,
    hours: int,
    m: float,
    n: float,
    temporary_big_m: float,
    time_limit: float | None = None,
) -> Stage1Result:
    """Solve the first-stage minimum-staffing MILP for one site."""

    if milp is None:
        raise RuntimeError("Stage 1 requires scipy; install scipy>=1.11")
    if hours != 24 or days < 1:
        raise ValueError("the corrected Eq. (25) implementation expects 24 hours per day")
    name = str(site.get("name", "site"))
    n_regular = int(site["N"])
    # Stage 1 meaning of O(i): site-day regular utilization threshold for
    # enabling temporary staff. It is not an employee-specific attendance cap.
    oi = float(site.get("O_threshold", site["O"]))
    if n_regular <= 0 or not (0 < oi <= 1):
        raise ValueError(f"invalid N or O for {name}")
    nij = _site_nij(site, days, n_regular)
    demand = _demand(site, days, hours)
    shifts = 6
    ns = days * shifts
    nvar = 2 * ns + days

    def ix(j: int, k: int) -> int:
        return j * shifts + k

    def iy(j: int, k: int) -> int:
        return ns + j * shifts + k

    def iz(j: int) -> int:
        return 2 * ns + j

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coeff: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        rows.append(coeff)
        lower.append(lo)
        upper.append(hi)

    # Eq. (25): the first five checkpoints are upper bounds and the 24-hour
    # row is a lower bound. At 13 h, shift 4 has coefficient 1.
    for j in range(days):
        for row, hour in zip(COVERAGE_COEFFICIENTS, CHECK_HOURS):
            coeff: dict[int, float] = {}
            for k, weight in enumerate(row):
                if weight:
                    coeff[ix(j, k)] = float(weight * m)
                    coeff[iy(j, k)] = float(weight * n)
            if hour == 24:
                add(coeff, lo=float(demand[j, hour - 1]))
            else:
                add(coeff, hi=float(demand[j, hour - 1]))

        # Eq. (22): regular staff do not exceed N(i) on any day.
        add({ix(j, k): 1.0 for k in range(shifts)}, hi=float(n_regular))

        # Eqs. (23)-(24): the site-wide regular attendance/utilization on day j
        # must reach O_threshold*Nij before temporary staff may be used.
        threshold = oi * float(nij[j])
        add({**{ix(j, k): 1.0 for k in range(shifts)}, iz(j): -threshold}, lo=0.0)
        add(
            {**{iy(j, k): 1.0 for k in range(shifts)}, iz(j): -temporary_big_m},
            hi=0.0,
        )

    matrix = coo_matrix(
        (
            [value for row in rows for value in row.values()],
            (
                [r for r, row in enumerate(rows) for _ in row],
                [c for row in rows for c in row],
            ),
        ),
        shape=(len(rows), nvar),
    ).tocsr()
    objective = np.zeros(nvar, dtype=float)
    objective[:ns] = 1.0
    objective[ns : 2 * ns] = 1.0
    integrality = np.ones(nvar, dtype=int)
    lb = np.zeros(nvar, dtype=float)
    ub = np.full(nvar, float(temporary_big_m), dtype=float)
    ub[2 * ns :] = 1.0
    options = {} if time_limit is None else {"time_limit": float(time_limit)}
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    if not result.success or result.x is None:
        return Stage1Result("infeasible_or_failed", str(result.message), None, None, None, None)
    x = np.rint(result.x[:ns]).astype(int).reshape(days, shifts)
    y = np.rint(result.x[ns : 2 * ns]).astype(int).reshape(days, shifts)
    p = (x + y).sum(axis=1)
    return Stage1Result(
        "optimal" if result.status == 0 else "feasible",
        str(result.message),
        float(result.fun),
        x,
        y,
        p,
    )


def _stage1_produced(x: np.ndarray, y: np.ndarray, m: float, n: float) -> np.ndarray:
    """Return the six Eq. (25) left-hand sides for one site-day."""

    return COVERAGE_COEFFICIENTS @ (float(m) * x + float(n) * y)


def _stage1_feasibility(
    x: np.ndarray,
    y: np.ndarray,
    demand_checkpoints: np.ndarray,
    m: float,
    n: float,
    threshold: float,
    temporary_big_m: float,
) -> tuple[float, dict[str, float]]:
    """Compute a scalar Eq. (25)/gate violation for one site-day."""

    produced = _stage1_produced(x, y, m, n)
    upper = np.maximum(produced[:5] - demand_checkpoints[:5], 0.0).sum()
    lower = max(float(demand_checkpoints[5] - produced[5]), 0.0)
    # O*Nij is a gate for using temporary staff.  It is not a minimum number
    # of regular staff when y=0; a low-demand day may use fewer regular staff
    # and no temporary staff at all.
    regular_gate = max(float(threshold - x.sum()), 0.0) if y.sum() > 0 else 0.0
    temporary_bound = max(float(y.sum() - temporary_big_m), 0.0)
    violation = float(upper + lower + regular_gate + temporary_bound)
    return violation, {
        "upper_violation": float(upper),
        "lower_violation": float(lower),
        "regular_gate_violation": float(regular_gate),
        "temporary_bound_violation": float(temporary_bound),
    }


def _repair_stage1_temporary(
    x: np.ndarray,
    demand_checkpoints: np.ndarray,
    *,
    m: float,
    n: float,
    temporary_big_m: float,
    minimum_total: int = 0,
) -> np.ndarray | None:
    """Greedily add temporary workers without violating upper checkpoints.

    The regular chromosome is kept separate from this repair. This makes the
    first-stage GA responsible for the shift distribution while allowing the
    temporary workforce to remain a flexible reserve.
    """

    y = np.zeros(6, dtype=int)
    # BIG-M is a modelling bound, not an iteration count.  The final row of
    # Eq. (25) gives an explicit finite upper bound on the number of temporary
    # workers needed when n>0; cap it by BIG-M for the same model semantics.
    if n <= 0:
        return None if demand_checkpoints[5] > _stage1_produced(x, y, m, n)[5] else y
    max_needed = max(
        int(np.ceil(max(float(demand_checkpoints[5]), 0.0) / (8.0 * float(n)))) + 1,
        max(0, int(minimum_total) - int(x.sum())),
    )
    max_additions = min(int(max(0.0, temporary_big_m)), max_needed)
    for _ in range(max_additions + 1):
        produced = _stage1_produced(x, y, m, n)
        if np.any(produced[:5] > demand_checkpoints[:5] + 1.0e-9):
            return None
        if (produced[5] + 1.0e-9 >= demand_checkpoints[5]
                and x.sum() + y.sum() >= minimum_total):
            return y
        # The gate on regular staffing is checked by the caller after repair,
        # because it depends on whether any temporary worker was added.
        choices: list[tuple[float, float, int]] = []
        for k in range(6):
            candidate = y.copy()
            candidate[k] += 1
            candidate_produced = _stage1_produced(x, candidate, m, n)
            if np.all(candidate_produced[:5] <= demand_checkpoints[:5] + 1.0e-9):
                # Prefer the latest shift when the final-hour gain is equal;
                # it leaves more room below the early upper bounds.
                slack = float(np.maximum(demand_checkpoints[:5] - candidate_produced[:5], 0).sum())
                choices.append((float(candidate_produced[5] - produced[5]), slack, k))
        if not choices:
            return None
        choices.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        y[choices[0][2]] += 1
    return None


def solve_stage1_ga(
    site: dict[str, Any],
    *,
    days: int,
    hours: int,
    m: float,
    n: float,
    temporary_big_m: float,
    config: Stage1GAConfig | None = None,
) -> Stage1Result:
    """Search Stage 1 shift counts with a genetic algorithm.

    The chromosome is ``x[j,k]`` for regular workers.  Its daily total is
    bounded by ``N`` rather than fixed to ``N``; this preserves the first
    stage's minimum-staffing objective.  Temporary counts are regenerated for
    every chromosome and activate the O*Nij gate only when they are nonzero.
    Feasible candidates are ranked by total people plus a weighted regular-
    shift balance penalty. This is a trade-off, not a minimum-staffing proof.
    """

    if hours != 24 or days < 1:
        raise ValueError("the corrected Eq. (25) implementation expects 24 hours per day")
    cfg = config or Stage1GAConfig()
    rng = np.random.default_rng(cfg.seed)
    name = str(site.get("name", "site"))
    n_regular = int(site["N"])
    oi = float(site.get("O_threshold", site["O"]))
    if n_regular <= 0 or not (0 < oi <= 1):
        raise ValueError(f"invalid N or O for {name}")
    demand = _demand(site, days, hours)
    checkpoints = demand[:, np.asarray(CHECK_HOURS) - 1]
    nij = _site_nij(site, days, n_regular)
    thresholds = oi * nij

    def repair(x_row: np.ndarray, j: int) -> np.ndarray | None:
        if np.any(x_row < 0) or int(x_row.sum()) > n_regular:
            return None
        y_row = _repair_stage1_temporary(
            x_row, checkpoints[j], m=m, n=n, temporary_big_m=temporary_big_m
        )
        if y_row is None:
            return None
        # Temporary workers are allowed only after the site-wide threshold is
        # reached.  If no temporary worker is needed, no threshold is imposed.
        if y_row.sum() > 0 and x_row.sum() + 1.0e-9 < thresholds[j]:
            return None
        if _stage1_feasibility(
            x_row, y_row, checkpoints[j], m, n, thresholds[j], temporary_big_m
        )[0] > 1.0e-8:
            return None
        return y_row.astype(int)

    def minimum_seed(j: int) -> tuple[np.ndarray, np.ndarray] | None:
        # Search low staffing totals first, but construct several feasible
        # shift profiles at each total.  Starting every chromosome with all
        # workers in shift 6 makes early-shift transfers very unlikely under
        # the cumulative upper bounds.  A balanced feasible seed lets the GA
        # compare staffing cost and shift concentration from generation 0.
        shares = np.linspace(0.45, 1.0, 12)
        for total in range(n_regular + 1):
            if total < thresholds[j] and 8.0 * float(m) * total < checkpoints[j, 5]:
                continue
            candidates: list[tuple[tuple[float, float], np.ndarray, np.ndarray]] = []
            for share6 in shares:
                x_row = np.zeros(6, dtype=int)
                x_row[5] = int(round(total * float(share6)))
                remainder = total - int(x_row[5])
                x_row[:5] = remainder // 5
                x_row[: remainder % 5] += 1
                y_row = repair(x_row, j)
                if y_row is None:
                    continue
                balance = float(np.square(x_row - float(total) / 6.0).sum())
                people = float(x_row.sum() + y_row.sum())
                candidates.append(((people, balance), x_row, y_row))
            if candidates:
                _, x_best, y_best = min(candidates, key=lambda item: item[0])
                return x_best, y_best
        return None

    base_x = np.zeros((days, 6), dtype=int)
    base_y = np.zeros((days, 6), dtype=int)
    for j in range(days):
        seed = minimum_seed(j)
        if seed is None:
            return Stage1Result(
                "infeasible_or_failed",
                f"Stage-1 GA could not find a feasible staffing level on day {j + 1}",
                None,
                None,
                None,
                None,
                {"method": "genetic_algorithm", "seed": cfg.seed},
            )
        base_x[j], base_y[j] = seed

    def evaluate(x: np.ndarray) -> tuple[float, float, float, float, np.ndarray | None]:
        if np.any(x < 0) or np.any(x.sum(axis=1) > n_regular):
            return cfg.violation_penalty, 1.0, float(x.sum()), 1.0, None
        violations = 0.0
        total_people = 0.0
        balance = 0.0
        y_all = np.zeros((days, 6), dtype=int)
        for j in range(days):
            y_row = repair(x[j], j)
            if y_row is None:
                violations += 1.0
                continue
            y_all[j] = y_row
            total_people += float(x[j].sum() + y_row.sum())
            mean = float(x[j].sum()) / 6.0
            balance += float(np.square(x[j] - mean).sum()) / max(float(n_regular * n_regular), 1.0)
        score = cfg.violation_penalty * violations + total_people + cfg.balance_weight * balance
        return score, violations, total_people, balance, y_all if violations == 0 else None

    def valid(x: np.ndarray) -> bool:
        return evaluate(x)[1] <= 1.0e-8

    def fitness(x: np.ndarray) -> tuple[float, float, float, float, np.ndarray | None]:
        return evaluate(x)

    def mutate(parent: np.ndarray) -> np.ndarray:
        child = parent.copy()
        for _ in range(max(1, int(cfg.mutation_transfers))):
            j = int(rng.integers(days))
            source = int(rng.integers(6))
            target = int(rng.integers(6))
            candidate = child[j].copy()
            operation = float(rng.random())
            if operation < 0.65 and source != target and candidate[source] > 0:
                candidate[source] -= 1
                candidate[target] += 1
            elif operation < 0.82 and candidate.sum() < n_regular:
                candidate[target] += 1
            elif candidate.sum() > 0 and candidate[source] > 0:
                candidate[source] -= 1
            if repair(candidate, j) is not None:
                child[j] = candidate
        return child

    def crossover(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        child = first.copy()
        for j in range(days):
            if float(rng.random()) < 0.5:
                child[j] = second[j]
        return child

    population: list[np.ndarray] = [base_x.copy()]
    for _ in range(max(1, cfg.population_size - 1)):
        population.append(mutate(base_x))
    best = min(population, key=lambda item: fitness(item)[0])
    best_score = fitness(best)[0]
    stagnant = 0
    generations_completed = 0
    for generation in range(1, cfg.generations + 1):
        ranked = sorted(population, key=lambda item: fitness(item)[0])
        current_score = fitness(ranked[0])[0]
        if current_score + 1.0e-12 < best_score:
            best = ranked[0].copy()
            best_score = current_score
            stagnant = 0
        else:
            stagnant += 1
        generations_completed = generation
        if stagnant >= cfg.patience:
            break
        elites = ranked[: max(1, min(cfg.elite_count, len(ranked)))]
        next_population = [item.copy() for item in elites]
        while len(next_population) < cfg.population_size:
            first = elites[int(rng.integers(len(elites)))]
            second = elites[int(rng.integers(len(elites)))]
            next_population.append(mutate(crossover(first, second)))
        population = next_population

    evaluated = fitness(best)
    if not valid(best):
        return Stage1Result(
            "infeasible_or_failed",
            "Stage-1 GA ended without a feasible chromosome",
            None,
            None,
            None,
            None,
            {"method": "genetic_algorithm", "seed": cfg.seed, "generations_completed": generations_completed},
        )
    x = best.astype(int)
    y = evaluated[4].astype(int)
    p = (x + y).sum(axis=1)
    score, violations, total_people, balance, _ = evaluated
    return Stage1Result(
        "feasible",
        "Stage-1 GA found a feasible staffing/balance trade-off",
        float(total_people),
        x,
        y,
        p,
        {
            "method": "genetic_algorithm",
            "seed": cfg.seed,
            "population_size": cfg.population_size,
            "generations": cfg.generations,
            "generations_completed": generations_completed,
            "fitness": float(score),
            "constraint_violation": float(violations),
            "regular_shift_balance": float(balance),
            "daily_regular_totals": x.sum(axis=1).astype(int).tolist(),
            "temporary_totals": y.sum(axis=1).astype(int).tolist(),
        },
    )


def regular_assignment_quotas(
    p_min: np.ndarray,
    *,
    n_regular: int,
    oi: float,
    days: int,
    regular_assignment_budget: int | None = None,
) -> np.ndarray:
    """Allocate the weekly regular-worker budget across site-days."""

    p_min = np.asarray(p_min, dtype=int)
    if p_min.shape != (days,):
        raise ValueError("Pij must have one value per day")
    if np.any(p_min < 0):
        raise ValueError("Pij cannot be negative")
    max_regular_assignments = int(n_regular * math.floor(float(oi) * days + 1e-9))
    budget = max_regular_assignments if regular_assignment_budget is None else int(regular_assignment_budget)
    # The aggregate budget cannot exceed the individual integer attendance cap.
    budget = min(max(0, budget), max_regular_assignments)
    weights = np.maximum(p_min.astype(float), 1.0)
    raw = budget * weights / float(weights.sum()) if budget else np.zeros(days)
    quotas = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - quotas))[: max(0, budget - int(quotas.sum()))]:
        quotas[int(j)] += 1
    if np.any(quotas > n_regular):
        raise ValueError("daily regular quota exceeds N; reduce regular_assignment_budget")
    return quotas




class FlexibleStage2GA:
    """Jointly adjust regular-worker shifts and individual assignments.

    ``x_reference`` is the balanced Stage-1 output.  The GA receives it
    explicitly, converts it into a daily regular-worker quota, and then
    searches employee-by-day shift labels.  A mutation may change a worker's
    shift label, so the resulting ``x`` counts are not frozen.  Temporary
    counts are repaired for every chromosome from the Eq. (25) capacity rows.
    """

    def __init__(
        self,
        x_reference: np.ndarray,
        y_reference: np.ndarray | None,
        daily_quotas: np.ndarray,
        demand: np.ndarray,
        n_regular: int,
        attendance_rate_limit: float,
        k_max: int,
        m: float,
        n: float,
        temporary_big_m: float,
        config: GAConfig,
        minimum_staffing: np.ndarray | None = None,
    ):
        ref = np.asarray(x_reference, dtype=int)
        quotas = np.asarray(daily_quotas, dtype=int)
        demand = np.asarray(demand, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 6:
            raise ValueError("x_reference must have shape (days, 6)")
        if quotas.shape != (ref.shape[0],) or demand.shape != (ref.shape[0], 24):
            raise ValueError("Stage-2 GA inputs have inconsistent dimensions")
        if (quotas < 0).any() or (quotas > n_regular).any():
            raise ValueError("daily regular quotas must be between zero and N")
        self.reference = ref
        self.temporary_reference = None if y_reference is None else np.asarray(y_reference, dtype=int)
        if self.temporary_reference is not None and self.temporary_reference.shape != ref.shape:
            raise ValueError("y_reference must have the same shape as x_reference")
        self.daily_quotas = quotas
        # Pij is an explicit total-staff lower bound, separate from the
        # regular-only quota and the soft X/Y distribution references.
        self.minimum_staffing = (np.zeros(ref.shape[0], dtype=int)
                                 if minimum_staffing is None
                                 else np.asarray(minimum_staffing, dtype=int))
        if self.minimum_staffing.shape != quotas.shape or (self.minimum_staffing < 0).any():
            raise ValueError("minimum_staffing must contain one nonnegative Pij per day")
        self.demand = demand
        self.checkpoints = demand[:, np.asarray(CHECK_HOURS) - 1]
        self.days = ref.shape[0]
        self.shifts = 6
        self.n_regular = int(n_regular)
        self.attendance_rate_limit = float(attendance_rate_limit)
        self.k_max = max(1, int(k_max))
        self.max_attendance = int(math.floor(self.attendance_rate_limit * self.days + 1e-9))
        capacity = self.n_regular * self.max_attendance
        if int(self.daily_quotas.sum()) > capacity:
            raise ValueError(
                f"weekly regular quotas {int(self.daily_quotas.sum())} exceed the "
                f"attendance capacity N*floor(O*days)={capacity}"
            )
        self.m = float(m)
        self.n = float(n)
        self.temporary_big_m = float(temporary_big_m)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.initial_counts = self._proportional_counts(self.reference, self.daily_quotas)

    def _proportional_counts(self, reference: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        counts = np.zeros_like(reference, dtype=int)
        for j in range(self.days):
            weights = np.maximum(reference[j].astype(float), 0.0)
            if weights.sum() <= 0:
                weights[:] = 1.0
            raw = quotas[j] * weights / weights.sum()
            counts[j] = np.floor(raw).astype(int)
            remainder = int(quotas[j] - counts[j].sum())
            for k in np.argsort(-(raw - counts[j]))[:remainder]:
                counts[j, int(k)] += 1
        return counts

    def _random_schedule(self, counts: np.ndarray) -> np.ndarray:
        h = np.full((self.n_regular, self.days), -1, dtype=np.int16)
        attendance = np.zeros(self.n_regular, dtype=int)
        streak = np.zeros(self.n_regular, dtype=int)
        for j in range(self.days):
            labels = np.repeat(np.arange(self.shifts, dtype=np.int16), counts[j])
            self.rng.shuffle(labels)
            free = set(range(self.n_regular))
            for label in labels:
                feasible = [
                    p for p in free
                    if attendance[p] < self.max_attendance and streak[p] < self.k_max
                ]
                candidates = feasible or list(free)
                if not candidates:
                    raise RuntimeError("daily regular staffing count exceeds N")
                p = min(
                    candidates,
                    key=lambda q: (attendance[q], streak[q], float(self.rng.random()), q),
                )
                h[p, j] = label
                free.remove(p)
                attendance[p] += 1
                streak[p] += 1
            for p in range(self.n_regular):
                if h[p, j] < 0:
                    streak[p] = 0
        return h

    def _counts(self, h: np.ndarray) -> np.ndarray:
        result = np.zeros((self.days, self.shifts), dtype=int)
        for j in range(self.days):
            labels = h[:, j]
            result[j] = np.bincount(labels[labels >= 0], minlength=self.shifts)
        return result

    def _repair_daily_total(self, h: np.ndarray, j: int) -> None:
        target = int(self.daily_quotas[j])
        assigned = np.flatnonzero(h[:, j] >= 0)
        if len(assigned) > target:
            remove = self.rng.choice(assigned, size=len(assigned) - target, replace=False)
            h[remove, j] = -1
        elif len(assigned) < target:
            free = np.flatnonzero(h[:, j] < 0)
            add = self.rng.choice(free, size=target - len(assigned), replace=False)
            probs = self.reference[j].astype(float) + 1.0
            probs /= probs.sum()
            h[add, j] = self.rng.choice(self.shifts, size=len(add), p=probs)

    def _capacity(self, counts: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Repair temporary counts for every day; return None if any day fails."""

        y = np.zeros((self.days, self.shifts), dtype=int)
        violation = 0.0
        for j in range(self.days):
            repaired = _repair_stage1_temporary(
                counts[j],
                self.checkpoints[j],
                m=self.m,
                n=self.n,
                temporary_big_m=self.temporary_big_m,
                minimum_total=int(self.minimum_staffing[j]),
            )
            if repaired is None:
                violation += 1.0
            else:
                y[j] = repaired
        return (None, violation) if violation > 0 else (y, 0.0)

    def _streak_violations(self, h: np.ndarray) -> int:
        violations = 0
        for p in range(self.n_regular):
            run = 0
            for j in range(self.days):
                run = run + 1 if h[p, j] >= 0 else 0
                violations += max(0, run - self.k_max)
        return int(violations)

    def _score(self, h: np.ndarray) -> tuple[float, dict[str, Any]]:
        counts = self._counts(h)
        attendance = np.count_nonzero(h >= 0, axis=1)
        rates = attendance / self.days
        daily_error = int(np.abs(attendance.sum() - self.daily_quotas.sum()))
        attendance_violation = int(np.maximum(attendance - self.max_attendance, 0).sum())
        streak_violation = self._streak_violations(h)
        y, capacity_violation = self._capacity(counts)
        shift_balance = float(
            np.mean([np.var(row / max(float(row.sum()), 1.0)) for row in counts])
        )
        ref_share = self.reference / np.maximum(self.reference.sum(axis=1, keepdims=True), 1)
        actual_share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        reference_deviation = float(np.mean(np.square(actual_share - ref_share)))
        temp_total = float(y.sum()) if y is not None else float(self.temporary_big_m)
        if y is not None and self.temporary_reference is not None:
            y_ref_share = self.temporary_reference / np.maximum(
                self.temporary_reference.sum(axis=1, keepdims=True), 1
            )
            y_actual_share = y / np.maximum(y.sum(axis=1, keepdims=True), 1)
            temporary_reference_deviation = float(np.mean(np.square(y_actual_share - y_ref_share)))
        else:
            temporary_reference_deviation = 0.0
        penalty = self.config.violation_penalty * (
            daily_error + attendance_violation + streak_violation + capacity_violation
        )
        score = (
            float(np.var(rates))
            + self.config.shift_balance_weight * shift_balance
            + self.config.reference_shift_weight * reference_deviation
            + 0.05 * temporary_reference_deviation
            + self.config.temporary_weight * temp_total
            + penalty
        )
        feasible = daily_error == 0 and attendance_violation == 0 and streak_violation == 0 and y is not None
        return score, {
            "variance": float(np.var(rates)),
            "shift_balance": shift_balance,
            "reference_shift_deviation": reference_deviation,
            "temporary_reference_shift_deviation": temporary_reference_deviation,
            "temporary_total": temp_total,
            "daily_quota_error": float(daily_error),
            "attendance_violation": float(attendance_violation),
            "streak_violation": float(streak_violation),
            "capacity_violation": float(capacity_violation),
            "feasible": float(feasible),
            "x_counts": counts,
            "y_counts": y,
        }

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_regular) < 0.5
        child = np.where(mask[:, None], a, b).astype(np.int16, copy=False)
        for j in range(self.days):
            self._repair_daily_total(child, j)
        return child

    def _mutate(self, h: np.ndarray) -> np.ndarray:
        child = h.copy()
        for _ in range(max(1, self.config.mutation_swaps)):
            j = int(self.rng.integers(self.days))
            assigned = np.flatnonzero(child[:, j] >= 0)
            if len(assigned) == 0:
                continue
            p = int(self.rng.choice(assigned))
            if float(self.rng.random()) < 0.7:
                # Change the shift label while keeping the same employee/day.
                child[p, j] = int(self.rng.integers(self.shifts))
            else:
                # Move one attendance record to an off-duty employee.
                free = np.flatnonzero(child[:, j] < 0)
                if len(free):
                    q = int(self.rng.choice(free))
                    child[q, j] = child[p, j]
                    child[p, j] = -1
            self._repair_daily_total(child, j)
        return child

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        population = [self._random_schedule(self.initial_counts) for _ in range(self.config.population_size)]
        best: np.ndarray | None = None
        best_score = float("inf")
        best_meta: dict[str, Any] = {}
        stale = 0
        stopped_reason = "generation_limit"
        generation = -1
        for generation in range(self.config.generations):
            scored = [self._score(ind) for ind in population]
            scores = np.asarray([item[0] for item in scored])
            order = np.argsort(scores)
            current_score, current_meta = scored[int(order[0])]
            if current_score + 1e-12 < best_score:
                best = population[int(order[0])].copy()
                best_score = current_score
                best_meta = current_meta.copy()
                stale = 0
            else:
                stale += 1
            if stale >= self.config.patience:
                stopped_reason = "no_improvement_patience"
                break
            elite_count = max(1, min(self.config.elite_count, len(population)))
            elites = [population[int(i)].copy() for i in order[:elite_count]]
            next_population = elites[:]
            while len(next_population) < self.config.population_size:
                first = population[int(self.rng.choice(order[: max(elite_count, len(order) // 2)]))]
                second = population[int(self.rng.choice(order[: max(elite_count, len(order) // 2)]))]
                next_population.append(self._mutate(self._crossover(first, second)))
            population = next_population
        if best is None:
            raise RuntimeError("Stage-2 GA produced no candidate")
        final_score, final_meta = self._score(best)
        if final_meta.get("y_counts") is None:
            raise RuntimeError("Stage-2 GA did not produce a capacity-feasible candidate")
        x_counts = np.asarray(final_meta.pop("x_counts"), dtype=int)
        y_counts = np.asarray(final_meta.pop("y_counts"), dtype=int)
        metadata = {
            "status": "feasible" if final_meta["feasible"] else "best_effort_infeasible",
            "stopped_reason": stopped_reason,
            "generations_completed": generation + 1,
            "score": final_score,
            "attendance_rate_limit": self.attendance_rate_limit,
            "max_attendance_per_employee": self.max_attendance,
            "effective_discrete_attendance_rate": self.max_attendance / max(self.days, 1),
            "seed": self.config.seed,
            **final_meta,
            "daily_regular_quotas": self.daily_quotas.tolist(),
            "stage1_Pij_lower_bound": self.minimum_staffing.tolist(),
            "stage1_reference_counts": self.reference.tolist(),
            "stage1_temporary_reference_counts": (
                None if self.temporary_reference is None else self.temporary_reference.tolist()
            ),
            "stage2_regular_counts": x_counts.tolist(),
            "stage2_temporary_counts": y_counts.tolist(),
        }
        return best, x_counts, y_counts, metadata


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_two_stage(config: dict[str, Any], output_dir: Path, seed: int | None = None) -> dict[str, Any]:
    days = int(config.get("days", 7))
    hours = int(config.get("hours", 24))
    m = float(config["m"])
    n = float(config["n"])
    big_m = float(config.get("temporary_big_m", 100000))
    stage1_rows: list[list[Any]] = []
    shift_rows: list[list[Any]] = []
    stage2_shift_rows: list[list[Any]] = []
    assignment_rows: list[list[Any]] = []
    summaries: list[dict[str, Any]] = []
    ga_config_data = dict(config.get("ga", {}))
    if seed is not None:
        ga_config_data["seed"] = seed
    ga_config = GAConfig(**ga_config_data)
    stage1_method = str(config.get("stage1_method", "ga")).lower()
    stage1_ga_data = dict(config.get("stage1_ga", {}))
    if seed is not None:
        stage1_ga_data["seed"] = seed
    stage1_ga_config = Stage1GAConfig(**stage1_ga_data)

    for site in config["sites"]:
        name = str(site.get("name", f"site_{len(summaries) + 1}"))
        if stage1_method == "milp":
            stage1 = solve_stage1(
                site,
                days=days,
                hours=hours,
                m=m,
                n=n,
                temporary_big_m=big_m,
                time_limit=config.get("stage1_time_limit"),
            )
        elif stage1_method == "ga":
            stage1 = solve_stage1_ga(
                site,
                days=days,
                hours=hours,
                m=m,
                n=n,
                temporary_big_m=big_m,
                config=stage1_ga_config,
            )
        else:
            raise ValueError("stage1_method must be 'ga' or 'milp'")
        summary: dict[str, Any] = {
            "site": name,
            "stage1_method": stage1_method,
            "stage1_status": stage1.status,
            "stage1_message": stage1.message,
            "stage1_objective": stage1.objective,
            "stage1_ga": stage1.metadata,
        }
        if stage1.x is None or stage1.y is None or stage1.p is None:
            summaries.append(summary)
            continue
        for j in range(days):
            stage1_rows.append(
                [name, j + 1, int(stage1.p[j]), int(stage1.x[j].sum()), int(stage1.y[j].sum())]
            )
            for k in range(6):
                shift_rows.append([name, j + 1, k + 1, int(stage1.x[j, k]), int(stage1.y[j, k])])
        quotas = regular_assignment_quotas(
            stage1.p,
            n_regular=int(site["N"]),
            oi=float(site.get("O_attendance", site["O"])),
            days=days,
            # Preserve the Stage-1 regular workload when it fits the
            # individual attendance cap; only the excess is replaced by
            # flexible temporary staff in Stage 2.
            regular_assignment_budget=int(stage1.x.sum()),
        )
        ga = FlexibleStage2GA(
            stage1.x,
            stage1.y,
            quotas,
            np.asarray(site["demand"], dtype=float),
            int(site["N"]),
            float(site.get("O_attendance", site["O"])),
            int(site["K"]),
            m,
            n,
            big_m,
            ga_config,
            minimum_staffing=stage1.p,
        )
        schedule, x_plan, y_plan, ga_meta = ga.solve()
        summary["stage2"] = ga_meta
        for j in range(days):
            for k in range(6):
                stage2_shift_rows.append(
                    [name, j + 1, k + 1, int(x_plan[j, k]), int(y_plan[j, k])]
                )
        for p in range(schedule.shape[0]):
            for j in range(days):
                label = int(schedule[p, j])
                assignment_rows.append(
                    [name, p + 1, j + 1, label + 1 if label >= 0 else "", int(label >= 0)]
                )
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "stage1_daily.csv",
        ["site", "day", "Pij", "regular_total", "temporary_total"],
        stage1_rows,
    )
    _write_csv(
        output_dir / "stage1_shift_counts.csv",
        ["site", "day", "shift", "x_regular", "y_temporary"],
        shift_rows,
    )
    _write_csv(
        output_dir / "stage2_assignments.csv",
        ["site", "employee", "day", "shift", "assigned"],
        assignment_rows,
    )
    _write_csv(
        output_dir / "stage2_shift_counts.csv",
        ["site", "day", "shift", "x_regular", "y_temporary"],
        stage2_shift_rows,
    )
    manifest = {
        "model": "two_stage_staff_schedule",
        "days": days,
        "hours": hours,
        "m": m,
        "n": n,
        "temporary_big_m": big_m,
        "stage1_method": stage1_method,
        "stage1_ga": asdict(stage1_ga_config),
        "ga": asdict(ga_config),
        "sites": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input JSON")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--seed", type=int, help="override GA seed")
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    if str(config.get("stage1_method", "ga")).lower() == "milp" and milp is None:
        raise SystemExit("scipy is required for stage1_method='milp'")
    manifest = run_two_stage(config, args.out, seed=args.seed)
    failed = [
        s["site"] for s in manifest["sites"]
        if s["stage1_status"] not in {"optimal", "feasible"}
    ]
    print(json.dumps(
        {"sites": len(manifest["sites"]), "stage1_failed": failed}, ensure_ascii=False
    ))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
