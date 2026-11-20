"""Original Stage-1 GA and shared configuration types.

Only the active GA implementation is included. Its constraints, fitness and
random operations are unchanged. The archive observes feasible candidates
without altering the evolutionary search.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

CHECK_HOURS = (8, 13, 16, 20, 22, 24)


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
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    mutation_swaps: int = 2
    violation_penalty: float = 1000.0
    range_regularizer: float = 0.05
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
        # The first temporary worker activates the Stage-1 site-wide gate.
        if y.sum() == 0:
            # The caller checks this gate through _stage1_feasibility; keeping
            # the check here avoids returning a candidate that cannot be used.
            # A value below the gate is still allowed when no temporary worker
            # is needed, handled by the return above.
            pass
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
    candidate_archive: dict | None = None,
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
        # Optional observation only: preserve feasible candidates for a separate
        # Stage-2 assignment trial. This does not alter Stage-1 fitness,
        # constraints, selection, RNG state, or population evolution.
        if candidate_archive is not None and violations == 0:
            candidate_archive.setdefault(x.tobytes(), {
                "x": x.copy(), "y": y_all.copy(), "fitness": float(score),
                "people": float(total_people), "balance": float(balance),
            })
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
            "stage1_objective_is_total_people": True,
        },
    )

