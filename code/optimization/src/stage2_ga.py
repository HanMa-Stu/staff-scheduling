"""Stage-2 GA with exact Stage-1 shift counts (Eq. 29).

Genes are employee-by-day shift labels, -1 means off duty. Crossover exchanges
whole days, and mutation swaps two employees within a day. These operators
preserve every shift count. Attendance repair also swaps within a day.
"""
import numpy as np


class FixedStage2GA:
    def __init__(self, x, y, employees, max_days, max_consecutive, config):
        self.x, self.y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
        self.employees, self.cap, self.k = int(employees), int(max_days), int(max_consecutive)
        self.days, self.shifts = self.x.shape
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if (self.x < 0).any() or (self.x.sum(axis=1) > employees).any():
            raise ValueError("Invalid fixed Stage-1 counts")
        # This is a Stage-2 feasibility check, never a Stage-1 constraint.
        if self.x.sum() > self.employees * self.cap:
            raise ValueError("Fixed Stage-1 counts exceed individual attendance capacity")

    def counts(self, h):
        return np.array([np.bincount(h[:,j][h[:,j] >= 0], minlength=self.shifts)
                         for j in range(self.days)])

    def longest_run(self, row):
        best = current = 0
        for label in row:
            current = current + 1 if label >= 0 else 0
            best = max(best, current)
        return best

    def valid(self, h):
        return (np.array_equal(self.counts(h), self.x)
                and ((h >= 0).sum(axis=1) <= self.cap).all()
                and all(self.longest_run(row) <= self.k for row in h))

    def initialize(self):
        # A circular allocation of rest slots spreads attendance evenly.
        # For this seven-day/six-day case every employee receives >=1 rest day.
        h = np.full((self.employees,self.days), -1, dtype=int)
        order = self.rng.permutation(self.employees)
        position = 0
        for j in range(self.days):
            off = self.employees-int(self.x[j].sum())
            rested = order[(position+np.arange(off)) % self.employees]
            position += off
            workers = np.setdiff1d(np.arange(self.employees),rested)
            labels = np.repeat(np.arange(self.shifts),self.x[j])
            self.rng.shuffle(labels)
            h[workers,j] = labels
        if not self.valid(h):
            raise ValueError("Constructive initialization failed the exact constraints")
        return h

    def repair_attendance(self, h, fallback):
        # Swap a working gene with an OFF gene, retaining its day and shift.
        attendance = (h >= 0).sum(axis=1)
        for p in self.rng.permutation(self.employees):
            while attendance[p] > self.cap or self.longest_run(h[p]) > self.k:
                moved = False
                for j in self.rng.permutation(self.days):
                    if h[p,j] < 0:
                        continue
                    receivers = np.flatnonzero((h[:,j] < 0) & (attendance < self.cap))
                    for q in self.rng.permutation(receivers):
                        h[q,j],h[p,j] = h[p,j],-1
                        if self.longest_run(h[q]) <= self.k:
                            attendance[p] -= 1
                            attendance[q] += 1
                            moved = True
                            break
                        h[p,j],h[q,j] = h[q,j],-1
                    if moved:
                        break
                if not moved:
                    return fallback.copy()
        return h if self.valid(h) else fallback.copy()

    def crossover(self, a, b):
        if self.rng.random() >= self.config.crossover_rate:
            return a.copy()
        mask = self.rng.random(self.days) < 0.5
        return self.repair_attendance(np.where(mask[None,:],a,b).copy(),a)

    def mutate(self, parent):
        child = parent.copy()
        if self.rng.random() < self.config.mutation_rate:
            for _ in range(self.config.mutation_swaps):
                j = int(self.rng.integers(self.days))
                p,q = self.rng.choice(self.employees,2,replace=False)
                child[p,j],child[q,j] = child[q,j],child[p,j]
            child = self.repair_attendance(child,parent)
        return child

    def score(self, h):
        # Retain the existing objective weights. Aggregate shift balance and
        # temporary use are constants here; reference deviations are zero.
        rates = (h >= 0).sum(axis=1)/self.days
        shares = self.x/np.maximum(self.x.sum(axis=1,keepdims=True),1)
        return (float(np.var(rates))
                + self.config.shift_balance_weight*float(np.mean(np.var(shares,axis=1)))
                + self.config.temporary_weight*float(self.y.sum()))

    def solve(self):
        cfg = self.config
        population = [self.initialize() for _ in range(cfg.population_size)]
        best, best_score, stale = population[0].copy(), float('inf'), 0
        stopped = 'generation_limit'
        for generation in range(cfg.generations):
            scores = np.array([self.score(h) for h in population])
            # Random tie-breaking preserves diversity without adding an objective.
            order = np.lexsort((self.rng.random(len(scores)),scores))
            current = int(order[0])
            if scores[current] < best_score-1e-12:
                best,best_score,stale = population[current].copy(),float(scores[current]),0
            else:
                stale += 1
                if abs(scores[current]-best_score)<1e-12:
                    best = population[current].copy()
            if stale >= cfg.patience:
                stopped='no_improvement_patience'
                break
            elite_count=max(1,min(cfg.elite_count,len(population)))
            next_population=[population[int(i)].copy() for i in order[:elite_count]]
            def select():
                contestants=self.rng.choice(len(population),size=cfg.tournament_size,replace=True)
                return population[int(contestants[np.argmin(scores[contestants])])]
            while len(next_population)<cfg.population_size:
                next_population.append(self.mutate(self.crossover(select(),select())))
            population=next_population
        assert self.valid(best)
        attendance=(best>=0).sum(axis=1)
        return best,dict(status='feasible',seed=cfg.seed,score=best_score,
                         generations_completed=generation+1,stopped_reason=stopped,
                         attendance_days_min=int(attendance.min()),
                         attendance_days_max=int(attendance.max()),
                         attendance_variance=float(np.var(attendance/self.days)),
                         max_consecutive_days=max(self.longest_run(row) for row in best),
                         equation29_exact=True,stage1_x_fixed=True,stage1_y_retained=True)
