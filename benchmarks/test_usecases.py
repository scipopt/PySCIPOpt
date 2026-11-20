import random

import pytest

from pyscipopt import SCIP_RESULT, Model, Pricer, quicksum


def test_many_small_models(bench):
    rng = random.Random(0)
    instances = [
        [[rng.randint(1, 20) for _ in range(100)] for _ in range(50)] for _ in range(100)
    ]

    def run():
        for rows in instances:
            m = Model()
            x = [m.addVar(vtype="B", obj=-i) for i in range(100)]
            for w in rows:
                m.addCons(quicksum(wi * xi for wi, xi in zip(w, x)) <= 500)

    bench(run, rounds=3)


def test_modify_model(bench):
    rng = random.Random(0)
    n = 50_000
    m = Model()
    x = [m.addVar(ub=10) for _ in range(n)]
    conss = [m.addCons(quicksum(rng.sample(x, 10)) <= 50) for _ in range(5000)]
    objectives = [[rng.random() for _ in range(n)] for _ in range(5)]

    def run():
        for i, c in enumerate(objectives):
            m.setObjective(quicksum(ci * v for ci, v in zip(c, x)), "maximize")
            for v in x:
                m.chgVarUb(v, 5 + i)
                m.chgVarLb(v, i % 2)
            for cons in conss:
                m.chgRhs(cons, 50 + i)
                m.chgLhs(cons, -i)
                m.chgCoefLinear(cons, x[i], 2.0 + i)
            extra = [m.addCons(x[j] + x[j + 1] <= 15) for j in range(0, n - 1, 10)]
            for cons in extra:
                m.delCons(cons)

    bench(run, rounds=3)


def test_resolve_cycle(bench):
    rng = random.Random(0)
    n = 20_000
    m = Model()
    m.hideOutput()
    m.setPresolve(0)
    x = [m.addVar(ub=10) for _ in range(n)]
    budget = m.addCons(quicksum(x) <= 100)
    objectives = [[rng.random() for _ in range(n)] for _ in range(2)]

    def run():
        for i, c in enumerate(objectives):
            m.freeTransform()
            m.setObjective(quicksum(ci * v for ci, v in zip(c, x)), "maximize")
            m.chgRhs(budget, 100 + i)
            for v in x:
                m.chgVarUb(v, 5 + i % 5)
            m.optimize()
            [m.getVal(v) for v in x]

    bench(run, rounds=3)


class PoolPricer(Pricer):
    def __init__(self, conss, pool, batch):
        self.conss, self.pool, self.batch = conss, pool, batch

    def pricerinit(self):
        self.conss = [self.model.getTransformedCons(c) for c in self.conss]
        self.next = 0

    def pricerredcost(self):
        [self.model.getDualsolLinear(c) for c in self.conss]
        for rows in self.pool[self.next:self.next + self.batch]:
            var = self.model.addVar(obj=1000.0, pricedVar=True)
            for r in rows:
                self.model.addConsCoeff(self.conss[r], var, 1.0)
        self.next += self.batch
        return {"result": SCIP_RESULT.SUCCESS}


def _column_generation(rows=500, columns=200_000, nnz=20, batch=1000):
    rng = random.Random(0)
    m = Model()
    m.hideOutput()
    m.setPresolve(0)
    x = [m.addVar(obj=10.0) for _ in range(rows)]
    conss = [m.addCons(xi >= 1, separate=False, modifiable=True) for xi in x]
    pool = [rng.sample(range(rows), nnz) for _ in range(columns)]
    m.includePricer(PoolPricer(conss, pool, batch), "pool", "")
    return m


@pytest.mark.timing
def test_branch_and_price_columns(bench):
    bench(lambda m: m.optimize(), setup=lambda: ((_column_generation(),), {}), rounds=3)
