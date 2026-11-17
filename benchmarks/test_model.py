import pytest

from pyscipopt import Model, quicksum


def _model_with_vars(n):
    m = Model()
    m.hideOutput()
    return m, [m.addVar(f"x{i}") for i in range(n)]


def test_add_vars(bench):
    def run():
        m = Model()
        for i in range(100_000):
            m.addVar(f"x{i}", vtype="I", lb=0, ub=10, obj=1.0)

    bench(run)


@pytest.mark.timing
def test_add_linear_conss(bench):
    n = 2_000_000

    def run(m, x):
        for i in range(n - 1):
            m.addCons(x[i] + 2 * x[i + 1] <= 10)

    bench(run, setup=lambda: (_model_with_vars(n), {}))


def test_getters_after_solve(bench):
    m, x = _model_with_vars(100_000)
    m.setObjective(quicksum(x))
    m.optimize()
    sol = m.getBestSol()

    def run():
        for _ in range(6):
            for v in x:
                m.getVal(v)
                m.getSolVal(sol, v)
                v.getLbGlobal()
                v.getObj()

    bench(run)
