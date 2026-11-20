import pytest

from pyscipopt import Model, exp, quicksum, sqrt


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


def test_add_cons_linear(bench):
    n = 300_000

    def run(m, x):
        for i in range(n - 1):
            m.addConsLinear([x[i], x[i + 1]], [1.0, 2.0], rhs=10)

    bench(run, setup=lambda: (_model_with_vars(n), {}))


def test_add_nonlinear_conss(bench):
    n = 20_000

    def run(m, x):
        for i in range(n - 1):
            m.addCons(exp(x[i]) + x[i] * x[i + 1] + sqrt(x[i + 1] + 1) <= 10)

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


def test_enum_getters(bench):
    m, x = _model_with_vars(1000)
    m.setObjective(quicksum(x))
    m.optimize()
    sol = m.getBestSol()

    def run():
        for _ in range(1000):
            m.getStage()
            sol.getOrigin()
            for v in x:
                v.getImplType()

    bench(run)
