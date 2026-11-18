import numpy as np

from pyscipopt import Model, cos, exp, log, quicksum, sin, sqrt


def _vars(n):
    m = Model()
    return m, [m.addVar() for _ in range(n)]


def test_quicksum(bench):
    _, x = _vars(100_000)
    bench(lambda: quicksum(c * v for c in range(1, 9) for v in x))


def test_expr_arithmetic(bench):
    _, x = _vars(8000)

    def run():
        e = 0
        for i, v in enumerate(x):
            e = e + 3 * v - i
        return e <= 1

    bench(run)


def test_repeated_terms(bench):
    _, x = _vars(100)
    bench(lambda: quicksum(float(i) * x[i % 100] for i in range(1_000_000)))


def test_expr_ops(bench):
    _, x = _vars(3000)
    exprs = [x[i] + 2 * x[i + 1] - 3 for i in range(len(x) - 1)]

    def run():
        for _ in range(30):
            for e, f in zip(exprs, exprs[1:]):
                -e
                3.0 * e
                e + f
                e - f
                e * f

    bench(run)


def test_quadratic_expr(bench):
    _, x = _vars(1000)
    bench(lambda: quicksum(x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))))


def test_nonlinear_expr(bench):
    _, x = _vars(10_000)
    bench(lambda: quicksum(exp(v) + sqrt(v + 1) for v in x) <= 10)


def test_evaluate_nonlinear(bench):
    m, x = _vars(3000)
    exprs = [exp(v) + log(v + 1) + sqrt(v + 2) + sin(v) + cos(v) + v**2.5 + abs(v - 1) for v in x]
    sol = m.createSol()
    for i, v in enumerate(x):
        m.setSolVal(sol, v, i / len(x))

    def run():
        for _ in range(100):
            for e in exprs:
                sol[e]

    bench(run)


def test_matrix_ops(bench):
    n = 100
    m = Model()
    x = m.addMatrixVar((n, n))
    a = np.arange(n * n, dtype=float).reshape(n, n)

    def run():
        x.sum()
        x.sum(axis=0)
        np.add.reduce(x)
        -x
        2 * x
        a @ x
        return x <= 1

    bench(run)
