import numpy as np

from pyscipopt import Model, exp, quicksum, sqrt


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


def test_quadratic_expr(bench):
    _, x = _vars(1000)
    bench(lambda: quicksum(x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))))


def test_nonlinear_expr(bench):
    _, x = _vars(10_000)
    bench(lambda: quicksum(exp(v) + sqrt(v + 1) for v in x) <= 10)


def test_matrix_ops(bench):
    n = 100
    m = Model()
    x = m.addMatrixVar((n, n))
    a = np.arange(n * n, dtype=float).reshape(n, n)

    def run():
        x.sum(axis=0)
        a @ x
        return x <= 1

    bench(run)
