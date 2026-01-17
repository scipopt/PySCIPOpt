import numpy as np
import pytest

from pyscipopt import Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import Term


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    return m, x, y


def test_getitem(model):
    m, x, y = model

    assert x[x] == 1
    assert y[Term(y)] == 1


def test_iter(model):
    m, x, y = model

    assert list(x) == [Term(x)]


def test_add(model):
    m, x, y = model

    assert str(x + y) == "Expr({Term(x): 1.0, Term(y): 1.0})"
    assert str(0 + x) == "Expr({Term(x): 1.0})"

    y += y
    assert str(y) == "Expr({Term(y): 2.0})"


def test_sub(model):
    m, x, y = model

    assert str(1 - x) == "Expr({Term(x): -1.0, Term(): 1.0})"
    assert str(y - x) == "Expr({Term(y): 1.0, Term(x): -1.0})"

    y -= x
    assert str(y) == "Expr({Term(y): 1.0, Term(x): -1.0})"


def test_mul(model):
    m, x, y = model

    assert str(0 * x) == "Expr({Term(): 0.0})"
    assert str((2 * x) * y) == "Expr({Term(x, y): 2.0})"

    y *= -1
    assert str(y) == "Expr({Term(y): -1.0})"


def test_div(model):
    m, x, y = model

    assert str(x / x) == "Expr({Term(): 1.0})"
    assert str(1 / x) == "PowExpr(Expr({Term(x): 1.0}), -1.0)"
    assert str(1 / -x) == "PowExpr(Expr({Term(x): -1.0}), -1.0)"


def test_pow(model):
    m, x, y = model

    assert str(x**3) == "Expr({Term(x, x, x): 1.0})"
    assert str(3**x) == "ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(3.0)): 1.0}))"


def test_le(model):
    m, x, y = model

    assert str(x <= y) == "ExprCons(Expr({Term(x): 1.0, Term(y): -1.0}), None, 0.0)"


def test_ge(model):
    m, x, y = model

    assert str(x >= y) == "ExprCons(Expr({Term(x): 1.0, Term(y): -1.0}), 0.0, None)"


def test_eq(model):
    m, x, y = model

    assert str(x == y) == "ExprCons(Expr({Term(x): 1.0, Term(y): -1.0}), 0.0, 0.0)"


def test_abs(model):
    m, x, y = model
    assert str(abs(x)) == "AbsExpr(Term(x))"
    assert str(np.abs([x, y])) == "[AbsExpr(Term(x)) AbsExpr(Term(y))]"


def test_exp(model):
    m, x, y = model

    expr = exp([x, y])
    assert type(expr) is np.ndarray
    assert str(expr) == "[ExpExpr(Term(x)) ExpExpr(Term(y))]"
    assert str(expr) == str(np.exp([x, y]))


def test_log(model):
    m, x, y = model

    expr = log([x, y])
    assert type(expr) is np.ndarray
    assert str(expr) == "[LogExpr(Term(x)) LogExpr(Term(y))]"
    assert str(expr) == str(np.log([x, y]))


def test_sin(model):
    m, x, y = model

    expr = sin([x, y])
    assert type(expr) is np.ndarray
    assert str(expr) == "[SinExpr(Term(x)) SinExpr(Term(y))]"
    assert str(expr) == str(np.sin([x, y]))
    assert str(expr) == str(sin(np.array([x, y])))
    assert str(expr) == str(np.sin(np.array([x, y])))


def test_cos(model):
    m, x, y = model

    expr = cos([x, y])
    assert type(expr) is np.ndarray
    assert str(expr) == "[CosExpr(Term(x)) CosExpr(Term(y))]"
    assert str(expr) == str(np.cos([x, y]))


def test_sqrt(model):
    m, x, y = model

    expr = sqrt([x, y])
    assert type(expr) is np.ndarray
    assert str(expr) == "[SqrtExpr(Term(x)) SqrtExpr(Term(y))]"
    assert str(expr) == str(np.sqrt([x, y]))


def test_degree(model):
    m, x, y = model

    assert x.degree() == 1
    assert y.degree() == 1
