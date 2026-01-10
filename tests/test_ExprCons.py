import pytest

from pyscipopt import Expr, ExprCons, Model
from pyscipopt.scip import CONST, Term


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    return m, x


def test_init_error(model):
    with pytest.raises(TypeError):
        ExprCons({CONST: 1.0})

    m, x = model
    with pytest.raises(ValueError):
        ExprCons(Expr({Term(x): 1.0}))


def test_le_error(model):
    m, x = model

    cons = ExprCons(Expr({Term(x): 1.0}), 1, 1)

    with pytest.raises(TypeError):
        cons <= "invalid"

    with pytest.raises(TypeError):
        cons <= None

    with pytest.raises(TypeError):
        cons <= 1

    cons = ExprCons(Expr({Term(x): 1.0}), None, 1)
    with pytest.raises(TypeError):
        cons <= 1

    cons = ExprCons(Expr({Term(x): 1.0}), 1, None)
    with pytest.raises(AttributeError):
        cons._lhs = None  # force to None for catching the error


def test_ge_error(model):
    m, x = model

    cons = ExprCons(Expr({Term(x): 1.0}), 1, 1)

    with pytest.raises(TypeError):
        cons >= [1, 2, 3]

    with pytest.raises(TypeError):
        cons >= 1

    cons = ExprCons(Expr({Term(x): 1.0}), 1, None)
    with pytest.raises(TypeError):
        cons >= 1

    cons = ExprCons(Expr({Term(x): 1.0}), 1, None)
    with pytest.raises(AttributeError):
        cons._rhs = None  # force to None for catching the error


def test_eq_error(model):
    m, x = model

    with pytest.raises(NotImplementedError):
        ExprCons(Expr({Term(x): 1.0}), 1, 1) == 1.0


def test_bool(model):
    m, x = model

    with pytest.raises(TypeError):
        bool(ExprCons(Expr({Term(x): 1.0}), 1, 1))


def test_cmp(model):
    m, x = model

    assert str(1 <= (x <= 1)) == "ExprCons(Expr({Term(x): 1.0}), 1.0, 1.0)"
    assert str((1 <= x) <= 1) == "ExprCons(Expr({Term(x): 1.0}), 1.0, 1.0)"
