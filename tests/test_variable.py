import pytest

from pyscipopt import Model, quicksum, sqrt
from pyscipopt.scip import (
    CONST,
    Expr,
    ExprCons,
    ExprLike,
    GenExpr,
    Term,
    Variable,
)


@pytest.fixture
def model():
    """Tests for the M4 semantics: Variable is an ExprLike, not an Expr."""
    m = Model()
    m.hideOutput()
    x = m.addVar(name="x", lb=0, ub=10)
    y = m.addVar(name="y", lb=0, ub=10)
    return m, x, y


def test_isinstance_semantics(model):
    _, x, _ = model
    assert isinstance(x, ExprLike)
    assert not isinstance(x, Expr)
    assert isinstance(x, Variable)

    expr = x + 1
    assert isinstance(expr, Expr)
    assert not isinstance(expr, Variable)

    gen = sqrt(x)
    assert isinstance(gen, GenExpr)
    assert isinstance(gen, ExprLike)


def test_add_variables(model):
    _, x, y = model
    expr = x + y
    assert isinstance(expr, Expr)
    assert len(expr.terms) == 2
    assert expr[x] == 1.0
    assert expr[y] == 1.0
    assert expr[CONST] == 0.0


def test_multiply_variable_with_itself(model):
    _, x, _ = model
    square = x * x
    assert isinstance(square, Expr)
    assert len(square.terms) == 1
    assert square[Term(x) * Term(x)] == 1.0
    assert square[x] == 0.0


def test_subtract_variable_from_itself(model):
    _, x, _ = model
    diff = x - x
    assert isinstance(diff, Expr)
    assert diff[x] == 0.0
    assert diff[CONST] == 0.0

    # Normalization through ExprCons drops the zero coefficient.
    cons = ExprCons(diff, lhs=0.0)
    assert cons.expr.terms == {}


def test_quicksum_matches_addition(model):
    _, x, y = model
    quick = quicksum([x, y])
    added = x + y
    assert isinstance(quick, Expr)
    # quicksum starts from an Expr carrying an explicit zero constant term,
    # so compare the coefficients instead of the raw dict.
    assert quick[CONST] == added[CONST]
    assert len(quick.terms) - 1 == len(added.terms)
    for var in (x, y):
        assert quick[var] == added[var]


def test_operations_do_not_mutate_variable_view(model):
    _, x, y = model
    _ = x + y
    _ = 2 * x
    _ = quicksum([x, y])
    # x still represents exactly the variable term x
    assert str(x) == "x"
    assert x[x] == 1.0
    assert x[y] == 0.0


def test_degree_and_membership(model):
    _, x, y = model
    assert x.degree() == 1.0

    expr = 2 * x + 3 * y
    assert x in expr
    assert y in expr
    assert x not in (y + 1)
    assert set(expr) == {Term(x), Term(y)}


def test_chgReoptObjective_with_variable(model):
    m, x, y = model
    m.enableReoptimization()
    m.addCons(x + y <= 15)
    m.setObjective(x + y, sense="maximize")
    m.optimize()
    assert m.getObjVal() == pytest.approx(15.0)

    m.freeReoptSolve()
    # Regression: Variable used to be an Expr with a .terms attribute.
    m.chgReoptObjective(x, sense="maximize")
    m.optimize()
    assert m.getVal(x) == pytest.approx(10.0)
