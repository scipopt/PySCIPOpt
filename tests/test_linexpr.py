import pytest

from pyscipopt import Model
from pyscipopt.scip import Expr, ExprCons, Term, quicksum

@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z

CONST = Term()

def test_term(model):
    m, x, y, z = model
    assert x[x] == 1.0
    assert x[y] == 0.0

def test_operations_linear(model):
    m, x, y, z = model
    expr = x + y
    assert isinstance(expr, Expr)
    assert expr[x] == 1.0
    assert expr[y] == 1.0
    assert expr[z] == 0.0

    expr = -x
    assert isinstance(expr, Expr)
    assert expr[x] == -1.0
    assert expr[y] ==  0.0

    expr = x*4
    assert isinstance(expr, Expr)
    assert expr[x] == 4.0
    assert expr[y] == 0.0

    expr = 4*x
    assert isinstance(expr, Expr)
    assert expr[x] == 4.0
    assert expr[y] == 0.0

    expr = x + y + x
    assert isinstance(expr, Expr)
    assert expr[x] == 2.0
    assert expr[y] == 1.0

    expr = x + y - x
    assert isinstance(expr, Expr)
    assert expr[x] == 0.0
    assert expr[y] == 1.0

    expr = 3*x + 1.0
    assert isinstance(expr, Expr)
    assert expr[x] == 3.0
    assert expr[y] == 0.0
    assert expr[CONST] == 1.0

    expr = 1.0 + 3*x
    assert isinstance(expr, Expr)
    assert expr[x] == 3.0
    assert expr[y] == 0.0
    assert expr[CONST] == 1.0

def test_operations_quadratic(model):
    m, x, y, z = model
    expr = x*x
    assert isinstance(expr, Expr)
    assert expr[x] == 0.0
    assert expr[y] == 0.0
    assert expr[CONST] == 0.0
    assert expr[Term(x,x)] == 1.0

    expr = x*y
    assert isinstance(expr, Expr)
    assert expr[x] == 0.0
    assert expr[y] == 0.0
    assert expr[CONST] == 0.0
    assert expr[Term(x,y)] == 1.0

    expr = (x - 1)*(y + 1)
    assert isinstance(expr, Expr)
    assert expr[x] == 1.0
    assert expr[y] == -1.0
    assert expr[CONST] == -1.0
    assert expr[Term(x,y)] == 1.0

def test_power_for_quadratic(model):
    m, x, y, z = model
    expr = x**2 + x + 1
    assert isinstance(expr, Expr)
    assert expr[Term(x,x)] == 1.0
    assert expr[x] == 1.0
    assert expr[CONST] == 1.0
    assert len(expr.terms) == 3

    assert (x**2).terms == (x*x).terms
    assert ((x + 3)**2).terms == (x**2 + 6*x + 9).terms

def test_operations_poly(model):
    m, x, y, z = model
    expr = x*x*x + 2*y*y
    assert isinstance(expr, Expr)
    assert expr[x] == 0.0
    assert expr[y] == 0.0
    assert expr[CONST] == 0.0
    assert expr[Term(x,x,x)] == 1.0
    assert expr[Term(y,y)] == 2.0
    assert expr.terms == (x**3 + 2*y**2).terms

def test_degree(model):
    m, x, y, z = model
    expr = Expr()
    assert expr.degree() == 0

    expr = Expr() + 3.0
    assert expr.degree() == 0

    expr = x + 1
    assert expr.degree() == 1

    expr = x*x + y - 2
    assert expr.degree() == 2

    expr = (x + 1)*(y + 1)*(x - 1)
    assert expr.degree() == 3

def test_inequality(model):
    m, x, y, z = model
    expr = x + 2*y
    cons = expr <= 0
    assert isinstance(cons, ExprCons)
    assert cons._lhs is None
    assert cons._rhs == 0.0
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[CONST] == 0.0
    assert CONST not in cons.expr.terms

    cons = expr >= 5
    assert isinstance(cons, ExprCons)
    assert cons._lhs == 5.0
    assert cons._rhs is None
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[CONST] == 0.0
    assert CONST not in cons.expr.terms

    cons = 5 <= x + 2*y - 3
    assert isinstance(cons, ExprCons)
    assert cons._lhs == 8.0
    assert cons._rhs is None
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[CONST] == 0.0
    assert CONST not in cons.expr.terms

def test_ranged(model):
    m, x, y, z = model
    expr = x + 2*y
    cons = expr >= 3
    ranged = cons <= 5
    assert isinstance(ranged, ExprCons)
    assert ranged._lhs == 3.0
    assert ranged._rhs == 5.0
    assert ranged.expr[y] == 2.0
    assert ranged.expr[CONST] == 0.0

    # again, more or less directly:
    ranged = 3 <= (x + 2*y <= 5)
    assert isinstance(ranged, ExprCons)
    assert ranged._lhs == 3.0
    assert ranged._rhs == 5.0
    assert ranged.expr[y] == 2.0
    assert ranged.expr[CONST] == 0.0
    # we must use the parenthesis, because
    #     x <= y <= z
    # is a "chained comparison", which will be interpreted by Python
    # to be equivalent to
    #     (x <= y) and (y <= z)
    # where "and" can not be overloaded and the expressions in
    # parenthesis are coerced to booleans.

    with pytest.raises(TypeError):
        ranged = (x + 2*y <= 5) <= 3

    with pytest.raises(TypeError):
        ranged = 3 >= (x + 2*y <= 5)

    with pytest.raises(TypeError):
        ranged = (1 <= x + 2*y <= 5)

def test_equation(model):
    m, x, y, z = model
    equat = 2*x - 3*y == 1
    assert isinstance(equat, ExprCons)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1.0
    assert equat.expr[x] == 2.0
    assert equat.expr[y] == -3.0
    assert equat.expr[CONST] == 0.0

def test_objective(model):
    m, x, y, z = model

    # setting linear objective
    m.setObjective(x + y)

    # using quicksum
    m.setObjective(quicksum(2 * v for v in [x, y, z]))

    # setting affine objective
    m.setObjective(x + y + 1)
    assert m.getObjoffset() == 1