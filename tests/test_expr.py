import math

import numpy as np
import pytest

from pyscipopt import Model, cos, exp, log, quickprod, sin, sqrt
from pyscipopt.scip import CONST, Expr, ExprCons, GenExpr, MatrixGenExpr


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z


def test_upgrade(model):
    m, x, y, z = model
    expr = x + y
    assert isinstance(expr, Expr)
    expr += exp(z)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr -= exp(z)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr /= x
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr *= sqrt(x)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr **= 1.5
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    assert isinstance(expr + exp(x), GenExpr)
    assert isinstance(expr - exp(x), GenExpr)
    assert isinstance(expr/x, GenExpr)
    assert isinstance(expr * x**1.2, GenExpr)
    assert isinstance(sqrt(expr), GenExpr)
    assert isinstance(abs(expr), GenExpr)
    assert isinstance(log(expr), GenExpr)
    assert isinstance(exp(expr), GenExpr)
    assert isinstance(sin(expr), GenExpr)
    assert isinstance(cos(expr), GenExpr)

    with pytest.raises(ZeroDivisionError):
        expr /= 0.0

def test_genexpr_op_expr(model):
    m, x, y, z = model
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    genexpr += x**2
    assert isinstance(genexpr, GenExpr)
    genexpr += 1
    assert isinstance(genexpr, GenExpr)
    genexpr += x
    assert isinstance(genexpr, GenExpr)
    genexpr += 2 * y
    assert isinstance(genexpr, GenExpr)
    genexpr -= x**2
    assert isinstance(genexpr, GenExpr)
    genexpr -= 1
    assert isinstance(genexpr, GenExpr)
    genexpr -= x
    assert isinstance(genexpr, GenExpr)
    genexpr -= 2 * y
    assert isinstance(genexpr, GenExpr)
    genexpr *= x + y
    assert isinstance(genexpr, GenExpr)
    genexpr *= 2
    assert isinstance(genexpr, GenExpr)
    genexpr /= 2
    assert isinstance(genexpr, GenExpr)
    genexpr /= x + y
    assert isinstance(genexpr, GenExpr)
    assert isinstance(x**1.2 + x + y, GenExpr)
    assert isinstance(x**1.2 - x, GenExpr)
    assert isinstance(x**1.2 *(x+y), GenExpr)

def test_genexpr_op_genexpr(model):
    m, x, y, z = model
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    genexpr **= 2.2
    assert isinstance(genexpr, GenExpr)
    genexpr += exp(x)
    assert isinstance(genexpr, GenExpr)
    genexpr -= exp(x)
    assert isinstance(genexpr, GenExpr)
    genexpr /= log(x + 1)
    assert isinstance(genexpr, GenExpr)
    genexpr *= (x + y)**1.2
    assert isinstance(genexpr, GenExpr)
    genexpr /= exp(2)
    assert isinstance(genexpr, GenExpr)
    genexpr /= x + y
    assert isinstance(genexpr, GenExpr)
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    assert isinstance(sqrt(x) + genexpr, GenExpr)
    assert isinstance(exp(x) + genexpr, GenExpr)
    assert isinstance(sin(x) + genexpr, GenExpr)
    assert isinstance(cos(x) + genexpr, GenExpr)
    assert isinstance(1/x + genexpr, GenExpr)
    assert isinstance(1/x**1.5 - genexpr, GenExpr)
    assert isinstance(y/x - exp(genexpr), GenExpr)

    # sqrt(2) is not a constant expression and
    # we can only power to constant expressions!
    with pytest.raises(NotImplementedError):
        genexpr **= sqrt(2)


def test_degree(model):
    m, x, y, z = model
    expr = GenExpr()
    assert expr.degree() == float('inf')

# In contrast to Expr inequalities, we can't expect much of the sides
def test_inequality(model):
    m, x, y, z = model

    expr = x + 2*y
    assert isinstance(expr, Expr)
    cons = expr <= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs is None
    assert cons._rhs == 0.0

    assert isinstance(expr, Expr)
    cons = expr >= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs == 0.0
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = expr >= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs == 0.0 # NOTE: the 1 is passed to the other side because of the way GenExprs work
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = exp(expr) <= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._rhs == 0.0
    assert cons._lhs is None


def test_equation(model):
    m, x, y, z = model
    equat = 2*x**1.2 - 3*sqrt(y) == 1
    assert isinstance(equat, ExprCons)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1.0

    equat = exp(x+2*y) == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, GenExpr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 0.0

    equat = x == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, GenExpr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 0.0

def test_rpow_constant_base(model):
    m, x, y, z = model
    a = 2**x
    b = exp(x * log(2.0))
    assert isinstance(a, GenExpr)
    assert repr(a) == repr(b) # Structural equality is not implemented; compare strings
    m.addCons(2**x <= 1)

    with pytest.raises(ValueError):
        c = (-2)**x


def test_getVal_with_GenExpr():
    m = Model()
    x = m.addVar(lb=1, ub=1, name="x")
    y = m.addVar(lb=2, ub=2, name="y")
    z = m.addVar(lb=0, ub=0, name="z")
    m.optimize()

    # test "Expr({Term(x, y, z): 1.0})"
    assert m.getVal(z * x * y) == 0
    # test "Expr({Term(x): 1.0, Term(y): 1.0, Term(): 1.0})"
    assert m.getVal(x + y + 1) == 4
    # test "prod(1.0,sum(0.0,prod(1.0,x)),**(sum(0.0,prod(1.0,x)),-1))"
    assert m.getVal(x / x) == 1
    # test "prod(1.0,sum(0.0,prod(1.0,y)),**(sum(0.0,prod(1.0,x)),-1))"
    assert m.getVal(y / x) == 2
    # test "**(prod(1.0,**(sum(0.0,prod(1.0,x)),-1)),2)"
    assert m.getVal((1 / x) ** 2) == 1
    # test "sin(sum(0.0,prod(1.0,x)))"
    assert round(m.getVal(sin(x)), 6) == round(math.sin(1), 6)

    with pytest.raises(TypeError):
        m.getVal(1)

    with pytest.raises(ZeroDivisionError):
        m.getVal(1 / z)


def test_unary_ufunc(model):
    m, x, y, z = model

    res = "abs(sum(0.0,prod(1.0,x)))"
    assert str(abs(x)) == res
    assert str(np.absolute(x)) == res

    res = "[sin(sum(0.0,prod(1.0,x))) sin(sum(0.0,prod(1.0,y)))]"
    assert str(sin([x, y])) == res
    assert str(np.sin([x, y])) == res

    res = "[cos(sum(0.0,prod(1.0,x))) cos(sum(0.0,prod(1.0,y)))]"
    assert str(cos([x, y])) == res
    assert str(np.cos([x, y])) == res

    res = "[sqrt(sum(0.0,prod(1.0,x))) sqrt(sum(0.0,prod(1.0,y)))]"
    assert str(sqrt([x, y])) == res
    assert str(np.sqrt([x, y])) == res

    res = "[exp(sum(0.0,prod(1.0,x))) exp(sum(0.0,prod(1.0,y)))]"
    assert str(exp([x, y])) == res
    assert str(np.exp([x, y])) == res

    res = "[log(sum(0.0,prod(1.0,x))) log(sum(0.0,prod(1.0,y)))]"
    assert str(log([x, y])) == res
    assert str(np.log([x, y])) == res

    assert str(log([1, x])) == "[log(1.0) log(sum(0.0,prod(1.0,x)))]"

    assert str(sqrt(4)) == "sqrt(4.0)"
    assert str(sqrt([4, 4])) == "[sqrt(4.0) sqrt(4.0)]"
    assert str(exp(3)) == "exp(3.0)"
    assert str(exp([3, 3])) == "[exp(3.0) exp(3.0)]"
    assert str(log(5)) == "log(5.0)"
    assert str(log([5, 5])) == "[log(5.0) log(5.0)]"
    assert str(sin(1)) == "sin(1.0)"
    assert str(sin([[1, 1]])) == "[[sin(1.0) sin(1.0)]]"
    assert str(cos(1)) == "cos(1.0)"
    assert str(cos([[1]])) == "[[cos(1.0)]]"

    assert isinstance(sqrt(2), GenExpr)
    assert isinstance(sqrt([2, 2]), MatrixGenExpr)
    assert isinstance(sqrt([[2], [2]]), MatrixGenExpr)
    assert isinstance(sqrt([2, x]), MatrixGenExpr)
    assert isinstance(sqrt([[2], [x]]), MatrixGenExpr)

    # test invalid unary operations
    with pytest.raises(TypeError):
        np.arcsin(x)

    with pytest.raises(TypeError):
        # forbid modifying Variable/Expr/GenExpr in-place via out parameter
        np.sin(x, out=np.array([0]))

    # test np.negative
    assert str(np.negative(x)) == "Expr({Term(x): -1.0})"


def test_binary_ufunc(model):
    m, x, y, z = model

    # test np.add
    assert str(np.add(x, 1)) == "Expr({Term(x): 1.0, Term(): 1.0})"
    assert str(np.add(1, x)) == "Expr({Term(x): 1.0, Term(): 1.0})"
    a = np.array([1])
    assert str(np.add(x, a)) == "[Expr({Term(x): 1.0, Term(): 1.0})]"
    assert str(np.add(a, x)) == "[Expr({Term(x): 1.0, Term(): 1.0})]"

    # test np.subtract
    assert str(np.subtract(x, 1)) == "Expr({Term(x): 1.0, Term(): -1.0})"
    assert str(np.subtract(1, x)) == "Expr({Term(x): -1.0, Term(): 1.0})"
    assert str(np.subtract(x, a)) == "[Expr({Term(x): 1.0, Term(): -1.0})]"
    assert str(np.subtract(a, x)) == "[Expr({Term(x): -1.0, Term(): 1.0})]"

    # test np.multiply
    a = np.array([2])
    assert str(np.multiply(x, 2)) == "Expr({Term(x): 2.0})"
    assert str(np.multiply(2, x)) == "Expr({Term(x): 2.0})"
    assert str(np.multiply(x, a)) == "[Expr({Term(x): 2.0})]"
    assert str(np.multiply(a, x)) == "[Expr({Term(x): 2.0})]"

    # test np.divide
    assert str(np.divide(x, 2)) == "Expr({Term(x): 0.5})"
    assert str(np.divide(2, x)) == "prod(2.0,**(sum(0.0,prod(1.0,x)),-1))"
    assert str(np.divide(x, a)) == "[Expr({Term(x): 0.5})]"
    assert str(np.divide(a, x)) == "[prod(2.0,**(sum(0.0,prod(1.0,x)),-1))]"

    # test np.power
    assert str(np.power(x, 2)) == "Expr({Term(x, x): 1.0})"
    assert str(np.power(2, x)) == "exp(prod(1.0,sum(0.0,prod(1.0,x)),log(2.0)))"
    assert str(np.power(x, a)) == "[Expr({Term(x, x): 1.0})]"
    assert str(np.power(a, x)) == "[exp(prod(1.0,sum(0.0,prod(1.0,x)),log(2.0)))]"

    # test np.less_equal
    assert str(np.less_equal(x, a)) == "[ExprCons(Expr({Term(x): 1.0}), None, 2.0)]"
    assert str(np.less_equal(a, x)) == "[ExprCons(Expr({Term(x): 1.0}), 2.0, None)]"

    # test np.equal
    assert str(np.equal(x, a)) == "[ExprCons(Expr({Term(x): 1.0}), 2.0, 2.0)]"
    assert str(np.equal(a, x)) == "[ExprCons(Expr({Term(x): 1.0}), 2.0, 2.0)]"

    # test np.greater_equal
    assert str(np.greater_equal(x, a)) == "[ExprCons(Expr({Term(x): 1.0}), 2.0, None)]"
    assert str(np.greater_equal(a, x)) == "[ExprCons(Expr({Term(x): 1.0}), None, 2.0)]"


def test_mul():
    m = Model()
    x = m.addVar(name="x")
    y = m.addVar(name="y")

    # test Expr * number
    assert str((x + y) * 2.0) == "Expr({Term(x): 2.0, Term(y): 2.0})"
    assert str(2.0 * (x + y)) == "Expr({Term(x): 2.0, Term(y): 2.0})"

    # test Expr * Expr
    assert str(Expr({CONST: 1.0}) * x) == "Expr({Term(x): 1.0})"
    assert str(y * Expr({CONST: -1.0})) == "Expr({Term(y): -1.0})"
    assert str((x - x) * y) == "Expr({Term(x, y): 0.0})"
    assert str(y * (x - x)) == "Expr({Term(x, y): 0.0})"
    assert (
        str((x + 1) * (y - 1))
        == "Expr({Term(x, y): 1.0, Term(x): -1.0, Term(y): 1.0, Term(): -1.0})"
    )
    assert (
        str((x + 1) * (x + 1) * y)
        == "Expr({Term(x, x, y): 1.0, Term(x, y): 2.0, Term(y): 1.0})"
    )


def test_abs_abs_expr():
    m = Model()
    x = m.addVar(name="x")

    # should print abs(x) not abs(abs(x))
    assert str(abs(abs(x))) == str(abs(x))


def test_NotImplemented():
    m = Model()
    x = m.addVar(name="x")

    with pytest.raises(TypeError):
        "y" + x
    with pytest.raises(TypeError):
        x + "y"

    with pytest.raises(TypeError):
        y = "y"
        y += x
    with pytest.raises(TypeError):
        x += "y"

    with pytest.raises(TypeError):
        "y" * x
    with pytest.raises(TypeError):
        x * "y"

    with pytest.raises(TypeError):
        "y" / x
    with pytest.raises(TypeError):
        x / "y"

    with pytest.raises(TypeError):
        "1" <= x
    with pytest.raises(TypeError):
        x >= "1"
    with pytest.raises(TypeError):
        x >= "1"
    with pytest.raises(TypeError):
        "1" == x
    with pytest.raises(TypeError):
        x == "1"

    genexpr = sqrt(x)

    with pytest.raises(TypeError):
        "y" + genexpr
    with pytest.raises(TypeError):
        genexpr + "y"

    with pytest.raises(TypeError):
        y = "y"
        y += genexpr
    with pytest.raises(TypeError):
        genexpr += "y"

    with pytest.raises(TypeError):
        "y" * genexpr
    with pytest.raises(TypeError):
        genexpr * "y"

    with pytest.raises(TypeError):
        "y" / genexpr
    with pytest.raises(TypeError):
        genexpr / "y"

    with pytest.raises(TypeError):
        "1" <= genexpr
    with pytest.raises(TypeError):
        "1" >= genexpr
    with pytest.raises(TypeError):
        genexpr >= "1"
    with pytest.raises(TypeError):
        genexpr <= "1"
    with pytest.raises(TypeError):
        "1" == genexpr
    with pytest.raises(TypeError):
        genexpr == "1"

    # test Expr + GenExpr
    assert str(x + genexpr) == "sum(0.0,sqrt(sum(0.0,prod(1.0,x))),prod(1.0,x))"
    assert str(genexpr + x) == "sum(0.0,sqrt(sum(0.0,prod(1.0,x))),prod(1.0,x))"

    # test Expr * GenExpr
    assert (
        str(x * genexpr) == "prod(1.0,sqrt(sum(0.0,prod(1.0,x))),sum(0.0,prod(1.0,x)))"
    )

    # test Expr + array
    a = np.array([1])
    assert str(x + a) == "[Expr({Term(x): 1.0, Term(): 1.0})]"
    # test GenExpr + array
    assert str(genexpr + a) == "[sum(1.0,sqrt(sum(0.0,prod(1.0,x))))]"

    a = m.addMatrixVar(1)
    # test Expr >= array
    assert str(x >= a) == "[ExprCons(Expr({Term(x2): 1.0, Term(x): -1.0}), None, 0.0)]"
    # test GenExpr >= array
    assert (
        str(genexpr >= a)
        == "[ExprCons(sum(0.0,prod(-1.0,sqrt(sum(0.0,prod(1.0,x)))),prod(1.0,x2)), None, 0.0)]"
    )
    # test Expr <= array
    assert str(x <= a) == "[ExprCons(Expr({Term(x2): 1.0, Term(x): -1.0}), 0.0, None)]"
    # test GenExpr <= array
    assert (
        str(genexpr <= a)
        == "[ExprCons(sum(0.0,prod(-1.0,sqrt(sum(0.0,prod(1.0,x)))),prod(1.0,x2)), 0.0, None)]"
    )
    # test Expr == array
    assert str(x == a) == "[ExprCons(Expr({Term(x2): 1.0, Term(x): -1.0}), 0.0, 0.0)]"
    # test GenExpr == array
    assert (
        str(genexpr == a)
        == "[ExprCons(sum(0.0,prod(-1.0,sqrt(sum(0.0,prod(1.0,x)))),prod(1.0,x2)), 0.0, 0.0)]"
    )

    # test Expr += GenExpr
    x += genexpr
    assert str(x) == "sum(0.0,sqrt(sum(0.0,prod(1.0,x))),prod(1.0,x))"


def test_term_eq():
    m = Model()

    x = m.addMatrixVar(1000)
    y = m.addVar()
    z = m.addVar()

    e1 = quickprod(x.flat)
    e2 = quickprod(x.flat)
    t1 = next(iter(e1))
    t2 = next(iter(e2))
    t3 = next(iter(e1 * y))
    t4 = next(iter(e2 * z))

    assert t1 == t1  # same term
    assert t1 == t2  # same term
    assert t3 != t4  # same length, but different term
    assert t1 != t3  # different length
    assert t1 != "not a term"  # different type
