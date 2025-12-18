import operator
from time import time

import numpy as np
import pytest

from pyscipopt import (
    Expr,
    ExprCons,
    MatrixConstraint,
    MatrixExpr,
    MatrixExprCons,
    MatrixVariable,
    Model,
    Variable,
    cos,
    exp,
    log,
    sin,
    sqrt,
)
from pyscipopt.scip import CONST, GenExpr


def test_catching_errors():
    m = Model()

    x = m.addVar()
    y = m.addMatrixVar(shape=(3, 3))
    rhs = np.ones((2, 1))

    # require ExprCons
    with pytest.raises(Exception):
        m.addCons(y <= 3)

    # require MatrixExprCons or ExprCons
    with pytest.raises(Exception):
        m.addMatrixCons(x)

    # test shape mismatch
    with pytest.raises(Exception):
        m.addMatrixCons(y <= rhs)


def test_add_matrixVar():
    m = Model()
    m.hideOutput()
    vtypes = np.ndarray((3, 3, 4), dtype=object)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                if i == 0:
                    vtypes[i][j][k] = "C"
                elif i == 1:
                    vtypes[i][j][k] = "B"
                else:
                    vtypes[i][j][k] = "I"

    matrix_variable = m.addMatrixVar(shape=(3, 3, 4), name="", vtype=vtypes, ub=8.5, obj=1.0,
                                     lb=np.ndarray((3, 3, 4), dtype=object))

    assert (isinstance(matrix_variable, MatrixVariable))
    for i in range(3):
        for j in range(3):
            for k in range(4):
                if i == 0:
                    assert matrix_variable[i][j][k].vtype() == "CONTINUOUS"
                    assert m.isInfinity(-matrix_variable[i][j][k].getLbOriginal())
                    assert m.isEQ(matrix_variable[i][j][k].getUbOriginal(), 8.5)
                elif i == 1:
                    assert matrix_variable[i][j][k].vtype() == "BINARY"
                    assert m.isEQ(matrix_variable[i][j][k].getLbOriginal(), 0)
                    assert m.isEQ(matrix_variable[i][j][k].getUbOriginal(), 1)
                else:
                    assert matrix_variable[i][j][k].vtype() == "INTEGER"
                    assert m.isInfinity(-matrix_variable[i][j][k].getLbOriginal())
                    assert m.isEQ(matrix_variable[i][j][k].getUbOriginal(), 8.5)

                assert isinstance(matrix_variable[i][j][k], Variable)
                assert matrix_variable[i][j][k].name == f"x{i * 12 + j * 4 + k + 1}"

    sum_all_expr = matrix_variable.sum()
    m.setObjective(sum_all_expr, "maximize")
    m.addCons(sum_all_expr <= 1)
    assert m.getNVars() == 3 * 3 * 4

    m.optimize()

    assert m.getStatus() == "optimal"
    assert m.getObjVal() == 1

    sol = m.getBestSol()

    sol_matrix = sol[matrix_variable]
    assert sol_matrix.shape == (3, 3, 4)
    assert m.isEQ(sol_matrix.sum(), 1)


def index_from_name(name: str) -> list:
    name = name[2:]
    return list(map(int, name.split("_")))


def test_expr_from_matrix_vars():
    m = Model()

    mvar = m.addMatrixVar(shape=(2, 2), vtype="B", name="A")
    mvar2 = m.addMatrixVar(shape=(2, 2), vtype="B", name="B")

    mvar_double = 2 * mvar
    assert isinstance(mvar_double, MatrixExpr)
    for expr in np.nditer(mvar_double, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 1
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 1
        first_term, coeff = expr_list[0]
        assert coeff == 2
        vars_in_term = list(first_term)
        first_var_in_term = vars_in_term[0]
        assert isinstance(first_var_in_term, Variable)
        assert first_var_in_term.vtype() == "BINARY"

    sum_expr = mvar + mvar2
    assert isinstance(sum_expr, MatrixExpr)
    for expr in np.nditer(sum_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 1
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 2

    dot_expr = mvar * mvar2
    assert isinstance(dot_expr, MatrixExpr)
    for expr in np.nditer(dot_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 2
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 1
        for term, coeff in expr_list:
            assert coeff == 1
            assert len(term) == 2
            vars_in_term = list(term)
            indices = [index_from_name(var.name) for var in vars_in_term]
            assert indices[0] == indices[1]

    mul_expr = mvar @ mvar2
    assert isinstance(mul_expr, MatrixExpr)
    for expr in np.nditer(mul_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 2
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 2
        for term, coeff in expr_list:
            assert coeff == 1

            assert len(term) == 2

    power_3_expr = mvar ** 3
    assert isinstance(power_3_expr, MatrixExpr)
    for expr in np.nditer(power_3_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 3
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 1
        for term, coeff in expr_list:
            assert coeff == 1
            assert len(term) == 3

    power_3_mat_expr = np.linalg.matrix_power(mvar, 3)
    assert isinstance(power_3_expr, MatrixExpr)
    for expr in np.nditer(power_3_mat_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert (isinstance(expr, Expr))
        assert expr.degree() == 3
        expr_list = list(expr.terms.items())
        for term, coeff in expr_list:
            assert len(term) == 3


def test_matrix_sum_error():
    m = Model()
    x = m.addMatrixVar((2, 3), "x", "I", ub=4)

    # test axis type
    with pytest.raises(TypeError):
        x.sum("0")

    # test axis value (out of range)
    with pytest.raises(ValueError):
        x.sum(2)

    # test axis value (out of range)
    with pytest.raises(ValueError):
        x.sum((-3,))

    # test axis value (duplicate)
    with pytest.raises(ValueError):
        x.sum((0, 0))


def test_matrix_sum_axis():
    # compare the result of summing matrix variable after optimization
    m = Model()

    # Return a array when axis isn't None
    res = m.addMatrixVar((3, 1)).sum(axis=0)
    assert isinstance(res, MatrixExpr) and res.shape == (1,)

    # compare the result of summing 2d array to a scalar with a scalar
    x = m.addMatrixVar((2, 3), "x", "I", ub=4)
    # `axis=tuple(range(x.ndim))` is `axis=None`
    m.addMatrixCons(x.sum(axis=tuple(range(x.ndim))) == 24)

    # compare the result of summing 2d array to 1d array
    y = m.addMatrixVar((2, 4), "y", "I", ub=4)
    m.addMatrixCons(x.sum(axis=1) == y.sum(axis=1))

    # compare the result of summing 3d array to a 2d array with a 2d array
    z = m.addMatrixVar((2, 3, 4), "z", "I", ub=4)
    m.addMatrixCons(z.sum(2) == x)
    m.addMatrixCons(z.sum(axis=1) == y)

    # to fix the element values
    m.addMatrixCons(z == np.ones((2, 3, 4)))

    m.setObjective(x.sum() + y.sum() + z.sum(tuple(range(z.ndim))), "maximize")
    m.optimize()

    assert (m.getVal(x) == np.full((2, 3), 4)).all().all()
    assert (m.getVal(y) == np.full((2, 4), 3)).all().all()


@pytest.mark.parametrize(
    "axis, keepdims",
    [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 2), False),
        ((0, 2), True),
    ],
)
def test_matrix_sum_result(axis, keepdims):
    # directly compare the result of np.sum and MatrixExpr.sum
    _getVal = np.vectorize(lambda e: e.terms[CONST])
    a = np.arange(6).reshape((1, 2, 3))

    np_res = a.sum(axis, keepdims=keepdims)
    scip_res = MatrixExpr.sum(a, axis, keepdims=keepdims)
    assert (np_res == _getVal(scip_res)).all()
    assert np_res.shape == _getVal(scip_res).shape


@pytest.mark.parametrize("n", [50, 100])
def test_matrix_sum_axis_is_none_performance(n):
    model = Model()
    x = model.addMatrixVar((n, n))

    # Original sum via `np.ndarray.sum`, `np.sum` will call subclass method
    start_orig = time()
    np.ndarray.sum(x)
    end_orig = time()

    # Optimized sum via `quicksum`
    start_matrix = time()
    x.sum()
    end_matrix = time()

    assert model.isGT(end_orig - start_orig, end_matrix - start_matrix)


@pytest.mark.parametrize("n", [50, 100])
def test_matrix_sum_axis_not_none_performance(n):
    model = Model()
    x = model.addMatrixVar((n, n))

    # Original sum via `np.ndarray.sum`, `np.sum` will call subclass method
    start_orig = time()
    np.ndarray.sum(x, axis=0)
    end_orig = time()

    # Optimized sum via `quicksum`
    start_matrix = time()
    x.sum(axis=0)
    end_matrix = time()

    assert model.isGT(end_orig - start_orig, end_matrix - start_matrix)


def test_add_cons_matrixVar():
    m = Model()
    matrix_variable = m.addMatrixVar(shape=(3, 3), vtype="B", name="A", obj=1)
    other_matrix_variable = m.addMatrixVar(shape=(3, 3), vtype="B", name="B")
    single_var = m.addVar(vtype="B", name="x")

    # all supported use cases
    c = matrix_variable <= np.ones((3, 3))
    assert isinstance(c, MatrixExprCons)
    d = matrix_variable <= 1
    assert isinstance(c, MatrixExprCons)
    for i in range(3):
        for j in range(3):
            expr_c = c[i][j].expr
            expr_d = d[i][j].expr
            assert isinstance(expr_c, Expr)
            assert isinstance(expr_d, Expr)
            assert m.isEQ(c[i][j]._rhs, 1)
            assert m.isEQ(d[i][j]._rhs, 1)
            for _, coeff in list(expr_c.terms.items()):
                assert m.isEQ(coeff, 1)
            for _, coeff in list(expr_d.terms.items()):
                assert m.isEQ(coeff, 1)
    c = matrix_variable <= other_matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = matrix_variable <= single_var
    assert isinstance(c, MatrixExprCons)
    c = 1 <= matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = np.ones((3, 3)) <= matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = other_matrix_variable <= matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = single_var <= matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = single_var >= matrix_variable
    assert isinstance(c, MatrixExprCons)
    c = single_var == matrix_variable
    assert isinstance(c, MatrixExprCons)

    sum_expr = matrix_variable + single_var
    assert isinstance(sum_expr, MatrixExpr)
    sum_expr = single_var + matrix_variable
    assert isinstance(sum_expr, MatrixExpr)

    m.addMatrixCons(matrix_variable >= 1)

    log(matrix_variable)
    exp(matrix_variable)
    cos(matrix_variable)
    sin(matrix_variable)
    sqrt(matrix_variable)
    log(log(matrix_variable))
    log(log(matrix_variable)) <= 9

    m.addMatrixCons(matrix_variable <= other_matrix_variable)
    m.addMatrixCons(log(matrix_variable) <= other_matrix_variable)
    m.addMatrixCons(exp(matrix_variable) <= other_matrix_variable)
    m.addMatrixCons(sqrt(matrix_variable) <= other_matrix_variable)
    m.addMatrixCons(sin(matrix_variable) <= 37)
    m.addMatrixCons(cos(matrix_variable) <= other_matrix_variable)

    m.optimize()


def test_add_conss_matrixCons():
    m = Model()
    matrix_variable = m.addMatrixVar(shape=(2, 3, 4, 5), vtype="B", name="A", obj=1)

    conss = m.addConss(matrix_variable <= 2)

    assert len(conss) == 2 * 3 * 4 * 5
    assert m.getNConss() == 2 * 3 * 4 * 5


def test_correctness():
    m = Model()
    x = m.addMatrixVar(shape=(2, 2), vtype="I", name="x", obj=np.array([[5, 1], [4, 9]]), lb=np.array([[1, 2], [3, 4]]))
    y = m.addMatrixVar(shape=(2, 2), vtype="I", name="y", obj=np.array([[3, 4], [8, 3]]), lb=np.array([[5, 6], [7, 8]]))

    res = x * y
    m.addMatrixCons(res >= 15)
    m.optimize()

    assert np.array_equal(m.getVal(res), np.array([[15, 18], [21, 32]]))


def test_documentation():
    m = Model()
    shape = (2, 2)
    x = m.addMatrixVar(shape, vtype='C', name='x', ub=8)
    assert x[0][0].name == "x_0_0"
    assert x[0][1].name == "x_0_1"
    assert x[1][0].name == "x_1_0"
    assert x[1][1].name == "x_1_1"

    x = m.addMatrixVar(shape, vtype='C', name='x', ub=np.array([[5, 6], [2, 8]]))
    assert x[0][0].getUbGlobal() == 5
    assert x[0][1].getUbGlobal() == 6
    assert x[1][0].getUbGlobal() == 2
    assert x[1][1].getUbGlobal() == 8

    x = m.addMatrixVar(shape=(2, 2), vtype="B", name="x")
    y = m.addMatrixVar(shape=(2, 2), vtype="C", name="y", ub=5)
    z = m.addVar(vtype="C", name="z", ub=7)

    c1 = m.addMatrixCons(x + y <= z)
    c2 = m.addMatrixCons(exp(x) + sin(sqrt(y)) == z + y)
    e1 = x @ y
    c3 = m.addMatrixCons(y <= e1)
    c4 = m.addMatrixCons(e1 <= x)
    c4 = m.addCons(x.sum() <= 2)

    assert (isinstance(x, MatrixVariable))
    assert (isinstance(c1, MatrixConstraint))
    assert (isinstance(e1, MatrixExpr))

    x = m.addVar()
    matrix_x = m.addMatrixVar(shape=(2, 2))

    assert (x.vtype() == matrix_x[0][0].vtype())

    x = m.addMatrixVar(shape=(2, 2))
    assert (isinstance(x, MatrixVariable))
    assert (isinstance(x[0][0], Variable))
    cons = x <= 2
    assert (isinstance(cons, MatrixExprCons))
    assert (isinstance(cons[0][0], ExprCons))

def test_MatrixVariable_attributes():
    m = Model()
    x = m.addMatrixVar(shape=(2,2), vtype='C', name='x', ub=np.array([[5, 6], [2, 8]]), obj=1)
    assert x.vtype().tolist() == [['CONTINUOUS', 'CONTINUOUS'], ['CONTINUOUS', 'CONTINUOUS']]
    assert x.isInLP().tolist() == [[False, False], [False, False]]
    assert x.getIndex().tolist() == [[0, 1], [2, 3]]
    assert x.getLbGlobal().tolist() == [[0, 0], [0, 0]]
    assert x.getUbGlobal().tolist() == [[5, 6], [2, 8]]
    assert x.getObj().tolist() == [[1, 1], [1, 1]]
    m.setMaximize()
    m.optimize()
    assert x.getUbLocal().tolist() == [[5, 6], [2, 8]]
    assert x.getLbLocal().tolist() == [[5, 6], [2, 8]]
    assert x.getLPSol().tolist() == [[5, 6], [2, 8]]
    assert x.getAvgSol().tolist() == [[5, 6], [2, 8]]
    assert x.varMayRound().tolist() == [[True, True], [True, True]]

@pytest.mark.skip(reason="Performance test")
def test_add_cons_performance():
    start_orig = time()
    m = Model()
    x = {}
    for i in range(1000):
        for j in range(100):
            x[(i, j)] = m.addVar(vtype="C", obj=1)

    for i in range(1000):
        for j in range(100):
            m.addCons(x[i, j] <= 1)

    end_orig = time()

    m = Model()
    start_matrix = time()
    x = m.addMatrixVar(shape=(1000, 100), vtype="C", obj=1)
    m.addMatrixCons(x <= 1)
    end_matrix = time()

    matrix_time = end_matrix - start_matrix
    orig_time = end_orig - start_orig

    assert m.isGT(orig_time, matrix_time)


def test_matrix_cons_indicator():
    m = Model()
    x = m.addMatrixVar((2, 3), vtype="I", ub=10)
    y = m.addMatrixVar(x.shape, vtype="I", ub=10)
    is_equal = m.addMatrixVar((1, 2), vtype="B")

    # shape of cons is not equal to shape of is_equal
    with pytest.raises(Exception):
        m.addMatrixConsIndicator(x >= y, is_equal)

    # require MatrixExprCons or ExprCons
    with pytest.raises(TypeError):
        m.addMatrixConsIndicator(x)

    # test MatrixExprCons
    for i in range(2):
        m.addMatrixConsIndicator(x[i] >= y[i], is_equal[0, i])
        m.addMatrixConsIndicator(x[i] <= y[i], is_equal[0, i])

        m.addMatrixConsIndicator(x[i] >= 5, is_equal[0, i])
        m.addMatrixConsIndicator(y[i] <= 5, is_equal[0, i])

    for i in range(3):
        m.addMatrixConsIndicator(x[:, i] >= y[:, i], is_equal[0])
        m.addMatrixConsIndicator(x[:, i] <= y[:, i], is_equal[0])

    # test ExprCons
    z = m.addVar(vtype="B")
    binvar = m.addVar(vtype="B")
    m.addMatrixConsIndicator(z >= 1, binvar, activeone=True)
    m.addMatrixConsIndicator(z <= 0, binvar, activeone=False)

    m.setObjective(is_equal.sum() + binvar, "maximize")
    m.optimize()

    assert m.getVal(is_equal).sum() == 2
    assert (m.getVal(x) == m.getVal(y)).all().all()
    assert (m.getVal(x) == np.array([[5, 5, 5], [5, 5, 5]])).all().all()
    assert m.getVal(z) == 1


def test_matrix_compare_with_expr():
    m = Model()
    var = m.addVar(vtype="B", ub=0)

    # test "<=" and ">=" operator
    x = m.addMatrixVar(3)
    m.addMatrixCons(x <= var + 1)
    m.addMatrixCons(x >= var + 1)

    # test "==" operator
    y = m.addMatrixVar(3)
    m.addMatrixCons(y == var + 1)

    m.setObjective(x.sum() + y.sum())
    m.optimize()

    assert (m.getVal(x) == np.ones(3)).all()
    assert (m.getVal(y) == np.ones(3)).all()


def test_ranged_matrix_cons_with_expr():
    m = Model()
    x = m.addMatrixVar(3)
    var = m.addVar(vtype="B", ub=0)

    # test MatrixExprCons vs Variable
    with pytest.raises(TypeError):
        m.addMatrixCons((x <= 1) >= var)

    # test "==" operator
    with pytest.raises(NotImplementedError):
        m.addMatrixCons((x <= 1) == 1)

    # test "<=" and ">=" operator
    m.addMatrixCons((x <= 1) >= 1)

    m.setObjective(x.sum())
    m.optimize()

    assert (m.getVal(x) == np.ones(3)).all()


_binop_model = Model()

def var():
    return _binop_model.addVar()

def genexpr():
    return _binop_model.addVar() ** 0.6

def matvar():
    return _binop_model.addMatrixVar((1,))

@pytest.mark.parametrize("right", [var(), genexpr(), matvar()], ids=["var", "genexpr", "matvar"])
@pytest.mark.parametrize("left", [var(), genexpr(), matvar()], ids=["var", "genexpr", "matvar"])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_binop(op, left, right):
    res = op(left, right)
    assert isinstance(res, (Expr, GenExpr, MatrixExpr))


def test_matrix_matmul_return_type():
    # test #1058, require returning type is MatrixExpr not MatrixVariable
    m = Model()

    # test 1D @ 1D → 0D
    x = m.addMatrixVar(3)
    assert type(x @ x) is MatrixExpr

    # test 1D @ 1D → 2D
    assert type(x[:, None] @ x[None, :]) is MatrixExpr

    # test 2D @ 2D → 2D
    y = m.addMatrixVar((2, 3))
    z = m.addMatrixVar((3, 4))
    assert type(y @ z) is MatrixExpr


def test_matrix_sum_return_type():
    # test #1117, require returning type is MatrixExpr not MatrixVariable
    m = Model()

    x = m.addMatrixVar((3, 2))
    assert type(x.sum(axis=1)) is MatrixExpr


def test_broadcast():
    # test #1065
    m = Model()
    x = m.addMatrixVar((2, 3), ub=10)

    m.addMatrixCons(x == np.zeros((2, 1)))

    m.setObjective(x.sum(), "maximize")
    m.optimize()

    assert (m.getVal(x) == np.zeros((2, 3))).all()
