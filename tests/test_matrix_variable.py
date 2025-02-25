import pdb
import pprint
import pytest
from pyscipopt import Model, Variable, log, exp, cos, sin, sqrt
from pyscipopt import Expr, MatrixExpr, MatrixVariable, MatrixExprCons, MatrixConstraint, ExprCons
from time import time

import numpy as np


def test_catching_errors():
    m = Model()

    x = m.addVar()
    y = m.addMatrixVar(shape=(3, 3))
    rhs = np.ones((2, 1))

    with pytest.raises(Exception):
        m.addMatrixCons(x <= 1)

    with pytest.raises(Exception):
        m.addCons(y <= 3)

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


@pytest.mark.skip(reason="Performance test")
def test_performance():
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
