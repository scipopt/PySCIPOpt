import pprint
import pytest
from pyscipopt import Model, Variable, Expr
from time import time

try:
    import numpy as np
    have_np = True
except ImportError:
    have_np = False

# not repeating reason unnecessarily
@pytest.mark.skipif(not have_np, reason="numpy is not installed")
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
                assert matrix_variable[i][j][k].name == f"x{i*12 + j*4 + k + 1}"


    sum_all_expr = matrix_variable.sum()
    m.setObjective(sum_all_expr, "maximize")
    m.addCons(sum_all_expr <= 1)
    assert m.getNVars() == 3 * 3 * 4

    m.optimize()

    assert m.getStatus() == "optimal"
    assert m.getObjVal() == 1

def index_from_name(name: str) -> list:
    name = name[2:]
    return list(map(int, name.split("_")))    

@pytest.mark.skipif(not have_np, reason="numpy is not installed")
def test_expr_from_matrix_vars():
    m = Model()

    mvar = m.addMatrixVar(shape=(2, 2), vtype="B", name="A")
    mvar2 = m.addMatrixVar(shape=(2, 2), vtype="B", name="B")

    mvar_double = 2 * mvar
    for expr in np.nditer(mvar_double, flags=["refs_ok"]):
        expr = expr.item()
        assert(isinstance(expr, Expr))
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
    for expr in np.nditer(sum_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert(isinstance(expr, Expr))
        assert expr.degree() == 1
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 2
    
    dot_expr = mvar * mvar2
    for expr in np.nditer(dot_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert(isinstance(expr, Expr))
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
    for expr in np.nditer(mul_expr, flags=["refs_ok"]):
        expr = expr.item()
        assert(isinstance(expr, Expr))
        assert expr.degree() == 2
        expr_list = list(expr.terms.items())
        assert len(expr_list) == 2
        for term, coeff in expr_list:
            assert coeff == 1

            assert len(term) == 2



#     assert False

# @pytest.mark.skipif(have_np, reason="numpy is not installed")
# def test_multiply_matrixVariable():
#     m = Model()

#     matrix_variable1 = m.addMatrixVar()
#     matrix_variable2 = m.addMatrixVar()
#     m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)
#     m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)

#     assert False

# @pytest.mark.skipif(have_np, reason="numpy is not installed")
# def test_matrixVariable_performance():
#     m = Model()
#     start = time()
#     m.addMatrixVar(shape=(10000, 10000))
#     finish = time()
#     assert True