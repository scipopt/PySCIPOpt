import pytest
from pyscipopt import Model, Variable
from time import time

try:
    import numpy as np
    have_np = True
except ImportError:
    have_np = False

@pytest.mark.skipif(not have_np, reason="numpy is not installed")
def test_create_matrixVariable():
    # will probably scrap create_matrixVariable
    return

# not repeating reason unnecessarily
@pytest.mark.skipif(not have_np, reason="numpy is not installed")
def test_add_matrixVar():
    m = Model()
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

    matrix_variable = m.addMatrixVar(shape=(3, 3, 4), name="", vtype=vtypes, ub=8,
                                     lb=np.ndarray((3, 3, 4), dtype=object))

    for i in range(3):
        for j in range(3):
            for k in range(4):
                if i == 0:
                    assert matrix_variable[i][j][k].vtype() == "CONTINUOUS"
                elif i == 1:
                    assert matrix_variable[i][j][k].vtype() == "BINARY"
                else:
                    assert matrix_variable[i][j][k].vtype() == "INTEGER"
                
                assert isinstance(matrix_variable[i][j][k], Variable)
                assert "x" in matrix_variable[i][j][k].name
                print(matrix_variable[i][j][k].getLbOriginal())
                print(i,j,k)
                # assert m.isInfinity(-matrix_variable[i][j][k].getLbOriginal())
                # assert matrix_variable[i][j][k].getUbOriginal() == 8

    assert False
    return

@pytest.mark.skipif(have_np, reason="numpy is not installed")
def test_create_cons_with_matrixVariable():
    m = Model()

    m.addMarixVariable()
    m.addMatrixCons()

    m.optimize()

    assert False

@pytest.mark.skipif(have_np, reason="numpy is not installed")
def test_multiply_matrixVariable():
    m = Model()

    matrix_variable1 = m.addMatrixVar()
    matrix_variable2 = m.addMatrixVar()
    m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)
    m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)

    assert False

@pytest.mark.skipif(have_np, reason="numpy is not installed")
def test_matrixVariable_performance():
    m = Model()
    start = time()
    m.addMatrixVar(shape=(10000, 10000))
    finish = time()
    assert True