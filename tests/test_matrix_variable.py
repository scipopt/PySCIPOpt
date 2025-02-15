import pytest
from pyscipopt import Model, Variable
from time import time

try:
    import numpy as np
except ImportError:
    have_np = False

@pytest.mark.skipif(not have_np, reason="numpy is not installed")
def test_create_matrixVariable():
    # will probably scrap create_matrixVariable
    return

# not repeating reason unnecessarily
@pytest.mark.skipif(not have_np)
def test_add_matrixVariable():
    m = Model()
    types=np.shape(3,3)
    for i in range(3):
        for j in range(3):
            if i == 0:
                types[i][j] = "C"
            elif i == 1:
                types[i][j] = "B"
            else:
                types[i][j] = "I"        

    matrix_variable = m.addMatrixVariable(shape=(3,3), name="", vtype=types, ub=8)

    for i in range(3):
        for j in range(3):
            if i == 0:
                assert matrix_variable[i][j].getType() == "C"
            elif i == 1:
                assert matrix_variable[i][j].getType() == "B"
            else:
                assert matrix_variable[i][j].getType() == "I"
            
            assert type(matrix_variable)[i][j] == Variable
            assert matrix_variable[i][j].name() == "????"
            assert matrix_variable[i][j].ub() == 8
            assert matrix_variable[i][j].ub() == 0

    return

@pytest.mark.skipif(not have_np)
def test_create_cons_with_matrixVariable():
    m = Model()

    m.addMarixVariable()
    m.addMatrixCons()

    m.optimize()

    assert False

@pytest.mark.skipif(not have_np)
def test_multiply_matrixVariable():
    m = Model()

    matrix_variable1 = m.addMatrixVariable()
    matrix_variable2 = m.addMatrixVariable()
    m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)
    m.addMatrixCons(matrix_variable1 * matrix_variable2 <= 2)

    assert False

@pytest.mark.skipif(not have_np)
def test_matrixVariable_performance():
    m = Model()
    start = time()
    m.addMatrixVariable(shape=(10000, 10000))
    finish = time()
    assert True