import re
import pytest
from pyscipopt import Model, scip, SCIP_PARAMSETTING, quicksum, quickprod


def test_solution_getbest():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x * x <= y)

    m.optimize()

    sol = m.getBestSol()
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0
    print(sol)  # prints the solution in the transformed space

    m.freeTransform()
    sol = m.getBestSol()
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0
    print(sol)  # prints the solution in the original space


def test_solution_create():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x * x <= y)

    s = m.createSol()
    s[x] = 2.0
    s[y] = 5.0
    assert not m.checkSol(s)
    assert m.addSol(s, free=True)

    s1 = m.createSol()
    m.setSolVal(s1, x, 1.0)
    m.setSolVal(s1, y, 2.0)
    assert m.checkSol(s1)

    m.optimize()

    assert m.getSolObjVal(s1) == -1
    m.freeSol(s1)

def test_createOrigSol():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=1, ub=4, obj=1)
    z = m.addVar("z", lb=1, ub=5, obj=10)
    m.addCons(x * x <= y*z)
    m.presolve()

    s = m.createOrigSol()
    s[x] = 2.0
    s[y] = 5.0
    s[z] = 10.0
    assert not m.checkSol(s)
    assert m.addSol(s, free=True)

    s1 = m.createOrigSol()
    m.setSolVal(s1, x, 1.0)
    m.setSolVal(s1, y, 1.0)
    m.setSolVal(s1, z, 1.0)
    assert m.checkSol(s1)
    assert m.addSol(s1, free=False)

    m.optimize()

    assert m.getSolObjVal(s1) == 10.0
    m.freeSol(s1)


def test_solution_evaluation():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x * x <= y)

    m.optimize()

    sol = m.getBestSol()

    # Variable evaluation
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0

    # Expression evaluation
    expr = x * x + 2 * x * y + y * y
    expr2 = x + 1
    assert round(sol[expr]) == 36.0
    assert round(sol[expr2]) == 3.0

    # Check consistency with Models's getVal method
    assert m.isEQ(sol[x], m.getVal(x))
    assert m.isEQ(m.getSolVal(sol, x), m.getVal(x))
    assert m.isEQ(sol[y], m.getVal(y))
    assert m.isEQ(m.getSolVal(sol, y), m.getVal(y))
    assert m.isEQ(sol[expr], m.getVal(expr))
    assert m.isEQ(m.getSolVal(sol, expr), m.getVal(expr))
    assert m.isEQ(sol[expr2], m.getVal(expr2))
    assert m.isEQ(m.getSolVal(sol, expr2), m.getVal(expr2))


def test_getSolTime():
    m = Model()
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    x = {}
    for i in range(20):
        x[i] = m.addVar(ub=i)

    for i in range(1, 6):
        m.addCons(quicksum(x[j] for j in range(20) if j % i == 0) >= i)
        m.addCons(quickprod(x[j] for j in range(20) if j % i == 0) <= i**3)

    m.setObjective(quicksum(x[i] for i in range(20)))
    m.optimize()
    for s in m.getSols():
        assert m.getSolTime(s) >= 0


def test_hasPrimalRay():
    m = Model()
    x = m.addVar()
    m.setObjective(x, "maximize")
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    m.optimize()

    assert m.hasPrimalRay()

    m = Model()
    x = m.addVar(lb=0)  # for readability
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    m.optimize()

    assert not m.hasPrimalRay()


def test_getPrimalRayVal():
    m = Model()
    x = m.addVar()
    m.setObjective(x, "maximize")
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    m.hideOutput()
    m.optimize()

    assert m.getPrimalRayVal(x) == 1


def test_getPrimalRay():
    m = Model()
    x = m.addVar()
    y = m.addVar()
    m.setObjective(x, "maximize")
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    m.hideOutput()
    m.optimize()

    assert m.getPrimalRay() == [1, 0]


def test_create_solution():
    with pytest.raises(ValueError):
        scip.Solution()


def test_print_solution():
    m = Model()

    m.addVar(obj=1, name="x")
    m.optimize()

    solution_str = str(m.getBestSol())
    assert re.match(r"{'x': -?\d+\.?\d*}", solution_str) is not None


def test_getSols():
    m = Model()

    x = m.addVar()
    m.optimize()

    assert len(m.getSols()) >= 1
    assert any(m.isEQ(sol[x], 0.0) for sol in m.getSols())
