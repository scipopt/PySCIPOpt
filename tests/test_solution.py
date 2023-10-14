from pyscipopt import Model, scip
import pytest

def test_solution_getbest():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x*x <= y)

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
    m.addCons(x*x <= y)

    s = m.createSol()
    s[x] = 2.0
    s[y] = 4.0
    assert m.addSol(s, free=True)


def test_solution_evaluation():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x*x <= y)

    m.optimize()

    sol = m.getBestSol()

    # Variable evaluation
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0

    # Expression evaluation
    expr = x*x + 2*x*y + y*y
    expr2 = x + 1
    assert round(sol[expr]) == 36.0
    assert round(sol[expr2]) == 3.0

    # Check consistency with Models's getVal method
    assert sol[x] == m.getVal(x)
    assert sol[y] == m.getVal(y)
    assert sol[expr] == m.getVal(expr)
    assert sol[expr2] == m.getVal(expr2)


def test_create_solution():
    with pytest.raises(ValueError):
        scip.Solution()

def test_print_soltion():
    m = Model()

    m.addVar()
    m.optimize()

    assert str(m.getBestSol()) == "{'x1': -0.0}"


def test_getSols():
    m = Model()

    x = m.addVar()
    m.optimize()

    assert len(m.getSols()) >= 1
    assert any(sol[x] == 0.0 for sol in m.getSols())

if __name__ == "__main__":
    test_solution_getbest()
    test_solution_create()
