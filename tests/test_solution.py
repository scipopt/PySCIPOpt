from pyscipopt import Model

def test_solution_getbest():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x*x <= y)

    m.optimize()

    sol = m.getBestSol()
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0
    print(sol) # prints the solution in the transformed space

    m.freeTransform()
    sol = m.getBestSol()
    assert round(sol[x]) == 2.0
    assert round(sol[y]) == 4.0
    print(sol) # prints the solution in the original space

def test_solution_create():
    m = Model()

    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    m.addCons(x*x <= y)

    s = m.createSol()
    s[x] = 2.0
    s[y] = 4.0
    assert m.addSol(s, free=True)

if __name__ == "__main__":
    test_solution_getbest()
    test_solution_create()
