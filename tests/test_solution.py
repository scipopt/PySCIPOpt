from pyscipopt import Model, SCIP_PARAMSETTING, quicksum, quickprod


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

def test_getSolTime():
    m = Model()
    m.setPresolve(SCIP_PARAMSETTING.OFF)

    x = {}
    for i in range(20):
        x[i] = m.addVar(ub=i)

    for i in range(1,6):
        m.addCons(quicksum(x[j] for j in range(20) if j%i==0) >= i)
        m.addCons(quickprod(x[j] for j in range(20) if j%i==0) <= i**3)
    
    m.setObjective(quicksum(x[i] for i in range(20)))
    m.optimize()
    for s in m.getSols():
        assert type(m.getSolTime(s)) == float
