
from pyscipopt import Model
from pyscipopt.recipes.piecewise import add_piecewise_linear_cons

def test_add_piecewise_linear_cons():
    m = Model()

    xpoints = [1, 3, 5]
    ypoints = [1, 2, 4]
    x = m.addVar(lb=xpoints[0], ub=xpoints[-1], obj=2)
    y = m.addVar(lb=-m.infinity(), obj=-3)
    add_piecewise_linear_cons(m, x, y, xpoints, ypoints)

    m.optimize()
    assert m.isEQ(m.getObjVal(), -2)

def test_add_piecewise_linear_cons2():
    m = Model()

    xpoints = [1, 3, 5]
    ypoints = [1, 2, 4]
    x = m.addVar(lb=xpoints[0], ub=xpoints[-1], obj=2)
    y = m.addVar(lb=-m.infinity(), obj=-3)
    add_piecewise_linear_cons(m, x, y, xpoints, ypoints)
    
    m.setMaximize()

    m.optimize()
    assert m.isEQ(m.getObjVal(), 0)
