
from pyscipopt import Model


def test_addPiecewiseLinearCons():
    m = Model()

    xpoints = [1, 3, 5]
    ypoints = [1, 2, 4]
    x = m.addVar(lb=xpoints[0], ub=xpoints[-1], obj=2)
    y = m.addVar(lb=-m.infinity(), obj=-3)
    m.addPiecewiseLinearCons(x, y, xpoints, ypoints)

    m.optimize()
    assert m.isEQ(m.getObjVal(), -2)
