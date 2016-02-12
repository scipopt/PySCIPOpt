from pyscipopt import Model

def test_simplelp():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype='C', obj=1.0)
    y = s.addVar("y", vtype='C', obj=2.0)

    # add some constraint
    coeffs = {x: 1.0, y: 2.0}
    s.addCons(coeffs, 5.0)

    # solve problem
    s.optimize()

    # retrieving the best solution
    solution = s.getBestSol()

    # print solution
    assert round(s.getVal(x, solution)) == 5.0
    assert round(s.getVal(y, solution)) == 0.0

    s.free()


def test_nicelp():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)

    # solve problem
    s.optimize()

    # print solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0

    s.free()

if __name__ == "__main__":
    test_simple()
    test_nicelp()
