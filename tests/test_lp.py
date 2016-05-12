from pyscipopt import Model, LP

def test_lp():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype='C', obj=1.0)
    y = s.addVar("y", vtype='C', obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5.0)

    # solve problem
    s.optimize()

    solution = s.getBestSol()

    # print solution
    assert (s.getVal(x) == s.getSolVal(solution, x))
    assert (s.getVal(y) == s.getSolVal(solution, y))
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0


def test_lpi():
    # create LP instance
    myLP = LP()

    myLP.addRow(entries = [(0,1),(1,2)] ,lhs = 5)
    lhs, rhs = myLP.getSides()
    assert lhs[0] == 5.0
    assert rhs[0] == myLP.infinity()

    assert(myLP.ncols() == 2)
    myLP.chgObj(0, 1.0)
    myLP.chgObj(1, 2.0)

    solval = myLP.solve()
    # create solver instance

    s = Model()
    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)
    # add some constraint
    s.addCons(x + 2*y >= 5)
    # solve problem
    s.optimize()

    # check solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0

    assert round(s.getObjVal() == solval)

if __name__ == "__main__":
    test_lp()
    test_lpi()
