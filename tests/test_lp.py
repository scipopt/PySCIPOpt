from pyscipopt import LP

def test_lp():
    # create LP instance, minimizing by default
    myLP = LP()

    # create cols w/o coefficients, 0 objective coefficient and 0,\infty bounds
    myLP.addCols(2 * [[]])

    # create rows
    myLP.addRow(entries = [(0,1),(1,2)] ,lhs = 5)
    lhs, rhs = myLP.getSides()
    assert lhs[0] == 5.0
    assert rhs[0] == myLP.infinity()

    assert(myLP.ncols() == 2)
    myLP.chgObj(0, 1.0)
    myLP.chgObj(1, 4.0)

    solval = myLP.solve()

    assert round(5.0 == solval)
