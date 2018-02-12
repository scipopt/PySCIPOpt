from pyscipopt import Model

def test_model():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)

    assert x.getObj() == 1.0
    assert y.getObj() == 2.0

    s.setObjective(4.0 * y + 10.5, clear = False)
    assert x.getObj() == 1.0
    assert y.getObj() == 4.0
    assert s.getObjoffset() == 10.5

    # add some constraint
    c = s.addCons(x + 2 * y >= 1.0)
    s.chgLhs(c, 5.0)
    s.chgRhs(c, 6.0)

    assert s.getLhs(c) == 5.0
    assert s.getRhs(c) == 6.0

    badsolution = s.createSol()
    s.setSolVal(badsolution, x, 2.0)
    s.setSolVal(badsolution, y, 2.0)
    assert s.getSlack(c, badsolution) == 0.0
    assert s.getSlack(c, badsolution, 'lhs') == 1.0
    assert s.getSlack(c, badsolution, 'rhs') == 0.0
    assert s.getActivity(c, badsolution) == 6.0
    s.freeSol(badsolution)

    # solve problem
    s.optimize()

    solution = s.getBestSol()

    # print solution
    assert (s.getVal(x) == s.getSolVal(solution, x))
    assert (s.getVal(y) == s.getSolVal(solution, y))
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0
    assert s.getSlack(c, solution) == 0.0
    assert s.getSlack(c, solution, 'lhs') == 0.0
    assert s.getSlack(c, solution, 'rhs') == 1.0
    assert s.getActivity(c, solution) == 5.0

    s.freeProb()
    s = Model()
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    c = s.addCons(x + 2 * y <= 1.0)
    s.setMaximize()

    s.delCons(c)

    s.optimize()

    assert s.getStatus() == 'unbounded'

if __name__ == "__main__":
    test_model()
