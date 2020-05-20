import pytest

from pyscipopt import Model

def test_model():
    # create solver instance
    s = Model()

    # test parameter methods
    pric = s.getParam('lp/pricing')
    s.setParam('lp/pricing', 'q')
    assert 'q' == s.getParam('lp/pricing')
    s.setParam('lp/pricing', pric)
    s.setParam('visual/vbcfilename', 'vbcfile')
    assert 'vbcfile' == s.getParam('visual/vbcfilename')

    assert 'lp/pricing' in s.getParams()
    s.setParams({'visual/vbcfilename': '-'})
    assert '-' == s.getParam('visual/vbcfilename')

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
    assert c.isLinear()
    s.chgLhs(c, 5.0)
    s.chgRhs(c, 6.0)

    assert s.getLhs(c) == 5.0
    assert s.getRhs(c) == 6.0

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

    # check expression evaluations
    expr = x*x + 2*x*y + y*y
    expr2 = x + 1
    assert s.getVal(expr) == s.getSolVal(solution, expr)
    assert s.getVal(expr2) == s.getSolVal(solution, expr2)
    assert round(s.getVal(expr)) == 25.0
    assert round(s.getVal(expr2)) == 6.0

    s.writeProblem('model')
    s.writeProblem('model.lp')

    s.freeProb()
    s = Model()
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    c = s.addCons(x + 2 * y <= 1.0)
    s.setMaximize()

    s.delCons(c)

    s.optimize()

    assert s.getStatus() == 'unbounded'


def test_model_ptr():
    model1 = Model()
    ptr1 = model1.to_ptr(give_ownership=True)
    assert not model1._freescip

    model2 = Model.from_ptr(ptr1, take_ownership=True)
    assert model2._freescip
    assert model2.to_ptr(False) == ptr1

    with pytest.raises(ValueError):
        Model.from_ptr("some gibberish", take_ownership=False)


if __name__ == "__main__":
    test_model()
    test_model_ptr()
