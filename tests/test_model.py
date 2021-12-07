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

def test_solve_concurrent():
    s = Model()
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    c = s.addCons(x + y <= 10.0)
    s.setMaximize()
    s.solveConcurrent()
    assert s.getStatus() == 'optimal'
    assert s.getObjVal() == 20.0

def test_multiple_cons_simple():
    def assert_conss_eq(a, b):
        assert a.name == b.name
        assert a.isInitial() == b.isInitial()
        assert a.isSeparated() == b.isSeparated()
        assert a.isEnforced() == b.isEnforced()
        assert a.isChecked() == b.isChecked()
        assert a.isPropagated() == b.isPropagated()
        assert a.isLocal() == b.isLocal()
        assert a.isModifiable() == b.isModifiable()
        assert a.isDynamic() == b.isDynamic()
        assert a.isRemovable() == b.isRemovable()
        assert a.isStickingAtNode() == b.isStickingAtNode()

    s = Model()
    s_x = s.addVar("x", vtype = 'C', obj = 1.0)
    s_y = s.addVar("y", vtype = 'C', obj = 2.0)
    s_cons = s.addCons(s_x + 2 * s_y <= 1.0)

    m = Model()
    m_x = m.addVar("x", vtype = 'C', obj = 1.0)
    m_y = m.addVar("y", vtype = 'C', obj = 2.0)
    m_conss = m.addConss([m_x + 2 * m_y <= 1.0])

    assert len(m_conss) == 1
    assert_conss_eq(s_cons, m_conss[0])

    s.freeProb()
    m.freeProb()


def test_multiple_cons_names():
    m = Model()
    x = m.addVar("x", vtype = 'C', obj = 1.0)
    y = m.addVar("y", vtype = 'C', obj = 2.0)

    names = list("abcdef")
    conss = m.addConss([x + 2 * y <= 1 for i in range(len(names))], names)

    assert len(conss) == len(names)
    assert all([c.name == n for c, n in zip(conss, names)])

    m.freeProb()

    m = Model()
    x = m.addVar("x", vtype = 'C', obj = 1.0)
    y = m.addVar("y", vtype = 'C', obj = 2.0)

    name = "abcdef"
    conss = m.addConss([x + 2 * y <= 1 for i in range(5)], name)

    assert len(conss) == 5
    assert all([c.name.startswith(name + "_") for c in conss])


def test_multiple_cons_params():
    """Test if setting the remaining parameters works as expected"""
    def assert_conss_neq(a, b):
        assert a.isInitial() != b.isInitial()
        assert a.isSeparated() != b.isSeparated()
        assert a.isEnforced() != b.isEnforced()
        assert a.isChecked() != b.isChecked()
        assert a.isPropagated() != b.isPropagated()
        assert a.isModifiable() != b.isModifiable()
        assert a.isDynamic() != b.isDynamic()
        assert a.isRemovable() != b.isRemovable()
        assert a.isStickingAtNode() != b.isStickingAtNode()

    kwargs = dict(initial=True, separate=True,
                  enforce=True, check=True, propagate=True, local=False,
                  modifiable=False, dynamic=False, removable=False,
                  stickingatnode=False)

    m = Model()
    x = m.addVar("x", vtype = 'C', obj = 1.0)
    y = m.addVar("y", vtype = 'C', obj = 2.0)

    conss = m.addConss([x + 2 * y <= 1], **kwargs)
    conss += m.addConss([x + 2 * y <= 1], **{k: not v for k, v in kwargs.items()})

    assert_conss_neq(conss[0], conss[1])


def test_model_ptr():
    model1 = Model()
    ptr1 = model1.to_ptr(give_ownership=True)
    assert not model1._freescip

    model2 = Model.from_ptr(ptr1, take_ownership=True)
    assert model2._freescip
    assert model2 == model1

    with pytest.raises(ValueError):
        Model.from_ptr("some gibberish", take_ownership=False)


if __name__ == "__main__":
    test_model()
    test_solve_concurrent()
    test_multiple_cons_simple()
    test_multiple_cons_names()
    test_multiple_cons_params()
    test_model_ptr()
