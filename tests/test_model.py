import pytest

from pyscipopt import Model, SCIP_STAGE

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
    s.presolve() # to test presolve method
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
    assert conss == m.getConss()
    assert m.getNConss() == 5


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

def test_addCoefLinear():
    m = Model()
    x = m.addVar(obj=1)
    y = m.addVar(obj=0)
    c = m.addCons(x >= 1)

    m.addCoefLinear(c, y, 1)

    m.optimize()
    assert m.getVal(x) == 0

def test_delCoefLinear():
    m = Model()
    x = m.addVar(obj=1)
    y = m.addVar(obj=0)
    c = m.addCons(x + y >= 1)

    m.delCoefLinear(c,y)

    m.optimize()
    assert m.getVal(x) == 1

def test_chgCoefLinear():
    m = Model()
    x = m.addVar(obj=10)
    y = m.addVar(obj=1)
    c = m.addCons(x + y >= 1)

    m.chgCoefLinear(c, y, 0.001)

    m.optimize()
    assert m.getObjVal() == 10

def test_model_ptr():
    model1 = Model()
    ptr1 = model1.to_ptr(give_ownership=True)
    assert not model1._freescip

    model2 = Model.from_ptr(ptr1, take_ownership=True)
    assert model2._freescip
    assert model2 == model1

    with pytest.raises(ValueError):
        Model.from_ptr("some gibberish", take_ownership=False)

def test_model_relax():
    model = Model()
    x = {}
    for i in range(5):
        x[i] = model.addVar(lb = -i, ub = i, vtype="C")
    for i in range(10,15):
        x[i] = model.addVar(lb = -i, ub = i, vtype="I")
    for i in range(20,25):
        x[i] = model.addVar(vtype="B")
    
    model.relax()
    for v in x.values():
        var_lb = v.getLbGlobal()
        var_ub = v.getUbGlobal()
        assert v.getLbGlobal() == var_lb
        assert v.getUbGlobal() == var_ub
        assert v.vtype() == "CONTINUOUS"

def test_getVarsDict():
    model = Model()
    x = {}
    for i in range(5):
        x[i] = model.addVar(lb = -i, ub = i, vtype="C")
    for i in range(10,15):
        x[i] = model.addVar(lb = -i, ub = i, vtype="I")
    for i in range(20,25):
        x[i] = model.addVar(vtype="B")
    
    model.hideOutput()
    model.optimize()
    var_dict = model.getVarDict()
    for v in x.values():
        assert v.name in var_dict
        assert model.getVal(v) == var_dict[v.name]

def test_objLim():
    m = Model()

    x = m.addVar(obj=1, lb=2)
    m.setObjlimit(1)

    m.optimize()
    assert m.getNLimSolsFound() == 0

    m = Model()
    x = m.addVar(obj=1, lb=2)

    m.setObjlimit(2)
    m.optimize()
    assert m.getNLimSolsFound() == 1
    
def test_getStage():
    m = Model() 

    assert m.getStage() == SCIP_STAGE.PROBLEM
    assert m.getStageName() == "PROBLEM"

    x = m.addVar()
    m.addCons(x >= 1)    
    
    print(m.getStage())
    assert m.getStage() == SCIP_STAGE.PROBLEM
    assert m.getStageName() == "PROBLEM" 

    m.optimize()

    print(m.getStage())
    assert m.getStage() == SCIP_STAGE.SOLVED
    assert m.getStageName() == "SOLVED"
