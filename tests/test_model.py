import pytest
import os
import itertools

from pyscipopt import Model, SCIP_STAGE, SCIP_PARAMSETTING, quicksum
from helpers.utils import random_mip_1

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
    s.printProblem()

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
    for i in range(5,10):
        x[i] = model.addVar(lb = -i, ub = i, vtype="I")
    for i in range(10,15):
        x[i] = model.addVar(vtype="B")

    model.addConsIndicator(x[0] <= 4, x[10])
    
    model.setPresolve(0)
    model.hideOutput()
    model.optimize()
    var_dict = model.getVarDict()
    var_dict_transformed = model.getVarDict(transformed=True)
    assert len(var_dict) == model.getNVars(transformed=False)
    assert len(var_dict_transformed) == model.getNVars(transformed=True)

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
    
    assert m.getStage() == SCIP_STAGE.PROBLEM
    assert m.getStageName() == "PROBLEM" 

    m.optimize()

    assert m.getStage() == SCIP_STAGE.SOLVED
    assert m.getStageName() == "SOLVED"

def test_getObjective():
    m = Model()
    m.addVar(obj=2, name="x1")
    m.addVar(obj=3, name="x2")

    assert str(m.getObjective()) == "Expr({Term(x1): 2.0, Term(x2): 3.0})"
    
    
def test_getTreesizeEstimation():
    m = Model()

    assert m.getTreesizeEstimation() == -1

    x = m.addVar("x", vtype='B', obj=1.0)
    y = m.addVar("y", vtype='B', obj=2.0)
    c = m.addCons(x + y <= 10.0)
    m.setMaximize()

    m.optimize()

    assert m.getTreesizeEstimation() > 0

def test_setLogFile():
    m = Model()
    x = m.addVar("x", vtype="I")
    y = m.addVar("y", vtype="I")
    m.addCons(x + y == 1)
    m.setObjective(2*x+y)
    
    log_file_name = "test_setLogFile.log"
    m.setLogfile(log_file_name)
    assert os.path.exists(log_file_name)
    
    m.optimize()
    del m
    assert os.path.getsize(log_file_name) > 0
    os.remove(log_file_name)

def test_setLogFile_none():
    m = Model()
    x = m.addVar("x", vtype="I")
    y = m.addVar("y", vtype="I")
    m.addCons(x + y == 1)
    m.setObjective(2*x+y)
    
    log_file_name = "test_setLogfile_none.log"
    m.setLogfile(log_file_name)
    assert os.path.exists(log_file_name)
    
    m.setLogfile(None)
    m.optimize()
    del m
    assert os.path.getsize(log_file_name) == 0
    os.remove(log_file_name)
   
def test_locale():
    on_release = os.getenv('RELEASE') is not None
    if on_release:
        pytest.skip("Skip this test on release builds")

    import locale

    m = Model()
    m.addVar(lb=1.1)
    
    try:
        locale.setlocale(locale.LC_NUMERIC, "pt_PT")
        assert locale.str(1.1) == "1,1"
    
        m.writeProblem("model.cip")

        with open("model.cip") as file:
            assert "1,1" not in file.read()
            
        m.readProblem(os.path.join("tests", "data", "test_locale.cip"))

        locale.setlocale(locale.LC_NUMERIC,"")
    except Exception:
        pytest.skip("pt_PT locale was not found. It might need to be installed.")    

def test_version_external_codes():
     scip = Model()
     scip.printVersion()
     scip.printExternalCodeVersions()

def test_primal_dual_limit():

    def build_scip_model():
        scip = Model()
        # Make a basic minimum spanning hypertree problem
        # Let's construct a problem with 15 vertices and 40 hyperedges. The hyperedges are our variables.
        v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        e = {}
        for i in range(40):
            e[i] = scip.addVar(vtype='B', name='hyperedge_{}'.format(i))

        # Construct a dummy incident matrix
        A = [[1, 2, 3], [2, 3, 4, 5], [4, 9], [7, 8, 9], [0, 8, 9],
             [1, 6, 8], [0, 1, 2, 9], [0, 3, 5, 7, 8], [2, 3], [6, 9],
             [5, 8], [1, 9], [2, 7, 8, 9], [3, 8], [2, 4],
             [0, 1], [0, 1, 4], [2, 5], [1, 6, 7, 8], [1, 3, 4, 7, 9],
             [11, 14], [0, 2, 14], [2, 7, 8, 10], [0, 7, 10, 14], [1, 6, 11],
             [5, 8, 12], [3, 4, 14], [0, 12], [4, 8, 12], [4, 7, 9, 11, 14],
             [3, 12, 13], [2, 3, 4, 7, 11, 14], [0, 5, 10], [2, 7, 13], [4, 9, 14],
             [7, 8, 10], [10, 13], [3, 6, 11], [2, 8, 9, 11], [3, 13]]

        # Create a cost vector for each hyperedge
        c = [2.5, 2.9, 3.2, 7, 1.2, 0.5,
             8.6, 9, 6.7, 0.3, 4,
             0.9, 1.8, 6.7, 3, 2.1,
             1.8, 1.9, 0.5, 4.3, 5.6,
             3.8, 4.6, 4.1, 1.8, 2.5,
             3.2, 3.1, 0.5, 1.8, 9.2,
             2.5, 6.4, 2.1, 1.9, 2.7,
             1.6, 0.7, 8.2, 7.9, 3]

        # Add constraint that your hypertree touches all vertices
        scip.addCons(quicksum((len(A[i]) - 1) * e[i] for i in range(len(A))) == len(v) - 1)

        # Now add the sub-tour elimination constraints.
        for i in range(2, len(v) + 1):
            for combination in itertools.combinations(v, i):
                scip.addCons(
                    quicksum(max(len(set(combination) & set(A[j])) - 1, 0) * e[j] for j in range(len(A))) <= i - 1,
                    name='cons_{}'.format(combination))

        # Add objective to minimise the cost
        scip.setObjective(quicksum(c[i] * e[i] for i in range(len(A))), sense='minimize')
        return scip

    scip = build_scip_model()
    scip.setParam("limits/primal", 100)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setParam("branching/random/priority", 1000000)
    scip.optimize()
    assert(scip.getStatus() == "primallimit"), scip.getStatus()

    scip = build_scip_model()
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setParam("limits/dual", -10)
    scip.optimize()
    assert (scip.getStatus() == "duallimit"), scip.getStatus()

def test_getObjVal():
    m = Model()

    x = m.addVar(obj=0)
    y = m.addVar(obj = 1)
    z = m.addVar(obj = 2)

    m.addCons(x+y+z >= 0)
    m.addCons(y+z >= 3)
    m.addCons(z >= 8)

    m.setParam("limits/solutions", 0)
    m.optimize()
    
    try:
        m.getObjVal()
    except Warning:
        pass

    try:
        m.getVal(x)
    except Warning:
        pass

    m.freeTransform()
    m.setParam("limits/solutions", 1)
    m.presolve()

    assert m.getObjVal()
    assert m.getVal(x)

    m.freeTransform()
    m.setParam("limits/solutions", -1)

    m.optimize()

    assert m.getObjVal() == 16 
    assert m.getVal(x) == 0

    assert m.getObjVal() == 16 
    assert m.getVal(x) == 0

# tests writeProblem() after redirectOutput()
def test_redirection():

    # create problem instances
    original = random_mip_1(False, False, False, -1, True)
    redirect = Model()

    # redirect console output
    original.redirectOutput()

    # write problem instance
    original.writeProblem("redirection.lp")

    # solve original instance
    original.optimize()

    # read problem instance
    redirect.readProblem("redirection.lp")

    # remove problem file
    os.remove("redirection.lp")

    # compare problem dimensions
    assert redirect.getNVars(False) == original.getNVars(False)
    assert redirect.getNConss(False) == original.getNConss(False)

    # solve redirect instance
    redirect.optimize()

    # compare objective values
    assert original.isEQ(redirect.getObjVal(), original.getObjVal())

def test_comparisons():
    from math import inf
    model = Model()

    assert model.isPositive(1.)
    assert model.isNegative(-1.)

    assert not model.isPositive(0.)
    assert not model.isNegative(0.)

    assert model.isHugeValue(inf)
