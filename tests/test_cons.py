from pyscipopt import Model, quicksum
import random
import pytest


def test_getConsNVars():
    n_vars = random.randint(100, 1000)
    m = Model()
    x = {}
    for i in range(n_vars):
        x[i] = m.addVar("%i" % i)

    c = m.addCons(quicksum(x[i] for i in x) <= 10)
    assert m.getConsNVars(c) == n_vars

    m.optimize()
    assert m.getConsNVars(c) == n_vars


def test_getConsVars():
    n_vars = random.randint(100, 1000)
    m = Model()
    x = {}
    for i in range(n_vars):
        x[i] = m.addVar("%i" % i)

    c = m.addCons(quicksum(x[i] for i in x) <= 1)
    assert m.getConsVars(c) == [x[i] for i in x]

def test_getConsVals():
    n_vars = 100
    m = Model()
    x = {}
    for i in range(n_vars):
        x[i] = m.addVar("%i" % i, vtype="B")

    c1 = m.addCons(quicksum(x[i] for i in x) <= 1)
    c2 = m.addConsKnapsack([x[i] for i in x], [i for i in range(1, n_vars+1)], 10)
    vals1 = m.getConsVals(c1)
    vals2 = m.getConsVals(c2)

    assert len(vals1) == n_vars
    assert all(isinstance(v, float) for v in vals1)
    assert len(vals2) == n_vars
    assert all(isinstance(v, float) for v in vals2)
    assert m.getConsVals(c2) == [i for i in range(1, n_vars+1)]

def test_constraint_option_setting():
    m = Model()
    x = m.addVar()
    c = m.addCons(x >= 3)

    for option in [True, False]:
        m.setCheck(c, option)
        m.setEnforced(c, option)
        m.setRemovable(c, option)
        m.setInitial(c, option)

        assert c.isChecked() == option
        assert c.isEnforced() == option
        assert c.isRemovable() == option
        assert c.isInitial() == option


def test_cons_logical():
    m = Model()
    x1 = m.addVar(vtype="B")
    x2 = m.addVar(vtype="B")
    x3 = m.addVar(vtype="B")
    x4 = m.addVar(vtype="B")
    result1 = m.addVar(vtype="B")
    result2 = m.addVar(vtype="B")

    m.addCons(x3 == 1 - x1)
    m.addCons(x4 == 1 - x2)

    # result1 true
    m.addConsAnd([x1, x2], result1)
    m.addConsOr([x1, x2], result1)
    m.addConsXor([x1, x3], True)

    # result2 false
    m.addConsOr([x3, x4], result2)
    m.addConsAnd([x1, x3], result2)
    m.addConsXor([x1, x2], False)

    m.optimize()

    assert m.isEQ(m.getVal(result1), 1)
    assert m.isEQ(m.getVal(result2), 0)

def test_cons_and():
    m = Model()
    x1 = m.addVar(vtype="B")
    x2 = m.addVar(vtype="B")
    result = m.addVar(vtype="B")

    and_cons = m.addConsAnd([x1, x2], result)

    assert m.getNVarsAnd(and_cons) == 2
    assert m.getVarsAnd(and_cons) == [x1, x2]
    resultant_var = m.getResultantAnd(and_cons)
    assert resultant_var is result
    m.optimize()

    m.sortAndCons(and_cons)
    assert m.isAndConsSorted(and_cons)
    
def test_cons_logical_fail():
    m = Model()
    x1 = m.addVar(vtype="B")
    x2 = m.addVar(vtype="B")
    x3 = m.addVar(vtype="B")
    x4 = m.addVar(vtype="B")
    result1 = m.addVar(vtype="B")

    m.addCons(x3 == 1 - x1)
    m.addCons(x4 == 1 - x2)

    # result1 false
    with pytest.raises(TypeError):
        m.addConsOr([x1*x3, x2*x4], result1)

    m.optimize()

def test_SOScons():
    m = Model()
    x = {}
    for i in range(6):
        x[i] = m.addVar(vtype="B", obj=-i)

    c1 = m.addConsSOS1([x[0]], [1])
    c2 = m.addConsSOS2([x[1]], [1])

    m.addVarSOS1(c1, x[2], 1)
    m.addVarSOS2(c2, x[3], 1)

    m.appendVarSOS1(c1, x[4])
    m.appendVarSOS2(c2, x[5])

    m.optimize()

    assert m.isEQ(m.getVal(x[0]), 0)
    assert m.isEQ(m.getVal(x[1]), 0)
    assert m.isEQ(m.getVal(x[2]), 0)
    assert m.isEQ(m.getVal(x[3]), 1)
    assert m.isEQ(m.getVal(x[4]), 1)
    assert m.isEQ(m.getVal(x[5]), 1)
    assert c1.getConshdlrName() == "SOS1"
    assert c2.getConshdlrName() == "SOS2"


def test_cons_indicator():
    m = Model()
    x = m.addVar(lb=0, obj=1)
    binvar = m.addVar(vtype="B", lb=1)

    c1 = m.addConsIndicator(x >= 1, binvar) 

    assert c1.name == "c1"

    c2 = m.addCons(x <= 3)

    c3 = m.addConsIndicator(x >= 0, binvar)
    assert c3.name == "c4"

    # because addConsIndicator actually adds two constraints
    assert m.getNConss() == 5

    slack = m.getSlackVarIndicator(c1)

    lin_cons = m.getLinearConsIndicator(c1)

    m.optimize()

    assert m.getNConss(transformed=False) == 5
    assert m.isEQ(m.getVal(slack), 0)
    assert m.isEQ(m.getVal(binvar), 1)
    assert m.isEQ(m.getVal(x), 1)
    assert c1.getConshdlrName() == "indicator"

def test_cons_indicator_with_matrix_binvar():
    # test matrix variable binvar #1043
    m = Model()
    x = m.addVar(vtype="B")

    # test binvar with int
    with pytest.raises(TypeError):
        m.addConsIndicator(x <= 0, 1)

    # test binvar with (1, 1, 1) shape of matrix variable
    with pytest.raises(ValueError):
        m.addConsIndicator(x <= 0, m.addMatrixVar(((1, 1, 1)), vtype="B"))

    # test binvar with (2, 3) shape of matrix variable
    with pytest.raises(ValueError):
        m.addConsIndicator(x <= 0, m.addMatrixVar(((2, 3)), vtype="B"))

    # test binvar with (2, 1) shape of list of lists
    with pytest.raises(ValueError):
        m.addConsIndicator(x <= 0, [[m.addVar(vtype="B")], [m.addVar(vtype="B")]])

    # test binvar with requiring type and dimension
    binvar = m.addMatrixVar(1, vtype="B")
    m.addConsIndicator(x >= 1, binvar, activeone=True)
    m.addConsIndicator(x <= 0, binvar, activeone=False)

    m.setObjective(binvar.sum(), "maximize")
    m.optimize()

    assert m.isEQ(m.getVal(x), 1)

def test_cons_knapsack_with_matrix_vars():
    # test matrix variable vars #1043
    m = Model()
    vars = m.addMatrixVar(3, vtype="B")
    m.addConsKnapsack(vars, [1, 2, 3], 5)

    m.setObjective(vars.sum(), "maximize")
    m.optimize()

    assert (m.getVal(vars) == [0, 1, 1]).all()

@pytest.mark.xfail(
    reason="addConsIndicator doesn't behave as expected when binary variable is False. See Issue #717."
)
def test_cons_indicator_fail():
    m = Model()
    binvar = m.addVar(vtype="B")
    x = m.addVar(vtype="C", lb=1, ub=3)
    m.addConsIndicator(x <= 2, binvar)

    m.setObjective(x, "maximize")

    sol = m.createSol(None)
    m.setSolVal(sol, x, 3)
    m.setSolVal(sol, binvar, 0)
    assert m.checkSol(sol)  # solution should be feasible

def test_addConsCardinality():
    m = Model()
    x = {}
    for i in range(5):
        x[i] = m.addVar(ub=1, obj=-1)

    m.addConsCardinality([x[i] for i in range(5)], 3)
    m.optimize()

    assert m.isEQ(m.getVal(quicksum(x[i] for i in range(5))), 3)

def test_getOrigConss():
    m = Model()
    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    z = m.addVar("z", lb=0, ub=5, obj=2)
    m.addCons(x <= y + z)
    m.addCons(x <= z + 100)
    m.addCons(y >= -100)
    m.addCons(x + y <= 1000)
    m.addCons(2* x + 2 * y <= 1000)
    m.addCons(x + y + z <= 7)
    m.optimize()
    assert len(m.getConss(transformed=False)) == m.getNConss(transformed=False)
    assert m.getNConss(transformed=False) == 6
    assert m.getNConss(transformed=True) < m.getNConss(transformed=False)


def test_printCons():
    m = Model()
    x = m.addVar()
    y = m.addVar()
    c = m.addCons(x * y <= 5)

    m.printCons(c)


def test_addConsElemDisjunction():
    m = Model()
    x = m.addVar(vtype="c", lb=-10, ub=2)
    y = m.addVar(vtype="c", lb=-10, ub=5)
    o = m.addVar(vtype="c")

    m.addCons(o <= (x + y))
    disj_cons = m.addConsDisjunction([])
    c1 = m.createConsFromExpr(x <= 1)
    c2 = m.createConsFromExpr(x <= 0)
    c3 = m.createConsFromExpr(y <= 0)
    m.addConsElemDisjunction(disj_cons, c1)
    disj_cons = m.addConsElemDisjunction(disj_cons, c2)
    disj_cons = m.addConsElemDisjunction(disj_cons, c3)
    m.setObjective(o, "maximize")
    m.optimize()
    assert m.isEQ(m.getVal(x), 1)
    assert m.isEQ(m.getVal(y), 5)
    assert m.isEQ(m.getVal(o), 6)


def test_addConsDisjunction_expr_init():
    m = Model()
    x = m.addVar(vtype="c", lb=-10, ub=2)
    y = m.addVar(vtype="c", lb=-10, ub=5)
    o = m.addVar(vtype="c")

    m.addCons(o <= (x + y))
    m.addConsDisjunction([x <= 1, x <= 0, y <= 0])
    m.setObjective(o, "maximize")
    m.optimize()
    assert m.isEQ(m.getVal(x), 1)
    assert m.isEQ(m.getVal(y), 5)
    assert m.isEQ(m.getVal(o), 6)

def test_cons_knapsack():
    m = Model()
    x = m.addVar("x", vtype="B", obj=-1)
    y = m.addVar("y", vtype="B", obj=0)
    z = m.addVar("z", vtype="B", obj=2)
    
    knapsack_cons = m.addConsKnapsack([x,y], [4,2], 10)
    assert knapsack_cons.getConshdlrName() == "knapsack"
    assert knapsack_cons.isKnapsack()

    assert m.getConsNVars(knapsack_cons) == 2
    assert m.getConsVars(knapsack_cons) == [x, y]

    m.chgCapacityKnapsack(knapsack_cons, 5)

    assert m.getCapacityKnapsack(knapsack_cons) == 5
    assert m.getRhs(knapsack_cons) == 5
    assert m.getLhs(knapsack_cons) == -m.infinity()

    m.addCoefKnapsack(knapsack_cons, z, 3)
    weights = m.getWeightsKnapsack(knapsack_cons)
    assert weights["x"] == 4
    assert weights["y"] == 2
    assert weights["z"] == 3

    m.optimize()
    assert m.getDualsolKnapsack(knapsack_cons) == 0
    assert m.getDualfarkasKnapsack(knapsack_cons) == 0

def test_getValsLinear():
    m = Model()
    x = m.addVar("x", lb=0, ub=2, obj=-1)
    y = m.addVar("y", lb=0, ub=4, obj=0)
    z = m.addVar("z", lb=0, ub=5, obj=2)
    
    c1 = m.addCons(2*x + y <= 5)
    c2 = m.addCons(x + 4*z <= 5)
    assert m.getValsLinear(c1) == {'x': 2, 'y': 1}

    m.optimize() # just to check if constraint transformation matters

    assert m.getValsLinear(c2) == {'x': 1, 'z': 4}

@pytest.mark.skip(reason="TODO: test getRowLinear()")
def test_getRowLinear():
    assert True
