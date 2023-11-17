from pyscipopt import Model
from pyscipopt import quicksum

try:
    import pytest
    itertools = pytest.importorskip("itertools")
except ImportError:
    import itertools
product = itertools.product


# Testing AND/OR/XOR constraints
#   check whether any error is raised
# see http://scip.zib.de/doc-5.0.1/html/cons__and_8c.php
#   for resultant and operators definition
# CAVEAT: ONLY binary variables are allowed
#   Integer and continuous variables behave unexpectedly (due to SCIP?)
# TBI: automatic assertion of expected resultant VS optimal resultant
#   (visual inspection at the moment)

verbose = True  # py.test ignores this


# AUXILIARY FUNCTIONS
def setModel(vtype="B", name=None, imax=2):
    """initialize model and its variables.
    imax (int): number of operators"""
    if name is None:
        name = "model"
    m = Model(name)
    m.hideOutput()
    i = 0
    r = m.addVar("r", vtype)
    while i < imax:
        m.addVar("v%s" % i, vtype)
        i += 1
    return m


def getVarByName(m, name):
    try:
        return [v for v in m.getVars() if name == v.name][0]
    except IndexError:
        return None
    

def getAllVarsByName(m, name):
    try:
        return [v for v in m.getVars() if name in v.name]
    except IndexError:
        return []


def setConss(m, vtype="B", val=0, imax=1):
    """set numeric constraints to the operators.
    val  (int): number to which the operators are constraint
    imax (int): number of operators affected by the constraint"""
    i = 0
    while i < imax:
        vi = getVarByName(m, "v%s" % i)
        m.addCons(vi == val, vtype)
        i += 1
    return


def printOutput(m):
    """print status and values of variables AFTER optimization."""
    status = m.getStatus()
    r = getVarByName(m, "r")
    rstr = "%d" % round(m.getVal(r))
    vs = getAllVarsByName(m, "v")
    vsstr = "".join(["%d" % round(m.getVal(v)) for v in vs])
    print("Status: %s, resultant: %s, operators: %s" % (status, rstr, vsstr))


# MAIN FUNCTIONS
def main_variable(model, logical, sense="min"):
    """r is the BINARY resultant variable
    v are BINARY operators
    cf. http://scip.zib.de/doc-5.0.1/html/cons__and_8h.php"""
    try:
        r = getVarByName(model, "r")
        vs = getAllVarsByName(model, "v")
        # addConsAnd/Or method (Xor: TBI, custom) ###
        method_name = "addCons%s" % logical.capitalize()
        if method_name == "addConsXor":
            n = model.addVar("n", "I")
            model.addCons(r+quicksum(vs) == 2*n)
        else:
            try:
                _model_addConsLogical = getattr(model, method_name)
                _model_addConsLogical(vs, r)
            except AttributeError as e:
                raise AttributeError("%s not implemented" % method_name)
        model.setObjective(r, sense="%simize" % sense)
        model.optimize()
        assert model.getStatus() == "optimal"
        if verbose:
            printOutput(model)
        return True
    except Exception as e:
        if verbose:
            print("%s: %s" % (e.__class__.__name__, e))
        return False


def main_boolean(model, logical, value=False):
    """r is the BOOLEAN rhs (NOT a variable!)
    v are BINARY operators
    cf. http://scip.zib.de/doc-5.0.1/html/cons__xor_8h.php"""
    try:
        r = value
        vs = getAllVarsByName(model, "v")
        # addConsXor method (And/Or: TBI) ###
        method_name = "addCons%s" % logical.capitalize()
        try:
            _model_addConsLogical = getattr(model, method_name)
            _model_addConsLogical(vs, r)
        except AttributeError as e:
            raise AttributeError("%s not implemented" % method_name)
        model.optimize()
        assert model.getStatus() == "optimal"
        if verbose:
            printOutput(model)
        return True
    except Exception as e:
        if verbose:
            print("%s: %s" % (e.__class__.__name__, e))
        return False


# TEST FUNCTIONS
@pytest.mark.parametrize("nconss", [1, 2, "all"])
@pytest.mark.parametrize("vconss", [0, 1])
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("logical", ["and", "or", "xor"])
@pytest.mark.parametrize("noperators", [2, 20, 51, 100])
@pytest.mark.parametrize("vtype", ["B"])
def test_variable(noperators, vtype, logical, sense, vconss, nconss):
    if nconss == "all":
        nconss = noperators
    if vtype in ["I", "C"]:
        pytest.skip("unsupported vtype \"%s\" may raise errors or unexpected results" % vtype)
    m = setModel(vtype, logical, noperators)
    setConss(m, vtype, vconss, nconss)
    success = main_variable(m, logical, sense)
    assert(success), "Status is not optimal"


@pytest.mark.parametrize("nconss", [1, 2, "all"])
@pytest.mark.parametrize("vconss", [0, 1])
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("logical", ["xor", "and", "or"])
@pytest.mark.parametrize("noperators", [2, 20, 51, 100])
@pytest.mark.parametrize("vtype", ["B"])
def test_boolean(noperators, vtype, logical, value, vconss, nconss):
    if nconss == "all":
        nconss = noperators
    if vtype in ["I", "C"]:
        pytest.skip("unsupported vtype \"%s\" may raise errors or unexpected results" % vtype)
    if logical in ["and", "or"]:
        pytest.skip("unsupported logical: %s" % vtype)
    if logical == "xor" and nconss == noperators and noperators % 2 & vconss != value:
        pytest.xfail("addConsXor cannot be %s if an %s number of variables are all constraint to %s" % (value, noperators, vconss))
    m = setModel(vtype, logical, noperators)
    setConss(m, vtype, vconss, nconss)
    success = main_boolean(m, logical, value)
    assert(success), "Test is not successful"
