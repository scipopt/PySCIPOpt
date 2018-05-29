from pyscipopt import Model

import pytest

itertools = pytest.importorskip("itertools")
product = itertools.product


################################################################################
#
# Testing AND/OR constraints
#   check whether any error is raised
# see http://scip.zib.de/doc-5.0.1/html/cons__and_8c.php
#   for resultant and operators definition
# CAVEAT: ONLY binary variables are allowed
#   Integer and continous variables behave unexpectedly (due to SCIP?)
# TBI: automatic assertion of expected resultant VS optimal resultant
#   (visual inspection at the moment)
# TBI: implement and test XOR constraint
#
################################################################################

verbose = True

### AUXILIARY ###
def setModel(vtype="B", name=None, imax=2):
    if name is None: name = "model"
    m = Model(name)
    m.hideOutput()
    i = 0
    m.addVar("r", vtype)
    while i < imax:
        m.addVar("v%s" % i, vtype)
        i+=1
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
    i = 0
    while i < imax:
        vi = getVarByName(m,"v%s" % i)
        m.addCons(vi == val, vtype)
        i+=1
    return

def printOutput(m):
    status = m.getStatus()
    r = getVarByName(m,"r")
    rstr = "%d" % round(m.getVal(r))
    vs = getAllVarsByName(m, "v")
    vsstr = "".join(["%d" % round(m.getVal(v)) for v in vs])
    print("Status: %s, resultant: %s, operators: %s" % (status, rstr, vsstr))

### MAIN ###
def main_logical(model, logical, sense="min"):
    try:
        r = getVarByName(model, "r")
        vs = getAllVarsByName(model, "v")
        ### addConsAnd/Or method (Xor: TBI) ###
        method_name = "addCons%s" % logical.capitalize()
        try:
            _model_addConsLogical = getattr(model, method_name)
        except AttributeError as e:
            if method_name == "addConsXor":
                pytest.xfail("addCons%s has to be implemented" % method_name)
            else:
                raise AttributeError("addCons%s not implemented" % method_name)
        _model_addConsLogical(vs,r)
        model.setObjective(r, sense="%simize" % sense)
        model.optimize()
        assert model.getStatus() == "optimal"
        if verbose: printOutput(model)
        return True
    except Exception as e:
        if verbose: print("%s: %s" % (e.__class__.__name__, e))
        return False

### TEST ###
@pytest.mark.parametrize("nconss", [1, 2, "all"])
@pytest.mark.parametrize("vconss", [0, 1])
@pytest.mark.parametrize("sense", ["min","max"])
@pytest.mark.parametrize("logical", ["and", "or", "xor"]) #xor TBI
@pytest.mark.parametrize("noperators", [2,20,200])
@pytest.mark.parametrize("vtype", ["B","I","C"]) #I and C may raise errors: see preamble
def test_logical(noperators, vtype, logical, sense, vconss, nconss):
    if nconss == "all": nconss = noperators
    if vtype in ["I","C"]:
        pytest.skip("unsupported vtype: %s" % vtype)
    m = setModel(vtype, logical, noperators)
    setConss(m,vtype, vconss, nconss)
    success = main_logical(m, logical, sense)
    assert(success), "Status is not optimal!"
