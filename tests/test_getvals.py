import pytest

from pyscipopt import Model

def setModel():
    m = Model()
    v0 = m.addVar("v0","B")
    v1 = m.addVar("v1","I")
    v2 = m.addVar("v2","C")
    v3 = m.addVar("v3","BINARY")
    v4 = m.addVar("v4","INTEGER")
    v5 = m.addVar("v5","CONTINUOUS")
    v6 = m.addVar("v6","B", None)
    v7 = m.addVar("v7","B", None, None)
    v8 = m.addVar("v8","B", None, 0.0)
    v9 = m.addVar()
    return m

def printAttrsOfVars(vs, attr):
    if hasattr(vs,"__iter__"):
        print([getattr(v,attr) for v in vs])
    elif vs:
        print(getattr(vs,attr))
    else:
        print(None)

def printReturnsOfVars(vs, attr):
    if hasattr(vs,"__iter__"):
        print([getattr(v,attr)() for v in vs])
    elif vs:
        print(getattr(vs,attr)())
    else:
        print(None)

@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("vtype", ["B", "C", "BINARY", "integer", "FRACTIONAL"])
def test_getbyvtype(vtype, first):
    m = setModel()
    vs = m.getVarsByVtype(vtype, first=first)
    printReturnsOfVars(vs, "vtype")

@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("degree", [0, 1, 2, "1"])
def test_getbydegree(degree, first):
    m = setModel()
    vs = m.getVarsByDegree(degree, first=first)
    printReturnsOfVars(vs, "degree")

@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("match", [True, False])
@pytest.mark.parametrize("name", ["v", "v6", "x", "x10"])
def test_getbyname(name, match, first):
    m = setModel()
    vs = m.getVarsByName(name, match=match, first=first)
    printAttrsOfVars(vs, "name")

@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("value", [
    1, "1", 0, "0"
    "B", "BINARY", "C", "c", "INTEGER", "NEGATIVE",
    True, False, "True", "False"
])
@pytest.mark.parametrize("attrname", [
    "vtype", "degree", "getCol", "getLPSol",
    "getLbGlobal", "getLbLocal", "getLbOriginal",
    "getObj", "getUbGlobal", "getUbLocal", "getUbOriginal"
])
def test_getbyreturn(attrname, value, match, first):
    if match:
        valtype = type(value)
        if "degree" == attrname or "get" in attrname:
            pytest.xfail("partial match unsupported for %s method" % attrname)
        if "vtype" == attrname and valtype is not str:
            pytest.xfail("partial match unsupported for %s method" % attrname)
    m = setModel()
    print m.getVarsByReturn(attrname, value, match, first)

@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("value", ["v","v0","v8"])
@pytest.mark.parametrize("attrname", ["name"])
def test_getbyattribute(attrname, value, match, first):
    m = setModel()
    vs = m.getVarsByAttr(attrname, value, match, first)
    printAttrsOfVars(vs, attrname)
