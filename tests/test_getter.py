from pyscipopt import Model
import pytest

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

@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("value", ["v","v0","v8"])
@pytest.mark.parametrize("attrname", ["name"])
def test_getterbyattribute(attrname, value, match, first):
    m = setModel()
    print m.getVarsByAttr(attrname, value, match, first)

@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("value", [
    1,
    "1",
    "B",
    "BINARY",
    "C",
    "c",
    "INTEGER",
    "NEGATIVE",
    True,
    False,
    "True",
    "False"
])
@pytest.mark.parametrize("attrname", [
    "vtype",
    "degree",
    "getCol",
    "getLPSol",
    "getLbGlobal",
    "getLbLocal",
    "getLbOriginal",
    "getObj",
    "getUbGlobal",
    "getUbLocal",
    "getUbOriginal",
])
def test_getterbyreturn(attrname, value, match, first):
    if match:
        valtype = type(value)
        if "degree" == attrname or "get" in attrname:
            pytest.xfail("match unsupported for %s method" % attrname)
        if "vtype" == attrname and valtype is not str:
            pytest.xfail("match unsupported for %s method" % attrname)
    m = setModel()
    print m.getVarsByReturn(attrname, value, match, first)
    
