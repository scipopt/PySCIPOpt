##@file tutorial/logical.py
#@brief Tutorial example on how to use AND/OR/XOR constraints.
"""
N.B.: standard SCIP XOR constraint works differently from AND/OR by design.
The constraint is set with a boolean rhs instead of an integer resultant.
cf. http://listserv.zib.de/pipermail/scip/2018-May/003392.html
A workaround to get the resultant as variable is here proposed.

Public Domain, WTFNMFPL Public Licence
"""
from pyscipopt import Model
from pyscipopt import quicksum

def _init():
    model = Model()
    model.hideOutput()
    x = model.addVar("x","B")
    y = model.addVar("y","B")
    z = model.addVar("z","B")
    return model, x, y, z

def _optimize(name, m):
    m.optimize()
    print("* %s constraint *" % name)
    objSet = bool(m.getObjective().terms.keys())
    print("* Is objective set? %s" % objSet)
    if objSet:
        print("* Sense: %s" % m.getObjectiveSense())
    status = m.getStatus()
    print("* Model status: %s" % status)
    if status == 'optimal':
        for v in m.getVars():
            if v.name != "n":
                print("%s: %d" % (v, round(m.getVal(v))))
    else:
        print("* No variable is printed if model status is not optimal")
    print("")

def and_constraint(v=1, sense="minimize"):
    """ AND constraint """
    assert v in [0,1], "v must be 0 or 1 instead of %s" % v.__repr__()
    model, x, y, z = _init()
    r = model.addVar("r", "B")
    model.addConsAnd([x,y,z], r)
    model.addCons(x==v)
    model.setObjective(r, sense=sense)
    _optimize("AND", model)


def or_constraint(v=0, sense="maximize"):
    """ OR constraint"""
    assert v in [0,1], "v must be 0 or 1 instead of %s" % v.__repr__()
    model, x, y, z = _init()
    r = model.addVar("r", "B")
    model.addConsOr([x,y,z], r)
    model.addCons(x==v)
    model.setObjective(r, sense=sense)
    _optimize("OR", model)

def xors_constraint(v=1):
    """ XOR (r as boolean) standard constraint"""
    assert v in [0,1], "v must be 0 or 1 instead of %s" % v.__repr__()
    model, x, y, z = _init()
    r = True
    model.addConsXor([x,y,z], r)
    model.addCons(x==v)
    _optimize("Standard XOR (as boolean)", model)

def xorc_constraint(v=0, sense="maximize"):
    """ XOR (r as variable) custom constraint"""
    assert v in [0,1], "v must be 0 or 1 instead of %s" % v.__repr__()
    model, x, y, z = _init()
    r = model.addVar("r", "B")
    n = model.addVar("n", "I") # auxiliary
    model.addCons(r+quicksum([x,y,z]) == 2*n)
    model.addCons(x==v)
    model.setObjective(r, sense=sense)
    _optimize("Custom XOR (as variable)", model)

if __name__ == "__main__":
    and_constraint()
    or_constraint()
    xors_constraint()
    xorc_constraint()

