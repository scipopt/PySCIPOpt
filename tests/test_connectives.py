from pyscipopt import Model

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
    print "Status: %s, resultant: %s, operators: %s" % (status, rstr, vsstr)

### TEST ###
def test_connective(m, connective, sense="min"):
    try:
        r = getVarByName(m,"r")
        vs = getAllVarsByName(m, "v")
        ### addConsAnd/Or method (Xor: TBI) ###
        _m_method = getattr(m, "addCons%s" % connective.capitalize())
        _m_method(vs,r)
        m.setObjective(r, sense="%simize" % sense)
        m.optimize()
        printOutput(m)
        return True
    except Exception as e:
        print "%s: %s" % (e.__class__.__name__, e)
        return False

### MAIN ###
if __name__ == "__main__":
    from itertools import product
    lvtype = ["B"]#,"I","C"] #I and C may raise errors: see preamble
    lconnective = ["and", "or"]
    lsense = ["min","max"]
    lnoperators = [2,20,200]
    lvconss = [0, 1]
    lnconss = [1, 2, None]
    cases = list(product(lnoperators, lvtype, lconnective, lsense, lvconss, lnconss))
    for c in cases:
        noperators, vtype, connective, sense, vconss, nconss = c
        if nconss == None: nconss = noperators
        c = (noperators, vtype, connective, sense, vconss, nconss)
        m = setModel(vtype, connective, noperators)
        setConss(m,vtype, vconss, nconss)
        teststr = ', '.join(list(str(ic) for ic in c))
        #print "Test: %s" % teststr
        print "Test: %3d operators of vtype %s; %s-constraint and sense %s; %d as constraint for %3d operator/s" % c
        success = test_connective(m, connective, sense) 
        print "Is test successful? %s" % success
        print
