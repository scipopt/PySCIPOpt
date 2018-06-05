from pyscipopt import Model

################################################################################
# EVEN OR ODD?
#
# This example is made for newcomers and motivated by:
# - modulus is unsupported for pyscipopt.scip.Variable and int
# - negative integer numbers are not default
# Based on this:
# https://github.com/SCIP-Interfaces/PySCIPOpt/issues/172#issuecomment-394644046
#
################################################################################

verbose = False
sdic = {0:"even",1:"odd"}

def parity(number):
    assert number == int(round(number))
    m = Model()
    m.hideOutput()

    ### "None" means -infinity as lower bound
    ### N.B.: 0 is the default lower bound
    ### thus negative numbers are not allowed
    x = m.addVar("x","I", None) #relative integer variable
    n = m.addVar("n","I", None) #relative integer variable
    s = m.addVar("s","B")

    m.addCons(x==number) #negative integer

    m.addCons(s == x-2*n)
    m.setObjective(s)
    m.optimize()

    assert m.getStatus() == "optimal"
    if verbose:
        for v in m.getVars():
            print("%s %d" % (v,m.getVal(v)))
        print("%d%%2 == %d?" % (m.getVal(x), m.getVal(s)))
        print(m.getVal(s) == m.getVal(x)%2)

    xval = m.getVal(x)
    sval = m.getVal(s)
    sstr = sdic[sval]
    print("%d is %s" % (xval, sstr))

for n in [-int(2**31), -101, -15., -10, 0, 1, 1.5, 20, 25, int(2**31-1)]:
    try:
        parity(n)
    except AssertionError:
        print("%s is neither even nor odd!" % n)
