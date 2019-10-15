##@file finished/even.py
#@brief model to decide whether argument is even or odd



################################################################################
#
# EVEN OR ODD?
#
# If a positional argument is given:
#   prints if the argument is even/odd/neither
# else:
#   prints if a value is even/odd/neither per each value in a example list
#
# This example is made for newcomers and motivated by:
# - modulus is unsupported for pyscipopt.scip.Variable and int
# - variables are non-integer by default
# Based on this:
# https://github.com/SCIP-Interfaces/PySCIPOpt/issues/172#issuecomment-394644046
#
################################################################################

from pyscipopt import Model

verbose = False
sdic = {0:"even",1:"odd"}

def parity(number):
    try:
        assert number == int(round(number))
        m = Model()
        m.hideOutput()
        
        ### variables are non-negative by default since 0 is the default lb.
        ### To allow for negative values, give None as lower bound
        ### (None means -infinity as lower bound and +infinity as upper bound)
        x = m.addVar("x", vtype="I", lb=None, ub=None) #ub=None is default
        n = m.addVar("n", vtype="I", lb=None)
        s = m.addVar("s", vtype="B")

        ### CAVEAT: if number is negative, x's lb must be None
        ### if x is set by default as non-negative and number is negative:
        ###     there is no feasible solution (trivial) but the program
        ###     does not highlight which constraints conflict.
        m.addCons(x==number)

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
    except (AssertionError, TypeError):
        print("%s is neither even nor odd!" % number.__repr__())

if __name__ == "__main__":
    import sys
    from ast import literal_eval as leval
    example_values = [0, 1, 1.5, "hallo welt", 20, 25, -101, -15., -10, -int(2**31), int(2**31-1), int(2**63)-1]
    try:
        try:
            n = leval(sys.argv[1])
        except ValueError:
            n = sys.argv[1]
        parity(n)
    except IndexError:
        for n in example_values:
            parity(n)
