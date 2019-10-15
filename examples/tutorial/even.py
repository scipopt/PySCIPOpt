##@file tutorial/even.py
#@brief Tutorial example to check whether values are even or odd
"""
Public Domain, WTFNMFPL Public Licence
"""
from pyscipopt import Model
from pprint import pformat as pfmt

example_values = [
               0,
               1,
             1.5,
    "helloworld", 
              20,
              25,
            -101,
            -15.,
             -10,
          -2**31,
     -int(2**31), 
       "2**31-1",
    int(2**31-1),
    int(2**63)-1 
]

verbose = False
#verbose = True # uncomment for additional info on variables!
sdic = {0: "even", 1: "odd"} # remainder to 2

def parity(number):
    """
    Prints if a value is even/odd/neither per each value in a example list

    This example is made for newcomers and motivated by:
    - modulus is unsupported for pyscipopt.scip.Variable and int
    - variables are non-integer by default
    Based on this: #172#issuecomment-394644046

    Args:
        number: value which parity is checked

    Returns:
        sval: 1 if number is odd, 0 if number is even, -1 if neither
    """
    sval = -1
    if verbose:
        print(80*"*")
    try:
        assert number == int(round(number))
        m = Model()
        m.hideOutput()

        # x and n are integer, s is binary
        # Irrespective to their type, variables are non-negative by default
        # since 0 is the default lb. To allow for negative values, give None
        # as lower bound.
        # (None means -infinity as lower bound and +infinity as upper bound)
        x = m.addVar("x", vtype="I", lb=None, ub=None) #ub=None is default
        n = m.addVar("n", vtype="I", lb=None)
        s = m.addVar("s", vtype="B")
        # CAVEAT: if number is negative, x's lower bound must be None
        # if x is set by default as non-negative and number is negative:
        #     there is no feasible solution (trivial) but the program
        #     does not highlight which constraints conflict.

        m.addCons(x==number)

        # minimize the difference between the number and twice a natural number
        m.addCons(s == x-2*n)
        m.setObjective(s)
        m.optimize()

        assert m.getStatus() == "optimal"
        boolmod = m.getVal(s) == m.getVal(x)%2
        if verbose:
            for v in m.getVars():
                print("%*s: %d" % (fmtlen, v,m.getVal(v)))
            print("%*d%%2 == %d?" % (fmtlen, m.getVal(x), m.getVal(s)))
            print("%*s" % (fmtlen, boolmod))

        xval = m.getVal(x)
        sval = m.getVal(s)
        sstr = sdic[sval]
        print("%*d is %s" % (fmtlen, xval, sstr))
    except (AssertionError, TypeError):
        print("%*s is neither even nor odd!" % (fmtlen, number.__repr__()))
    finally:
        if verbose:
            print(80*"*")
            print("")
    return sval

if __name__ == "__main__":
    """
    If positional arguments are given:
        the parity check is performed on each of them
    Else:
        the parity check is performed on each of the default example values
    """
    import sys
    from ast import literal_eval as leval
    try:
        # check parity for each positional arguments
        sys.argv[1]
        values = sys.argv[1:]
    except IndexError:
        # check parity for each default example value
        values = example_values
    # format lenght, cosmetics
    fmtlen = max([len(fmt) for fmt in pfmt(values,width=1).split('\n')])
    for value in values:
        try:
            n = leval(value)
        except (ValueError, SyntaxError): # for numbers or str w/ spaces
            n = value
        parity(n)
