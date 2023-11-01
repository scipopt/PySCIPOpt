from pyscipopt import Model, quickprod
from pyscipopt.scip import CONST
from operator import mul
import functools

def test_quickprod_model():
    m = Model("quickprod")
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    c = 2.3

    q = quickprod([x,y,z,c]) == 0.0
    s = functools.reduce(mul,[x,y,z,c],1) == 0.0

    assert(q.expr.terms == s.expr.terms)

def test_quickprod():
    empty = quickprod(1 for i in [])
    assert len(empty.terms) == 1
    assert CONST in empty.terms

def test_largequadratic():
    # inspired from performance issue on
    # http://stackoverflow.com/questions/38434300

    m = Model("dense_quadratic")
    dim = 20
    x = [m.addVar("x_%d" % i) for i in range(dim)]
    expr = quickprod((i+j+1)*x[i]*x[j]
                    for i in range(dim)
                    for j in range(dim))
    cons = expr <= 1.0
    #                              upper triangle,     diagonal
    assert cons.expr.degree() == 2*dim*dim
    m.addCons(cons)
    # TODO: what can we test beyond the lack of crashes?
