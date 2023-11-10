from pyscipopt import Model, quicksum
from pyscipopt.scip import CONST

def test_quicksum_model():
    m = Model("quicksum")
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    c = 2.3

    q = quicksum([x,y,z,c]) == 0.0
    s =      sum([x,y,z,c]) == 0.0

    assert(q.expr.terms == s.expr.terms)

def test_quicksum():
    empty = quicksum(1 for i in [])
    assert len(empty.terms) == 1
    assert CONST in empty.terms

def test_largequadratic():
    # inspired from performance issue on
    # http://stackoverflow.com/questions/38434300

    m = Model("dense_quadratic")
    dim = 200
    x = [m.addVar("x_%d" % i) for i in range(dim)]
    expr = quicksum((i+j+1)*x[i]*x[j]
                    for i in range(dim)
                    for j in range(dim))
    cons = expr <= 1.0
    #                              upper triangle,     diagonal
    assert len(cons.expr.terms) == dim * (dim-1) / 2 + dim
    m.addCons(cons)
    # TODO: what can we test beyond the lack of crashes?
