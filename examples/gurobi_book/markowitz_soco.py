"""
markowitz_soco.py:  simple markowitz model for portfolio optimization.

Approach: use second-order cone optimization.

Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
from gurobipy import *

def markowitz(I,sigma,r,alpha):
    """markowitz -- simple markowitz model for portfolio optimization.
    Parameters:
        - I: set of items
        - sigma[i]: standard deviation of item i
        - r[i]: revenue of item i
        - alpha: acceptance threshold
    Returns a model, ready to be solved.
    """

    model = Model("markowitz")
    x = {}
    for i in I:
        x[i] = model.addVar(vtype="C", name="x(%s)"%i)  # quantity of i to buy
    model.update()

    model.addConstr(quicksum(r[i]*x[i] for i in I) >= alpha)
    model.addConstr(quicksum(x[i] for i in I) == 1)

    model.setObjective(quicksum(sigma[i]**2 * x[i] * x[i] for i in I), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model




if __name__ == "__main__":
    # portfolio
    import math
    I,sigma,r = multidict(
        {1:[0.07,1.01],
         2:[0.09,1.05],
         3:[0.1,1.08],
         4:[0.2,1.10],
         5:[0.3,1.20]}
        )
    alpha = 1.05

    model = markowitz(I,sigma,r,alpha)
    model.optimize()

    x = model.__data
    EPS = 1.e-6
    print "%5s\t%8s" % ("i","x[i]")
    for i in I:
        print "%5s\t%8g" % (i,x[i].X)
    print "sum:",sum(x[i].X for i in I)
    print
    print "Obj:",model.ObjVal
