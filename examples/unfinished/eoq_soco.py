"""
eoq_soco.py:  model to the multi-item economic ordering quantity problem.

Approach: use second-order cone optimization.

Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def eoq_soco(I,F,h,d,w,W):
    """eoq_soco --  multi-item capacitated economic ordering quantity model using soco
    Parameters:
        - I: set of items
        - F[i]: ordering cost for item i
        - h[i]: holding cost for item i
        - d[i]: demand for item i
        - w[i]: unit weight for item i
        - W: capacity (limit on order quantity)
    Returns a model, ready to be solved.
    """
    model = Model("EOQ model using SOCO")

    T,c = {},{}
    for i in I:
        T[i] = model.addVar(vtype="C", name="T(%s)"%i)  # cycle time for item i
        c[i] = model.addVar(vtype="C", name="c(%s)"%i)  # total cost for item i

    for i in I:
        model.addCons(F[i] <= c[i]*T[i])

    model.addCons(quicksum(w[i]*d[i]*T[i] for i in I) <= W)

    model.setObjective(quicksum(c[i] + h[i]*d[i]*T[i]*0.5 for i in I), "minimize")

    model.data = T,c
    return model



if __name__ == "__main__":
    # multiple item EOQ
    I,F,h,d,w = multidict(
        {1:[300,10,10,20],
         2:[300,10,30,40],
         3:[300,10,50,10]}
        )
    W = 2000
    model = eoq_soco(I,F,h,d,w,W)
    model.optimize()

    T,c = model.data
    EPS = 1.e-6
    print("%5s\t%8s\t%8s" % ("i","T[i]","c[i]"))
    for i in I:
        print("%5s\t%8g\t%8g" % (i,model.getVal(T[i]),model.getVal(c[i])))
    print
    print("Optimal value:", model.getObjVal())
