"""
mkp.py: model for the multi-constrained knapsack problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

def mkp(I,J,v,a,b):
    """mkp -- model for solving the multi-constrained knapsack
    Parameters:
        - I: set of dimensions
        - J: set of items
        - v[j]: value of item j
        - a[i,j]: weight of item j on dimension i
        - b[i]: capacity of knapsack on dimension i
    Returns a model, ready to be solved.
    """
    model = Model("mkp")
    x = {}
    for j in J:
        x[j] = model.addVar(vtype="B", name="x(%s)"%j)
    model.update()

    for i in I:
        model.addConstr(quicksum(a[i,j]*x[j] for j in J) <= b[i], "Capacity(%s)"%i)

    model.setObjective(quicksum(v[j]*x[j] for j in J), GRB.MAXIMIZE)

    model.update()
    return model


def example():
    J,v = multidict({1:16, 2:19, 3:23, 4:28})
    a = {(1,1):2,    (1,2):3,    (1,3):4,    (1,4):5,
         (2,1):3000, (2,2):3500, (2,3):5100, (2,4):7200,
         }
    I,b = multidict({1:7, 2:10000})
    return I,J,v,a,b


if __name__ == "__main__":
    I,J,v,a,b = example()
    model = mkp(I,J,v,a,b)
    model.optimize()
    print "Optimal value=",model.ObjVal

    EPS = 1.e-6
    for v in model.getVars():
        if v.X > EPS:
            print v.VarName,v.X
