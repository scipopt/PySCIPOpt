"""
transp.py: a model for the transportation problem

Model for solving a transportation problem:
minimize the total transportation cost for satisfying demand at
customers, from capacitated facilities.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

def transp(I,J,c,d,M):
    """transp -- model for solving the transportation problem
    Parameters:
        I - set of customers
        J - set of facilities
        c[i,j] - unit transportation cost on arc (i,j)
        d[i] - demand at node i
        M[j] - capacity
    Returns a model, ready to be solved.
    """

    model = Model("transportation")

    # Create variables
    x = {}
    for i in I:
        for j in J:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)" % (i, j))
    model.update()

    # Demand constraints
    for i in I:
        model.addConstr(quicksum(x[i,j] for j in J if (i,j) in x) == d[i], name="Demand(%s)" % i)

    # Capacity constraints
    for j in J:
        model.addConstr(quicksum(x[i,j] for i in I if (i,j) in x) <= M[j], name="Capacity(%s)" % j)

    # Objective
    model.setObjective(quicksum(c[i,j]*x[i,j]  for (i,j) in x), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def make_inst1():
    I,d = multidict({1:80, 2:270, 3:250 , 4:160, 5:180}) # demand
    J,M = multidict({1:500, 2:500, 3:500})               # capacity
    c = {(1,1):4,    (1,2):6,    (1,3):9,  # cost
         (2,1):5,    (2,2):4,    (2,3):7,
         (3,1):6,    (3,2):3,    (3,3):4,
         (4,1):8,    (4,2):5,    (4,3):3,
         (5,1):10,   (5,2):8,    (5,3):4,
         }
    return I,J,c,d,M


def make_inst2():
    I,d = multidict({1:45, 2:20, 3:30 , 4:30}) # demand
    J,M = multidict({1:35, 2:50, 3:40})        # capacity
    c = {(1,1):8,    (1,2):9,    (1,3):14  ,   # {(customer,factory) : cost<float>}
         (2,1):6,    (2,2):12,   (2,3):9   ,
         (3,1):10,   (3,2):13,   (3,3):16  ,
         (4,1):9,    (4,2):7,    (4,3):5   ,
         }
    return I,J,c,d,M


if __name__ == "__main__":
    I,J,c,d,M = make_inst1();
    # I,J,c,d,M = make_inst2();
    model = transp(I,J,c,d,M)
    # model.write("transp.lp")
    model.optimize()
    print "Optimal value:",model.ObjVal

    EPS = 1.e-6
    x = model.__data
    for (i,j) in x:
        if x[i,j].X > EPS:
            print "sending quantity %10s from factory %3s to customer %3s" % (x[i,j].X,j,i)
