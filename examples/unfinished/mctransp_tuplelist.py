"""
transp.py: a model for the multi-commodity transportation problem

Model for solving the multi-commodity transportation problem:
minimize the total transportation cost for satisfying demand at
customers, from capacitated facilities.

Use tuplelist for selecting arcs.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model

def mctransp(I,J,K,c,d,M):
    """mctransp -- model for solving the Multi-commodity Transportation Problem
    Parameters:
        - I: set of customers
        - J: set of facilities
        - K: set of commodities
        - c[i,j,k]: unit transportation cost on arc (i,j) for commodity k
        - d[i][k]: demand for commodity k at node i
        - M[j]: capacity
    Returns a model, ready to be solved.
    """

    model = Model("multi-commodity transportation")

    # Create variables
    x = {}
    for (i,j,k) in c:
        x[i,j,k] = model.addVar(vtype="C", name="x(%s,%s,%s)" % (i,j,k), obj=c[i,j,k])

    # tuplelist is a Gurobi data structure to manage lists of equal sized tuples - try itertools as alternative
    arcs = tuplelist([(i,j,k) for (i,j,k) in x])

    # Demand constraints
    for i in I:
        for k in K:
            model.addCons(sum(x[i,j,k] for (i,j,k) in arcs.select(i,"*",k)) == d[i,k], "Demand(%s,%s)" % (i,k))

    # Capacity constraints
    for j in J:
        model.addCons(sum(x[i,j,k] for (i,j,k) in arcs.select("*",j,"*")) <= M[j], "Capacity(%s)" % j)

    model.data = x
    return model


def make_inst1():
    d = {(1,1):80,   (1,2):85,   (1,3):300,  (1,4):6, # {customer: {commodity:demand <float>}}
         (2,1):270,  (2,2):160,  (2,3):400,  (2,4):7,
         (3,1):250,  (3,2):130,  (3,3):350,  (3,4):4,
         (4,1):160,  (4,2):60,   (4,3):200,  (4,4):3,
         (5,1):180,  (5,2):40,   (5,3):150,  (5,4):5
         }
    I = set([i for (i,k) in d])
    K = set([k for (i,k) in d])
    M = {1:3000, 2:3000, 3:3000}  # capacity
    J = M.keys()

    produce = {1:[2,4], 2:[1,2,3], 3:[2,3,4]}  # products that can be produced in each facility
    weight = {1:5, 2:2, 3:3, 4:4}              # {commodity: weight<float>}
    cost = {(1,1):4,  (1,2):6, (1,3):9,        # {(customer,factory): cost<float>}
            (2,1):5,  (2,2):4, (2,3):7,
            (3,1):6,  (3,2):3, (3,3):4,
            (4,1):8,  (4,2):5, (4,3):3,
            (5,1):10, (5,2):8, (5,3):4
            }
    c = {}

    for i in I:
        for j in J:
            for k in produce[j]:
                c[i,j,k] = cost[i,j] * weight[k]

    return I,J,K,c,d,M

def make_inst2():
    d = {(1,1):45,  # {customer: {commodity:demand <float>}}
         (2,1):20,
         (3,1):30,
         (4,1):30,
         }
    I = set([i for (i,k) in d])
    K = set([k for (i,k) in d])
    M = {1:35, 2:50, 3:40}  # {factory: capacity<float>}}
    J = M.keys()

    produce = {1:[1], 2:[1], 3:[1]}            # products that can be produced in each facility
    weight = {1:1}                             # {commodity: weight<float>}
    cost = {(1,1):8,    (1,2):9,    (1,3):14,  # {(customer,factory): cost<float>}
            (2,1):6,    (2,2):12,   (2,3):9 ,
            (3,1):10,   (3,2):13,   (3,3):16,
            (4,1):9,    (4,2):7,    (4,3):5 ,
            }
    c = {}

    for i in I:
        for j in J:
            for k in produce[j]:
                c[i,j,k] = cost[i,j] * weight[k]

    return I,J,K,c,d,M


if __name__ == "__main__":
    I,J,K,c,d,M = make_inst1();
    model = mctransp(I,J,K,c,d,M)
    model.writeProblem("transp.lp")
    model.optimize()

    print("Optimal value:",model.getObjVal())

    EPS = 1.e-6
    x = model.data

    for (i,j,k) in x:
        if model.getVal(x[i,j,k]) > EPS:
            print("sending %10s units of %3s from plant %3s to customer %3s" % (model.getVal(x[i,j,k]),k,j,i))
