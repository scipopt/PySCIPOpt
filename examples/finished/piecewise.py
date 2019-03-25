##@file piecewise.py
#@brief several approaches for solving problems with piecewise linear functions.
"""
Approaches:
    - mult_selection: multiple selection model
    - convex_comb_sos: model with SOS2 constraints
    - convex_comb_dis: convex combination with binary variables (disaggregated model)
    - convex_comb_dis_log: convex combination with a logarithmic number of binary variables
    - convex_comb_agg: convex combination with binary variables (aggregated model)
    - convex_comb_agg_log: convex combination with a logarithmic number of binary variables

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict

def mult_selection(model,a,b):
    """mult_selection -- add piecewise relation with multiple selection formulation
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """

    K = len(a)-1
    w,z = {},{}
    for k in range(K):
        w[k] = model.addVar(lb=-model.infinity()) # do not name variables for avoiding clash
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity())

    for k in range(K):
        model.addCons(w[k] >= a[k]*z[k])
        model.addCons(w[k] <= a[k+1]*z[k])

    model.addCons(quicksum(z[k] for k in range(K)) == 1)
    model.addCons(X == quicksum(w[k] for k in range(K)))

    c = [float(b[k+1]-b[k])/(a[k+1]-a[k]) for k in range(K)]
    d = [b[k]-c[k]*a[k] for k in range(K)]
    model.addCons(Y == quicksum(d[k]*z[k] + c[k]*w[k] for k in range(K)))

    return X,Y,z


def convex_comb_sos(model,a,b):
    """convex_comb_sos -- add piecewise relation with gurobi's SOS constraints
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """
    K = len(a)-1
    z = {}
    for k in range(K+1):
        z[k] = model.addVar(lb=0, ub=1, vtype="C")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity(), vtype="C")

    model.addCons(X == quicksum(a[k]*z[k] for k in range(K+1)))
    model.addCons(Y == quicksum(b[k]*z[k] for k in range(K+1)))

    model.addCons(quicksum(z[k] for k in range(K+1)) == 1)
    model.addConsSOS2([z[k] for k in range(K+1)])

    return X,Y,z


def convex_comb_dis(model,a,b):
    """convex_comb_dis -- add piecewise relation with convex combination formulation
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """
    K = len(a)-1
    wL,wR,z = {},{},{}
    for k in range(K):
        wL[k] = model.addVar(lb=0, ub=1, vtype="C")
        wR[k] = model.addVar(lb=0, ub=1, vtype="C")
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity(), vtype="C")

    model.addCons(X == quicksum(a[k]*wL[k] + a[k+1]*wR[k] for k in range(K)))
    model.addCons(Y == quicksum(b[k]*wL[k] + b[k+1]*wR[k] for k in range(K)))
    for k in range(K):
        model.addCons(wL[k] + wR[k] == z[k])

    model.addCons(quicksum(z[k] for k in range(K)) == 1)

    return X,Y,z


def gray(i):
    """returns i^int(i/2)"""
    return i^(int(i/2))


def convex_comb_dis_log(model,a,b):
    """convex_comb_dis_log -- add piecewise relation with a logarithmic number of binary variables
    using the convex combination formulation.
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """
    K = len(a)-1
    G = int(math.ceil((math.log(K)/math.log(2))))     # number of required bits
    N = 1<<G                                          # number of required variables
    # print("K,G,N:",K,G,N
    wL,wR,z = {},{},{}
    for k in range(N):
        wL[k] = model.addVar(lb=0, ub=1, vtype="C")
        wR[k] = model.addVar(lb=0, ub=1, vtype="C")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity(), vtype="C")

    g = {}
    for j in range(G):
        g[j] = model.addVar(vtype="B")

    model.addCons(X == quicksum(a[k]*wL[k] + a[k+1]*wR[k] for k in range(K)))
    model.addCons(Y == quicksum(b[k]*wL[k] + b[k+1]*wR[k] for k in range(K)))
    model.addCons(quicksum(wL[k] + wR[k] for k in range(K)) == 1)

    # binary variables setup
    for j in range(G):
        ones = []
        zeros = []
        for k in range(K):
            if k & (1<<j):
                ones.append(k)
            else:
                zeros.append(k)
        model.addCons(quicksum(wL[k] + wR[k] for k in ones) <= g[j])
        model.addCons(quicksum(wL[k] + wR[k] for k in zeros) <= 1-g[j])

    return X,Y,wL,wR



def convex_comb_agg(model,a,b):
    """convex_comb_agg -- add piecewise relation convex combination formulation -- non-disaggregated.
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """
    K = len(a)-1
    w,z = {},{}
    for k in range(K+1):
        w[k] = model.addVar(lb=0, ub=1, vtype="C")
    for k in range(K):
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity(), vtype="C")

    model.addCons(X == quicksum(a[k]*w[k] for k in range(K+1)))
    model.addCons(Y == quicksum(b[k]*w[k] for k in range(K+1)))
    model.addCons(w[0] <= z[0])
    model.addCons(w[K] <= z[K-1])
    for k in range(1,K):
        model.addCons(w[k] <= z[k-1]+z[k])
    model.addCons(quicksum(w[k] for k in range(K+1)) == 1)
    model.addCons(quicksum(z[k] for k in range(K)) == 1)
    return X,Y,z



def convex_comb_agg_log(model,a,b):
    """convex_comb_agg_log -- add piecewise relation with a logarithmic number of binary variables
    using the convex combination formulation -- non-disaggregated.
    Parameters:
        - model: a model where to include the piecewise linear relation
        - a[k]: x-coordinate of the k-th point in the piecewise linear relation
        - b[k]: y-coordinate of the k-th point in the piecewise linear relation
    Returns the model with the piecewise linear relation on added variables X, Y, and z.
    """
    K = len(a)-1
    G = int(math.ceil((math.log(K)/math.log(2))))     # number of required bits
    w,g = {},{}
    for k in range(K+1):
        w[k] = model.addVar(lb=0, ub=1, vtype="C")
    for j in range(G):
        g[j] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-model.infinity(), vtype="C")

    model.addCons(X == quicksum(a[k]*w[k]  for k in range(K+1)))
    model.addCons(Y == quicksum(b[k]*w[k]  for k in range(K+1)))
    model.addCons(quicksum(w[k] for k in range(K+1)) == 1)

    # binary variables setup
    for j in range(G):
        zeros,ones = [0],[]
        # print(j,"\tinit zeros:",zeros,"ones:",ones
        for k in range(1,K+1):
            # print(j,k,"\t>zeros:",zeros,"ones:",ones
            if (1 & gray(k)>>j) == 1 and (1 & gray(k-1)>>j) == 1:
                ones.append(k)
            if (1 & gray(k)>>j) == 0 and (1 & gray(k-1)>>j) == 0:
                zeros.append(k)
            # print(j,k,"\tzeros>:",zeros,"ones:",ones

        # print(j,"\tzeros:",zeros,"ones:",ones
        model.addCons(quicksum(w[k] for k in ones) <= g[j])
        model.addCons(quicksum(w[k] for k in zeros) <= 1-g[j])

    return X,Y,w


if __name__ == "__main__":
    # random.seed(1)

    a = [ -10, 10, 15,  25, 30, 35, 40, 45, 50, 55, 60, 70]
    b = [ -20,-20, 15, -21,  0, 50, 18,  0, 15, 24, 10, 15]

    print("\n\n\npiecewise: multiple selection")
    model = Model("multiple selection")
    X,Y,z = mult_selection(model,a,b) # X,Y --> piecewise linear replacement of x,f(x) based on points a,b
    # model using X and Y (and possibly other variables)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))

    print("\n\n\npiecewise: disaggregated convex combination")
    model = Model("disaggregated convex combination")
    X,Y,z = convex_comb_dis(model,a,b)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))

    print("\n\n\npiecewise: disaggregated convex combination, logarithmic number of variables")
    model = Model("disaggregated convex combination (log)")
    X,Y,z = convex_comb_dis(model,a,b)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))

    print("\n\n\npiecewise: SOS2 constraint")
    model = Model("SOS2")
    X,Y,w = convex_comb_sos(model,a,b)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))

    print("\n\n\npiecewise: aggregated convex combination")
    model = Model("aggregated convex combination")
    X,Y,z = convex_comb_agg(model,a,b)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))

    print("\n\n\npiecewise: aggregated convex combination, logarithmic number of variables")
    model = Model("aggregated convex combination (log)")
    X,Y,w = convex_comb_agg_log(model,a,b)
    u = model.addVar(vtype="C", name="u")

    A = model.addCons(3*X + 4*Y <= 250, "A")
    B = model.addCons(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, "maximize")
    model.optimize()
    print("X:",model.getVal(X))
    print("Y:",model.getVal(Y))
    print("u:",model.getVal(u))
