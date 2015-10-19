"""
piecewise.py:  several approaches for solving problems with piecewise linear functions.

Approaches:
    - mult_selection: multiple selection model
    - convex_comb_sos: model with SOS2 constraints
    - convex_comb_dis: convex combination with binary variables (disaggregated model)
    - convex_comb_dis_log: convex combination with a logarithmic number of binary variables
    - convex_comb_agg: convex combination with binary variables (aggregated model)
    - convex_comb_agg_log: convex combination with a logarithmic number of binary variables

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *
import math
import random

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
        w[k] = model.addVar(lb=-GRB.INFINITY) # do not name variables for avoiding clash
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY)
    model.update()

    for k in range(K):
        model.addConstr(w[k] >= a[k]*z[k])
        model.addConstr(w[k] <= a[k+1]*z[k])

    model.addConstr(quicksum(z[k] for k in range(K)) == 1)
    model.addConstr(X == quicksum(w[k] for k in range(K)))

    c = [float(b[k+1]-b[k])/(a[k+1]-a[k]) for k in range(K)]
    d = [b[k]-c[k]*a[k] for k in range(K)]
    model.addConstr(Y == quicksum(d[k]*z[k] + c[k]*w[k] for k in range(K)))

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
        z[k] = model.addVar(lb=0, ub=1, vtype="C") # do not name variables for avoiding clash
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C")
    model.update()

    model.addConstr(X == quicksum(a[k]*z[k] for k in range(K+1)))
    model.addConstr(Y == quicksum(b[k]*z[k] for k in range(K+1)))

    model.addConstr(quicksum(z[k] for k in range(K+1)) == 1)
    model.addSOS(GRB.SOS_TYPE2, [z[k] for k in range(K+1)])

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
        wL[k] = model.addVar(lb=0, ub=1, vtype="C") # do not name variables for avoiding clash
        wR[k] = model.addVar(lb=0, ub=1, vtype="C")
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C")
    model.update()

    model.addConstr(X == quicksum(a[k]*wL[k] + a[k+1]*wR[k] for k in range(K)))
    model.addConstr(Y == quicksum(b[k]*wL[k] + b[k+1]*wR[k] for k in range(K)))
    for k in range(K):
        model.addConstr(wL[k] + wR[k] == z[k])

    model.addConstr(quicksum(z[k] for k in range(K)) == 1)

    return X,Y,z



def gray(i):
    return i^i/2



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
    # print "K,G,N:",K,G,N
    wL,wR,z = {},{},{}
    for k in range(N):
        wL[k] = model.addVar(lb=0, ub=1, vtype="C") # do not name variables for avoiding clash
        wR[k] = model.addVar(lb=0, ub=1, vtype="C")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C")

    g = {}
    for j in range(G):
        g[j] = model.addVar(vtype="B")
    model.update()

    model.addConstr(X == quicksum(a[k]*wL[k] + a[k+1]*wR[k] for k in range(K)))
    model.addConstr(Y == quicksum(b[k]*wL[k] + b[k+1]*wR[k] for k in range(K)))
    model.addConstr(quicksum(wL[k] + wR[k] for k in range(K)) == 1)

    # binary variables setup
    for j in range(G):
        ones = []
        zeros = []
        for k in range(K):
            if k & (1<<j):
                ones.append(k)
            else:
                zeros.append(k)
        model.addConstr(quicksum(wL[k] + wR[k] for k in ones) <= g[j])
        model.addConstr(quicksum(wL[k] + wR[k] for k in zeros) <= 1-g[j])

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
        w[k] = model.addVar(lb=0, ub=1, vtype="C") # do not name variables for avoiding clash
    for k in range(K):
        z[k] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C")
    model.update()

    model.addConstr(X == quicksum(a[k]*w[k] for k in range(K+1)))
    model.addConstr(Y == quicksum(b[k]*w[k] for k in range(K+1)))
    model.addConstr(w[0] <= z[0])
    model.addConstr(w[K] <= z[K-1])
    for k in range(1,K):
        model.addConstr(w[k] <= z[k-1]+z[k])
    model.addConstr(quicksum(w[k] for k in range(K+1)) == 1)
    model.addConstr(quicksum(z[k] for k in range(K)) == 1)
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
        w[k] = model.addVar(lb=0, ub=1, vtype="C") # do not name variables for avoiding clash
    for j in range(G):
        g[j] = model.addVar(vtype="B")
    X = model.addVar(lb=a[0], ub=a[K], vtype="C")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C")
    model.update()

    model.addConstr(X == quicksum(a[k]*w[k]  for k in range(K+1)))
    model.addConstr(Y == quicksum(b[k]*w[k]  for k in range(K+1)))
    model.addConstr(quicksum(w[k] for k in range(K+1)) == 1)

    # binary variables setup
    for j in range(G):
        zeros,ones = [0],[]
        # print j,"\tinit zeros:",zeros,"ones:",ones
        for k in range(1,K+1):
            # print j,k,"\t>zeros:",zeros,"ones:",ones
            if (1 & gray(k)>>j) == 1 and (1 & gray(k-1)>>j) == 1:
                ones.append(k)
            if (1 & gray(k)>>j) == 0 and (1 & gray(k-1)>>j) == 0:
                zeros.append(k)
            # print j,k,"\tzeros>:",zeros,"ones:",ones

        # print j,"\tzeros:",zeros,"ones:",ones
        model.addConstr(quicksum(w[k] for k in ones) <= g[j])
        model.addConstr(quicksum(w[k] for k in zeros) <= 1-g[j])

    return X,Y,w


if __name__ == "__main__":
    # random.seed(1)

    a = [ -10, 10, 15,  25, 30, 35, 40, 45, 50, 55, 60, 70]
    b = [ -20,-20, 15, -21,  0, 50, 18,  0, 15, 24, 10, 15]

    print "\n\n\npiecewise: multiple selection"
    model = Model("multiple selection")
    X,Y,z = mult_selection(model,a,b) # X,Y --> piecewise linear replacement of x,f(x) based on points a,b
    # model using X and Y (and possibly other variables)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X

    print "\n\n\npiecewise: disaggregated convex combination"
    model = Model("disaggregated convex combination")
    X,Y,z = convex_comb_dis(model,a,b)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X

    print "\n\n\npiecewise: disaggregated convex combination, logarithmic number of variables"
    model = Model("disaggregated convex combination (log)")
    X,Y,z = convex_comb_dis(model,a,b)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X

    print "\n\n\npiecewise: SOS2 constraint"
    model = Model("SOS2")
    X,Y,w = convex_comb_sos(model,a,b)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X

    print "\n\n\npiecewise: aggregated convex combination"
    model = Model("aggregated convex combination")
    X,Y,z = convex_comb_agg(model,a,b)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X

    print "\n\n\npiecewise: aggregated convex combination, logarithmic number of variables"
    model = Model("aggregated convex combination (log)")
    X,Y,w = convex_comb_agg_log(model,a,b)
    u = model.addVar(vtype="C", name="u")
    model.update()
    A = model.addConstr(3*X + 4*Y <= 250, "A")
    B = model.addConstr(7*X - 2*Y + 3*u == 170, "B")
    model.setObjective(2*X + 15*Y + 5*u, GRB.MAXIMIZE)
    model.optimize()
    print "X:",X.X
    print "Y:",Y.X
    print "u:",u.X
