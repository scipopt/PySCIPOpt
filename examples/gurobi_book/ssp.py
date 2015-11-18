"""
ssp.py: model for the stable set problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *

def ssp(V,E):
    """ssp -- model for the stable set problem
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("ssp")
    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
    model.update()

    for (i,j) in E:
        model.addConstr(x[i] + x[j] <= 1, "Edge(%s,%s)"%(i,j))

    model.setObjective(quicksum(x[i] for i in V), GRB.MAXIMIZE)

    model.update()
    model.__data = x
    return model


import random
def make_data(n,prob):
    """make_data: prepare data for a random graph
    Parameters:
       - n: number of vertices
       - prob: probability of existence of an edge, for each pair of vertices
    Returns a tuple with a list of vertices and a list edges.
       """
    V = range(1,n+1)
    E = [(i,j) for i in V for j in V if i < j and random.random() < prob]
    return V,E


if __name__ == "__main__":
    random.seed(1)
    V,E = make_data(100,.5)

    model = ssp(V,E)
    model.optimize()
    print "Opt.value=",model.ObjVal
    x = model.__data
    print "maximum stable set:"
    print [i for i in V if x[i].X > 0.5]
