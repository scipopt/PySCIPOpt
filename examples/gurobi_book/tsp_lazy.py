"""
tsp.py:  solve the traveling salesman problem

minimize the travel cost for visiting n customers exactly once
approach:
    - start with assignment model
    - add cuts until there are no sub-cycles

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
import networkx
from gurobipy import *

def tsp(V,c):
    """tsp -- model for solving the traveling salesman problem with callbacks
       - start with assignment model
       - add cuts until there are no sub-cycles
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
    Returns the optimum objective value and the list of edges used.
    """

    EPS = 1.e-6
    def tsp_callback(model,where):
        if where != GRB.Callback.MIPSOL:
            return

        edges = []
        for (i,j) in x:
            if model.cbGetSolution(x[i,j]) > EPS:
                edges.append( (i,j) )

        G = networkx.Graph()
        G.add_edges_from(edges)
        Components = networkx.connected_components(G)

        if len(Components) == 1:
            return
        for S in Components:
            model.cbLazy(quicksum(x[i,j] for i in S for j in S if j>i) <= len(S)-1)
            # print "cut: len(%s) <= %s" % (S,len(S)-1)
        return


    model = Model("tsp")
    # model.Params.OutputFlag = 0 # silent/verbose mode
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    model.update()

    for i in V:
        model.addConstr(quicksum(x[j,i] for j in V if j < i) + \
                        quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for i in V for j in V if j > i), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model,tsp_callback


def distance(x1,y1,x2,y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def make_data(n):
    """make_data: compute matrix distance based on euclidean distance"""
    V = range(1,n+1)
    x = dict([(i,random.random()) for i in V])
    y = dict([(i,random.random()) for i in V])
    c = {}
    for i in V:
        for j in V:
            if j > i:
                c[i,j] = distance(x[i],y[i],x[j],y[j])
    return V,c

def solve_tsp(V,c):
    model,tsp_callback = tsp(V,c)
    model.params.DualReductions = 0
    model.optimize(tsp_callback)
    x = model.__data

    if model.status == GRB.Status.TIME_LIMIT:
        return None,None

    EPS = 1.e-6
    edges = []
    for (i,j) in x:
        if x[i,j].X > EPS:
            edges.append( (i,j) )
    return model.ObjVal,edges

if __name__ == "__main__":
    import sys

    # Parse argument
    if len(sys.argv) < 2:
        print "Usage: %s instance" % sys.argv[0]
        exit(1)
        # n = 200
        # seed = 1
        # random.seed(seed)
        # V,c = make_data(n)

    from read_tsplib import read_tsplib
    try:
        V,c,x,y = read_tsplib(sys.argv[1])
    except:
        print "Cannot read TSPLIB file",sys.argv[1]
        exit(1)

    obj,edges = solve_tsp(V,c)

    print
    print "Optimal tour:",edges
    print "Optimal cost:",obj
    print
