"""
tsp_flow.py: solve the traveling salesman problem using flow formulation

minimize the travel cost for visiting n customers exactly once
approach:
    - start with assignment model
    - check flow from a source to every other node;
       - if no flow, a sub-cycle has been found --> add cut
       - otherwise, the solution is optimal

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
#import networkx todo
from pyscipopt import Model, quicksum, multidict

EPS = 1.e-6

def maxflow(V,M,source,sink):
    """maxflow: maximize flow from source to sink, taking into account arc capacities M
    Parameters:
        - V: set of vertices
        - M[i,j]: dictionary or capacity for arcs (i,j)
        - source: flow origin
        - sink: flow target
    Returns a model, ready to be solved.
        """
    # create max-flow underlying model, on which to find cuts
    model = Model("maxflow")
    
    f = {} # flow variable
    for (i,j) in M:
        f[i,j] = model.addVar(lb=-M[i,j], ub=M[i,j], name="flow(%s,%s)"%(i,j))

    cons = {}
    for i in V:
        if i != source and i != sink:
            cons[i] = model.addCons(
                quicksum(f[i,j] for j in V if i<j and (i,j) in M) - \
                quicksum(f[j,i] for j in V if i>j and (j,i) in M) == 0,
                "FlowCons(%s)"%i)

    model.setObjective(quicksum(f[i,j] for (i,j) in M if i==source), "maximize")

    # model.write("tmp.lp")
    model.data = f,cons
    return model

def solve_tsp(V,c):
    """solve_tsp -- solve the traveling salesman problem
       - start with assignment model
       - check flow from a source to every other node;
          - if no flow, a sub-cycle has been found --> add cut
          - otherwise, the solution is optimal
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
    Returns the optimum objective value and the list of edges used.
    """

    def addcut(X):
        for sink in V[1:]:
            mflow = maxflow(V,X,V[0],sink)
            mflow.optimize()
            f,cons = mflow.data
            if mflow.ObjVal < 2-EPS:  # no flow to sink, can add cut
                break
        else:
            return False

        #add a cut/constraint
        CutA = set([V[0]])
        for i in cons:
            if cons[i].Pi <= -1+EPS:
                CutA.add(i)
        CutB = set(V) - CutA
        main.addCons(
            quicksum(x[i,j] for i in CutA for j in CutB if j>i) + \
            quicksum(x[j,i] for i in CutA for j in CutB if j<i) >= 2)
        print("mflow:",mflow.getObjVal(),"cut:",CutA,"+",CutB,">= 2")
        print("mflow:",mflow.getObjVal(),"cut:",[(i,j) for i in CutA for j in CutB if j>i],"+",[(j,i) for i in CutA for j in CutB if j<i],">= 2")
        return True

    def isMIP(x):
        for var in x:
            if var.vtype == "CONTINUOUS":
                return False
        return True

    # main part of the solution process:
    main = Model("tsp")
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = main.addVar(ub=1, vtype="C", name="x(%s,%s)"%(i,j))

    for i in V:
        main.addCons(quicksum(x[j,i] for j in V if j < i) + \
                       quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)"%i)

    main.setObjective(quicksum(c[i,j]*x[i,j] for i in V for j in V if j > i), "minimize")

    while True:
        main.optimize()
        z = main.getObjVal()
        X = {}
        for (i,j) in x:
            if main.getVal(x[i,j]) > EPS:
                X[i,j] = main.getVal(x[i,j])

        if addcut(X) == False:  # i.e., components are connected
            if isMIP():      # integer variables, components connected: solution found
                break
            for (i,j) in x:     # all components connected, switch to integer model
                main.chgVarType(x[i,j], "BINARY")

    # process solution
    edges = []
    for (i,j) in x:
        if main.getVal(x[i,j]) > EPS:
            edges.append((i,j))
    return main.getObjVal(),edges


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


if __name__ == "__main__":
    import sys

    # Parse argument
    if len(sys.argv) < 2:
        print("Usage: %s instance" % sys.argv[0])
        exit(1)
        # n = 200
        # seed = 1
        # random.seed(seed)
        # V,c = make_data(n)

    from read_tsplib import read_tsplib
    try:
        V,c,x,y = read_tsplib(sys.argv[1])
    except:
        print("Cannot read TSPLIB file",sys.argv[1])
        exit(1)

    obj,edges = solve_tsp(V,c)

    print
    print("Optimal tour:",edges)
    print("Optimal cost:",obj)
    print
