##@file tsp.py
#@brief solve the traveling salesman problem
"""
minimize the travel cost for visiting n customers exactly once
approach:
    - start with assignment model
    - add cuts until there are no sub-cycles
    - two cutting plane possibilities (called inside "solve_tsp"):
        - addcut: limit the number of edges in a connected component S to |S|-1
        - addcut2: require the number of edges between two connected component to be >= 2

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
import networkx
from pyscipopt import Model, quicksum

def solve_tsp(V,c):
    """solve_tsp -- solve the traveling salesman problem
       - start with assignment model
       - add cuts until there are no sub-cycles
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
    Returns the optimum objective value and the list of edges used.
    """

    def addcut(cut_edges):
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))
        if len(Components) == 1:
            return False
        model.freeTransform()
        for S in Components:
            model.addCons(quicksum(x[i,j] for i in S for j in S if j>i) <= len(S)-1)
            print("cut: len(%s) <= %s" % (S,len(S)-1))
        return True


    def addcut2(cut_edges):
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False
        model.freeTransform()
        for S in Components:
            T = set(V) - set(S)
            print("S:",S)
            print("T:",T)
            model.addCons(quicksum(x[i,j] for i in S for j in T if j>i) +
                          quicksum(x[i,j] for i in T for j in S if j>i) >= 2)
            print("cut: %s >= 2" % "+".join([("x[%s,%s]" % (i,j)) for i in S for j in T if j>i]))
        return True

    # main part of the solution process:
    model = Model("tsp")

    model.hideOutput() # silent/verbose mode
    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = model.addVar(ub=1, name="x(%s,%s)"%(i,j))

    for i in V:
        model.addCons(quicksum(x[j,i] for j in V if j < i) + \
                        quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for i in V for j in V if j > i), "minimize")

    EPS = 1.e-6
    isMIP = False
    while True:
        model.optimize()
        edges = []
        for (i,j) in x:
            if model.getVal(x[i,j]) > EPS:
                edges.append( (i,j) )

        if addcut(edges) == False:
            if isMIP:     # integer variables, components connected: solution found
                break
            model.freeTransform()
            for (i,j) in x:     # all components connected, switch to integer model
                model.chgVarType(x[i,j], "B")
                isMIP = True

    return model.getObjVal(),edges


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
        print("Using randomized example instead")
        n = 200
        seed = 1
        random.seed(seed)
        V,c = make_data(n)
    else:
        from read_tsplib import read_tsplib
        try:
            V,c,x,y = read_tsplib(sys.argv[1])
        except:
            print("Cannot read TSPLIB file",sys.argv[1])
            exit(1)

    obj,edges = solve_tsp(V,c)

    print("\nOptimal tour:",edges)
    print("Optimal cost:",obj)
