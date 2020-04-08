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
from pyscipopt import Model, Conshdlr, quicksum, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING, SCIP_PARAMSETTING

class TSPconshdlr(Conshdlr):

    def findSubtours(self, checkonly, sol):
        EPS = 1.e-6
        edges = []
        x = self.model.data
        for (i, j) in x:
            if self.model.getSolVal(sol, x[i, j]) > EPS:
                edges.append((i,j))

        G = networkx.Graph()
        G.add_edges_from(edges)
        Components = list(networkx.connected_components(G))

        if len(Components) == 1:
            return False
        elif checkonly:
            return True

        for S in Components:
            self.model.addCons(quicksum(x[i, j] for i in S for j in S if j > i) <= len(S) - 1)
            print("cut: len(%s) <= %s" % (S, len(S) - 1))

        return True

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        if self.findSubtours(checkonly = True, sol = solution):
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        if self.findSubtours(checkonly = False, sol = None):
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, nlockspos, nlocksneg):
        pass

def tsp(V,c):
    """tsp -- model for solving the traveling salesman problem with callbacks
       - start with assignment model
       - add cuts until there are no sub-cycles
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
    Returns the optimum objective value and the list of edges used.
    """
    model = Model("TSP_lazy")
    conshdlr = TSPconshdlr()

    x = {}
    for i in V:
        for j in V:
            if j > i:
                x[i,j] = model.addVar(vtype = "B",name = "x(%s,%s)" % (i,j))

    for i in V:
        model.addCons(quicksum(x[j, i] for j in V if j < i) +
                      quicksum(x[i, j] for j in V if j > i) == 2, "Degree(%s)" % i)

    model.setObjective(quicksum(c[i, j] * x[i, j] for i in V for j in V if j > i), "minimize")

    model.data = x
    return model, conshdlr

def distance(x1,y1,x2,y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def make_data(n):
    """make_data: compute matrix distance based on euclidean distance"""
    V = range(1,n + 1)
    x = dict([(i,random.random()) for i in V])
    y = dict([(i,random.random()) for i in V])
    c = {}
    for i in V:
        for j in V:
            if j > i:
                c[i,j] = distance(x[i],y[i],x[j],y[j])
    return V,c,x,y

def solve_tsp(V,c):
    model, conshdlr = tsp(V, c)
    model.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator",
                          sepapriority = -1, enfopriority = -1, chckpriority = -1, sepafreq = -1, propfreq = -1,
                          eagerfreq = -1, maxprerounds = 0, delaysepa = False, delayprop = False, needscons = False,
                          presoltiming = SCIP_PRESOLTIMING.FAST, proptiming = SCIP_PROPTIMING.BEFORELP)
    model.setBoolParam("misc/allowstrongdualreds", 0)
    model.writeProblem("tsp.cip")
    model.optimize()
    x = model.data

    EPS = 1.e-6
    edges = []
    for (i,j) in x:
        if model.getVal(x[i, j]) > EPS:
            edges.append((i, j))
    return model.getObjVal(), edges

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: %s instance" % sys.argv[0])
        print("using randomly generated TSP instance")
        n = 200
        seed = 1
        random.seed(seed)
        V,c,x,y = make_data(n)

    else:
        from read_tsplib import read_tsplib
        try:
            V, c, x, y = read_tsplib(sys.argv[1])
        except:
            print("Cannot read TSPLIB file", sys.argv[1])
            exit(1)

    obj, edges = solve_tsp(V, c)

    print("")
    print("Optimal tour:", edges)
    print("Optimal cost:", obj)
    print("")
