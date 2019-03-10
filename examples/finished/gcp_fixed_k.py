##@file gcp_fixed_k.py
#@brief solve the graph coloring problem with fixed-k model
"""
Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum, multidict

def gcp_fixed_k(V,E,K):
    """gcp_fixed_k -- model for minimizing number of bad edges in coloring a graph
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: number of colors to be used
    Returns a model, ready to be solved.
    """
    model = Model("gcp - fixed k")

    x,z = {},{}
    for i in V:
        for k in range(K):
            x[i,k] = model.addVar(vtype="B", name="x(%s,%s)"%(i,k))
    for (i,j) in E:
        z[i,j] = model.addVar(vtype="B", name="z(%s,%s)"%(i,j))

    for i in V:
        model.addCons(quicksum(x[i,k] for k in range(K)) == 1, "AssignColor(%s)" % i)

    for (i,j) in E:
        for k in range(K):
            model.addCons(x[i,k] + x[j,k] <= 1 + z[i,j], "BadEdge(%s,%s,%s)"%(i,j,k))

    model.setObjective(quicksum(z[i,j] for (i,j) in E), "minimize")

    model.data = x,z
    return model


def solve_gcp(V,E):
    """solve_gcp -- solve the graph coloring problem with bisection and fixed-k model
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns tuple with number of colors used, and dictionary mapping colors to vertices
    """
    LB = 0
    UB = len(V)
    color = {}
    while UB-LB > 1:
        K = int((UB+LB) / 2)
        gcp = gcp_fixed_k(V,E,K)
        # gcp.Params.OutputFlag = 0 # silent mode
        #gcp.Params.Cutoff = .1
        gcp.setObjlimit(0.1)
        gcp.optimize()
        status = gcp.getStatus()
        if status == "optimal":
            x,z = gcp.data
            for i in V:
                for k in range(K):
                    if gcp.getVal(x[i,k]) > 0.5:
                        color[i] = k
                        break
                # else:
                #     raise "undefined color for", i
            UB = K
        else:
            LB = K

    return UB,color


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
    V,E = make_data(75,.25)

    K,color = solve_gcp(V,E)
    print("minimum number of colors:",K)
    print("solution:",color)
