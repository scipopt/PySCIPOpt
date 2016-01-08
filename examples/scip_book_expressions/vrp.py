"""
vrp.py:  solve the vehicle routing problem.

approach:
    - start with assignment model
    - add cuts until all components of the graph are connected

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
import networkx
from pyscipopt import Model, quicksum, multidict

def solve_vrp(V,c,m,q,Q):
    """solve_vrp -- solve the vehicle routing problem.
       - start with assignment model (depot has a special status)
       - add cuts until all components of the graph are connected
    Parameters:
        - V: set/list of nodes in the graph
        - c[i,j]: cost for traversing edge (i,j)
        - m: number of vehicles available
        - q[i]: demand for customer i
        - Q: vehicle capacity
    Returns the optimum objective value and the list of edges used.
    """

    def addcut(cut_edges):
        """addcut: add constraint to eliminate infeasible solutions
        Parameters:
            - cut_edges: list of edges in the current solution, except connections to depot
        Returns True if a cut was added, False otherwise
        """
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = networkx.connected_components(G)
        cut = False
        for S in Components:
            S_card = len(S)
            q_sum = sum(q[i] for i in S)
            NS = int(math.ceil(float(q_sum)/Q))
            S_edges = [(i,j) for i in S for j in S if i<j and (i,j) in cut_edges]
            if S_card >= 3 and (len(S_edges) >= S_card or NS > 1):
                add = model.addCons(quicksum(x[i,j] for i in S for j in S if j > i) <= S_card-NS)
                cut = True
        return cut

    model = Model("vrp")

    x = {}
    for i in V:
        for j in V:
            if j > i and i == V[0]:       # depot
                x[i,j] = model.addVar(ub=2, vtype="I", name="x(%s,%s)"%(i,j))
            elif j > i:
                x[i,j] = model.addVar(ub=1, vtype="I", name="x(%s,%s)"%(i,j))
    
    model.addCons(quicksum(x[V[0],j] for j in V[1:]) == 2*m, "DegreeDepot")
    for i in V[1:]:
        model.addCons(quicksum(x[j,i] for j in V if j < i) +
                        quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for i in V for j in V if j>i), "minimize")

    model.hideOutput()

    EPS = 1.e-6
    while True:
        model.optimize()
        edges = []
        for (i,j) in x:
            if model.getVal(x[i,j]) > EPS:
                if i != V[0] and j != V[0]:
                    edges.append((i,j))
        if addcut(edges) == False:
            break

    return model.getObjVal(),edges


def distance(x1,y1,x2,y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def make_data(n):
    """make_data: compute matrix distance based on euclidean distance"""
    V = range(1,n+1)
    x = dict([(i,random.random()) for i in V])
    y = dict([(i,random.random()) for i in V])
    c,q = {},{}
    Q = 100
    for i in V:
        q[i] = random.randint(10,20)
        for j in V:
            if j > i:
                c[i,j] = distance(x[i],y[i],x[j],y[j])
    return V,c,q,Q


if __name__ == "__main__":
    import sys

    n = 19
    m = 3
    seed = 1
    random.seed(seed)
    V,c,q,Q = make_data(n)
    z,edges = solve_vrp(V,c,m,q,Q)
    print("Optimal solution:",z)
    print("Edges in the solution:")
    print(sorted(edges))
