"""
vrp.py:  model for the vehicle routing problem using callback for adding cuts.

approach:
    - start with assignment model
    - add cuts until all components of the graph are connected

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
import networkx
from pyscipopt import Model, Conshdlr, quicksum, multidict, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING

class VRPconshdlr(Conshdlr):

    def addCuts(self, checkonly):
        """add cuts if necessary and return whether model is feasible"""
        cutsadded = False
        edges = []
        x = self.model.data
        for (i, j) in x:
            if self.model.getVal(x[i, j]) > .5:
                if i != V[0] and j != V[0]:
                    edges.append((i, j))
        G = networkx.Graph()
        G.add_edges_from(edges)
        Components = list(networkx.connected_components(G))
        for S in Components:
            S_card = len(S)
            q_sum = sum(q[i] for i in S)
            NS = int(math.ceil(float(q_sum) / Q))
            S_edges = [(i, j) for i in S for j in S if i < j and (i, j) in edges]
            if S_card >= 3 and (len(S_edges) >= S_card or NS > 1):
                cutsadded = True
                if checkonly:
                    break
                else:
                    self.model.addCons(quicksum(x[i, j] for i in S for j in S if j > i) <= S_card - NS)
                    print("adding cut for", S_edges)

        return cutsadded

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        if self.addCuts(checkonly = True):
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        if self.addCuts(checkonly = False):
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, nlockspos, nlocksneg):
        pass


def vrp(V, c, m, q, Q):
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

    model = Model("vrp")
    vrp_conshdlr = VRPconshdlr()

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
    model.data = x

    return model, vrp_conshdlr


def distance(x1,y1,x2,y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def make_data(n):
    """make_data: compute matrix distance based on euclidean distance"""
    V = range(1,n+1)
    x = dict([(i, random.random()) for i in V])
    y = dict([(i, random.random()) for i in V])
    c, q = {}, {}
    Q = 100
    for i in V:
        q[i] = random.randint(10, 20)
        for j in V:
            if j > i:
                c[i, j] = distance(x[i], y[i], x[j], y[j])
    return V, c, q, Q


if __name__ == "__main__":
    n = 19
    m = 3
    seed = 1
    random.seed(seed)
    V,c,q,Q = make_data(n)
    model, conshdlr = vrp(V, c, m, q, Q)

    model.setBoolParam("misc/allowstrongdualreds", 0)
    model.includeConshdlr(conshdlr, "VRP", "VRP constraint handler",
                          sepapriority = 0, enfopriority = 1, chckpriority = 1, sepafreq = -1, propfreq = -1,
                          eagerfreq = -1, maxprerounds = 0, delaysepa = False, delayprop = False, needscons = False,
                          presoltiming = SCIP_PRESOLTIMING.FAST, proptiming = SCIP_PROPTIMING.BEFORELP)
    model.optimize()
    x = model.data

    edges = []
    for (i, j) in x:
        if model.getVal(x[i, j]) > .5:
            if i != V[0] and j != V[0]:
                edges.append((i, j))

    print("Optimal solution:", model.getObjVal())
    print("Edges in the solution:")
    print(sorted(edges))
