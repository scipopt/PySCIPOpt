from pyscipopt import Model, Conshdlr, quicksum, SCIP_RESULT

import pytest

itertools = pytest.importorskip("itertools")
networkx = pytest.importorskip("networkx")

EPS = 1.e-6


# subtour elimination constraint handler
class TSPconshdlr(Conshdlr):

    def __init__(self, variables):
        self.variables = variables

    # find subtours in the graph induced by the edges {i,j} for which x[i,j] is positive
    # at the given solution; when solution is None, then the LP solution is used
    def find_subtours(self, solution = None):
        edges = []
        x = self.variables
        for (i, j) in x:
            if self.model.getSolVal(solution, x[i, j]) > EPS:
                edges.append((i, j))

        G = networkx.Graph()
        G.add_edges_from(edges)
        components = list(networkx.connected_components(G))

        if len(components) == 1:
            return []
        else:
            return components

    # checks whether solution is feasible, ie, if there are no subtours;
    # since the checkpriority is < 0, we are only called if the integrality
    # constraint handler didn't find infeasibility, so solution is integral
    def conscheck(self, constraints, solution, check_integrality,
                  check_lp_rows, print_reason, completely, **results):
        if self.find_subtours(solution):
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    # enforces LP solution
    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        subtours = self.find_subtours()
        if subtours:
            x = self.variables
            for subset in subtours:
                self.model.addCons(quicksum(x[i, j] for(i, j) in pairs(subset))
                                   <= len(subset) - 1)
                print("cut: len(%s) <= %s" % (subset, len(subset) - 1))
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


# builds tsp model; adds variables, degree constraint and the subtour elimination constaint handler
def create_tsp(vertices, distance):
    model = Model("TSP")

    x = {}  # binary variable to select edges
    for (i, j) in pairs(vertices):
        x[i, j] = model.addVar(vtype = "B", name = "x(%s,%s)" % (i, j))

    for i in vertices:
        model.addCons(quicksum(x[j, i] for j in vertices if j < i) +
                      quicksum(x[i, j] for j in vertices if j > i) == 2, "Degree(%s)" % i)

    conshdlr = TSPconshdlr(x)

    model.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator",
                          chckpriority = -10, needscons = False)
    model.setBoolParam("misc/allowstrongdualreds", False)

    model.setObjective(quicksum(distance[i, j] * x[i, j] for (i, j) in pairs(vertices)),
                       "minimize")

    return model, x


def solve_tsp(vertices, distance):
    model, x = create_tsp(vertices, distance)
    model.optimize()

    edges = []
    for (i, j) in x:
        if model.getVal(x[i, j]) > EPS:
            edges.append((i, j))

    return model.getObjVal(), edges


# returns all undirected edges between vertices
def pairs(vertices):
    return itertools.combinations(vertices, 2)


def test_main():
    vertices = [1, 2, 3, 4, 5, 6]
    distance = {(u, v):1 for (u, v) in pairs(vertices)}

    for u in vertices[:3]:
        for v in vertices[3:]:
            distance[u, v] = 10

    objective_value, edges = solve_tsp(vertices, distance)

    print("Optimal tour:", edges)
    print("Optimal cost:", objective_value)