import networkx
from pyscipopt import Model, Conshdlr, quicksum, SCIP_RESULT

EPS = 1.e-6

class TSPconshdlr(Conshdlr):

  def __init__(self, variables):
    self.variables = variables

  def find_subtours(self, solution=None):
    edges = []
    x = self.variables
    for (i,j) in x:
      if self.model.getSolVal(solution, x[i,j]) > EPS:
        edges.append((i,j))

    G = networkx.Graph()
    G.add_edges_from(edges)
    components = list(networkx.connected_components(G))

    if len(components) == 1:
      return []
    else:
      return components

  def conscheck(self, constraints, solution, check_integrality,
                check_lp_rows, print_reason, **results):
    if self.find_subtours(solution):
      return {"result": SCIP_RESULT.INFEASIBLE}
    else:
      return {"result": SCIP_RESULT.FEASIBLE}

  def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
    subtours = self.find_subtours()
    if subtours:
      x = self.variables
      for subset in subtours:
        self.model.addCons(quicksum(x[i,j] for(i, j) in pairs(subset))
                           <= len(subset) - 1)
        print("cut: len(%s) <= %s" % (subset, len(subset) - 1))
      return {"result": SCIP_RESULT.CONSADDED}
    else:
      return {"result": SCIP_RESULT.FEASIBLE}

  def conslock(self, constraint, nlockspos, nlocksneg):
    pass

def create_tsp(vertices, distance):
  model = Model("TSP")

  x = {} # binary variable to select edges
  for (i, j) in pairs(vertices):
    x[i,j] = model.addVar(vtype = "B",name = "x(%s,%s)" % (i,j))

  for i in vertices:
    model.addCons(
      quicksum(x[j,i] for j in vertices if j < i) +
      quicksum(x[i,j] for j in vertices if j > i) == 2, "Degree(%s)" % i)

  conshdlr = TSPconshdlr(x)
  model.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator",
                        needscons=False)
  model.setBoolParam("misc/allowdualreds", False)

  model.setObjective(
    quicksum(distance[i,j] * x[i,j] for (i,j) in pairs(vertices)),
    "minimize")

  return model, x

def solve_tsp(vertices, distance):
  model, x = create_tsp(vertices, distance)
  model.optimize()

  edges = []
  for (i,j) in x:
    if model.getVal(x[i,j]) > EPS:
      edges.append((i,j))

  return model.getObjVal(), edges

def pairs(vertices):
  for i in vertices:
    for j in vertices:
      if i < j:
        yield (i,j)

def test_main():
  vertices = [1, 2, 3, 4, 5, 6]
  distance = {(u,v):1 for (u,v) in pairs(vertices)}

  for u in vertices[:3]:
    for v in vertices[3:]:
      distance[u,v] = 10

  objective_value, edges = solve_tsp(vertices, distance)

  print("Optimal tour:", edges)
  print("Optimal cost:", objective_value)

if __name__ == "__main__":
  test_main()
