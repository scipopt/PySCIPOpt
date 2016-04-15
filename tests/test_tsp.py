import networkx
from pyscipopt import Model, Conshdlr, quicksum,  \
  SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING

class TSPconshdlr(Conshdlr):

  def findSubtours(self, checkonly, solution):
    edges = []
    x = self.model.data
    for (i,j) in x:
      if self.model.getSolVal(solution, x[i,j]) > 1.e-6:
        edges.append((i,j))

    G = networkx.Graph()
    G.add_edges_from(edges)
    Components = list(networkx.connected_components(G))

    if len(Components) == 1:
      return False
    elif checkonly:
      return True

    for S in Components:
      self.model.addCons(quicksum(x[i,j]
        for i in S for j in S if j > i) <= len(S) - 1)
      print("cut: len(%s) <= %s" % (S, len(S) - 1))

    return True

  def conscheck(self, constraints, solution, checkintegrality,
                checklprows, printreason):
    if self.findSubtours(checkonly = True, solution = solution):
      return {"result": SCIP_RESULT.INFEASIBLE}
    else:
      return {"result": SCIP_RESULT.FEASIBLE}

  def consenfolp(self, constraints, nusefulconss, solinfeasible):
    if self.findSubtours(checkonly = False, solution = None):
      return {"result": SCIP_RESULT.CONSADDED}
    else:
      return {"result": SCIP_RESULT.FEASIBLE}

  def conslock(self, constraint, nlockspos, nlocksneg):
    pass

def create_tsp(V,c):
  model = Model("TSP")

  x = {}
  for i in V:
    for j in V:
      if j > i:
        x[i,j] = model.addVar(vtype = "B",name = "x(%s,%s)" % (i,j))

  for i in V:
    model.addCons(
      quicksum(x[j,i] for j in V if j < i) +
      quicksum(x[i,j] for j in V if j > i) == 2, "Degree(%s)" % i)

  model.setObjective(
    quicksum(c[i,j] * x[i,j] for i in V for j in V if j > i),
    "minimize")

  model.data = x
  return model

def solve_tsp(V,c):
  model = create_tsp(V,c)
  conshdlr = TSPconshdlr()
  model.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator",
    sepapriority = 0, enfopriority = -1, chckpriority = -1,
    sepafreq = -1, propfreq = -1, eagerfreq = -1, maxprerounds = 0,
    delaysepa = False, delayprop = False, needscons = False,
    presoltiming = SCIP_PRESOLTIMING.FAST,
    proptiming = SCIP_PROPTIMING.BEFORELP)
  model.setBoolParam("misc/allowdualreds", 0)
  model.optimize()
  x = model.data
  edges = []
  for (i,j) in x:
    if model.getVal(x[i,j]) > 1.e-6:
      edges.append((i,j))
  return model.getObjVal(), edges


def test_main():
  V = [1, 2, 3, 4, 5, 6]
  c = {(u,v):1 for u in V for v in V if u < v}
  for u in [1, 2, 3]:
    for v in [4, 5, 6]:
      c[u,v] = 10

  obj, edges = solve_tsp(V, c)

  print("Optimal tour:", edges)
  print("Optimal cost:", obj)

if __name__ == "__main__":
  test_main()
