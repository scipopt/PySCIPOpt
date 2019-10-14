from pyscipopt import Model
from pyscipopt.scip import Nodesel

class FiFo(Nodesel):

  def nodeselect(self):
    '''first method called in each iteration in the main solving loop. '''

    leaves, children, siblings = self.model.getOpenNodes()
    nodes = leaves + children + siblings

    return {"selnode" : nodes[0]} if len(nodes) > 0 else {}

  def nodecomp(self, node1, node2):
    '''
    compare two leaves of the current branching tree

    It should return the following values:

      value < 0, if node 1 comes before (is better than) node 2
      value = 0, if both nodes are equally good
      value > 0, if node 1 comes after (is worse than) node 2.
    '''
    return 0

def test_nodesel():
    m = Model()
    m.hideOutput()

    # include node selector
    m.includeNodesel(FiFo(), "testnodeselector", "Testing a node selector.", 1073741823, 536870911)

    # add Variables
    x0 = m.addVar(vtype = "C", name = "x0", obj=-1)
    x1 = m.addVar(vtype = "C", name = "x1", obj=-1)
    x2 = m.addVar(vtype = "C", name = "x2", obj=-1)

    # add constraints
    m.addCons(x0 >= 2)
    m.addCons(x0**2 <= x1)
    m.addCons(x1 * x2 >= x0)

    m.setObjective(x1 + x0)
    m.optimize()


if __name__ == "__main__":
    test_nodesel()
