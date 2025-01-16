from pyscipopt import Model, SCIP_PARAMSETTING
from pyscipopt.scip import Nodesel
from helpers.utils import random_mip_1

class FiFo(Nodesel):

    def nodeselect(self):
        """first method called in each iteration in the main solving loop."""

        leaves, children, siblings = self.model.getOpenNodes()
        nodes = leaves + children + siblings

        return {"selnode": nodes[0]} if len(nodes) > 0 else {}

    def nodecomp(self, node1, node2):
        """
        compare two leaves of the current branching tree

        It should return the following values:

          value < 0, if node 1 comes before (is better than) node 2
          value = 0, if both nodes are equally good
          value > 0, if node 1 comes after (is worse than) node 2.
        """
        return 0


# Depth First Search Node Selector
class DFS(Nodesel):

    def __init__(self, scip, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scip = scip

    def nodeselect(self):

        selnode = self.scip.getPrioChild()
        if selnode is None:
            selnode = self.scip.getPrioSibling()
            if selnode is None:
                selnode = self.scip.getBestLeaf()

        return {"selnode": selnode}

    def nodecomp(self, node1, node2):
        """
        compare two leaves of the current branching tree

        It should return the following values:

        value < 0, if node 1 comes before (is better than) node 2
        value = 0, if both nodes are equally good
        value > 0, if node 1 comes after (is worse than) node 2.
        """
        depth_1 = node1.getDepth()
        depth_2 = node2.getDepth()
        if depth_1 > depth_2:
            return -1
        elif depth_1 < depth_2:
            return 1
        else:
            lb_1 = node1.getLowerbound()
            lb_2 = node2.getLowerbound()
            if lb_1 < lb_2:
                return -1
            elif lb_1 > lb_2:
                return 1
            else:
                return 0


def test_nodesel_fifo():
    m = Model()

    # include node selector
    m.includeNodesel(FiFo(), "testnodeselector", "Testing a node selector.", 1073741823, 536870911)

    # add Variables
    x0 = m.addVar(vtype="C", name="x0", obj=-1)
    x1 = m.addVar(vtype="C", name="x1", obj=-1)
    x2 = m.addVar(vtype="C", name="x2", obj=-1)

    # add constraints
    m.addCons(x0 >= 2)
    m.addCons(x0 ** 2 <= x1)
    m.addCons(x1 * x2 >= x0)

    m.setObjective(x1 + x0)
    m.optimize()

def test_nodesel_dfs():
    m = random_mip_1(node_lim=500)

    # include node selector
    dfs_node_sel = DFS(m)
    m.includeNodesel(dfs_node_sel, "DFS", "Depth First Search Nodesel.", 1000000, 1000000)

    m.optimize()