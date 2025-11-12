#############
Node Selector
#############

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model
  from pyscipopt.scip import Nodesel

  scip = Model()

.. contents:: Contents

What is a Node Selector?
========================

In the branch-and-bound tree an important question that must be answered is which node should currently
be processed. That is, given a branch-and-bound tree in an intermediate state, select a leaf node of the tree
that will be processed next (most likely branched on). In SCIP this problem has its own plug-in,
and thus custom algorithms can easily be included into the solving process!

Example Node Selector
=====================

In this example we are going to implement a depth first search node selection strategy.
There are two functions that we need to code ourselves when adding such a rule from python.
The first is the strategy on which node to select from all the current leaves, and the other
is a comparison function that decides which node is preferred from two candidates.

.. code-block:: python

    # Depth First Search Node Selector
    class DFS(Nodesel):

        def __init__(self, scip, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scip = scip

        def nodeselect(self):
            """Decide which of the leaves from the branching tree to process next"""
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

.. note:: In general when implementing a node selection rule you will commonly use either ``getPrioChild``
  or ``getBestChild``. The first returns the child of the current node with
  the largest node selection priority, as assigned by the branching rule. The second
  returns the best child of the current node with respect to the node selector's ordering relation as defined
  in ``nodecomp``.

To include the node selector in your SCIP Model one would use the following code:

.. code-block:: python

    dfs_node_sel = DFS(scip)
    scip.includeNodesel(dfs_node_sel, "DFS", "Depth First Search Nodesel.", 1000000, 1000000)

For a more complex example, see the `Hybrid Estimate Node Selector <https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/nodesel_hybridestim.py>`_ on GitHub.




