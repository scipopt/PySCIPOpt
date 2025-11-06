from pyscipopt import Model, SCIP_PARAMSETTING, Nodesel, SCIP_NODETYPE
from pyscipopt.scip import Node


class HybridEstim(Nodesel):
    """
    Hybrid best estimate / best bound node selection plugin.

    This implements the hybrid node selection strategy from SCIP, which combines
    best estimate and best bound search with a plunging heuristic.
    """

    def __init__(self, model, minplungedepth=-1, maxplungedepth=-1, maxplungequot=0.25,
                 bestnodefreq=1000, estimweight=0.10):
        """
        Initialize the hybrid estimate node selector.

        Parameters
        ----------
        model : Model
            The SCIP model
        minplungedepth : int
            Minimal plunging depth before new best node may be selected
            (-1 for dynamic setting)
        maxplungedepth : int
            Maximal plunging depth before new best node is forced to be selected
            (-1 for dynamic setting)
        maxplungequot : float
            Maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
            where plunging is performed
        bestnodefreq : int
            Frequency at which the best node instead of the hybrid best estimate/best bound
            is selected (0: never)
        estimweight : float
            Weight of estimate value in node selection score
            (0: pure best bound search, 1: pure best estimate search)
        """
        super().__init__()
        self.scip = model
        self.minplungedepth = minplungedepth
        self.maxplungedepth = maxplungedepth
        self.maxplungequot = maxplungequot
        self.bestnodefreq = bestnodefreq if bestnodefreq > 0 else float('inf')
        self.estimweight = estimweight

    def _get_nodesel_score(self, node: Node) -> float:
        """
        Returns a weighted sum of the node's lower bound and estimate value.

        Parameters
        ----------
        node : Node
            The node to evaluate

        Returns
        -------
        float
            The node selection score
        """
        return ((1.0 - self.estimweight) * node.getLowerbound() +
                self.estimweight * node.getEstimate())

    def nodeselect(self):
        """
        Select the next node to process.

        Returns
        -------
        dict
            Dictionary with 'selnode' key containing the selected node
        """
        # Calculate minimal and maximal plunging depth
        minplungedepth = self.minplungedepth
        maxplungedepth = self.maxplungedepth

        if minplungedepth == -1:
            minplungedepth = self.scip.getMaxDepth() // 10
            # Adjust based on strong branching iterations
            if (self.scip.getNStrongbranchLPIterations() >
                2 * self.scip.getNNodeLPIterations()):
                minplungedepth += 10
            if maxplungedepth >= 0:
                minplungedepth = min(minplungedepth, maxplungedepth)

        if maxplungedepth == -1:
            maxplungedepth = self.scip.getMaxDepth() // 2

        maxplungedepth = max(maxplungedepth, minplungedepth)

        # Check if we exceeded the maximal plunging depth
        plungedepth = self.scip.getPlungeDepth()

        if plungedepth > maxplungedepth:
            # We don't want to plunge again: select best node from the tree
            if self.scip.getNNodes() % self.bestnodefreq == 0:
                selnode = self.scip.getBestboundNode()
            else:
                selnode = self.scip.getBestNode()
        else:
            # Get global lower and cutoff bound
            lowerbound = self.scip.getLowerbound()
            cutoffbound = self.scip.getCutoffbound()

            # If we didn't find a solution yet, tighten the cutoff bound to 20% of the range between it and the lowerbound.
            if self.scip.getNSols() == 0:
                cutoffbound = lowerbound + 0.2 * (cutoffbound - lowerbound)

            # Check if plunging is forced at the current depth
            if plungedepth < minplungedepth:
                maxbound = float('inf')
            else:
                # Calculate maximal plunging bound
                maxbound = lowerbound + self.maxplungequot * (cutoffbound - lowerbound)

            # We want to plunge again: prefer children over siblings, and siblings over leaves
            # but only select a child or sibling if its estimate is small enough
            selnode = None

            # Try each node type in priority order
            node_getters = [
                self.scip.getPrioChild,
                self.scip.getBestChild,
                self.scip.getPrioSibling,
                self.scip.getBestSibling,
            ]

            for get_node in node_getters:
                node = get_node()
                if node is not None and node.getEstimate() < maxbound:
                    selnode = node
                    break

            # If no suitable child or sibling found, select from leaves
            if selnode is None:
                if self.scip.getNNodes() % self.bestnodefreq == 0:
                    selnode = self.scip.getBestboundNode()
                else:
                    selnode = self.scip.getBestNode()

        return {"selnode": selnode}

    def nodecomp(self, node1, node2):
        """
        Compare two nodes.

        Parameters
        ----------
        node1 : Node
            First node to compare
        node2 : Node
            Second node to compare

        Returns
        -------
        int
            -1 if node1 is better than node2
            0 if both nodes are equally good
            1 if node1 is worse than node2
        """
        score1 = self._get_nodesel_score(node1)
        score2 = self._get_nodesel_score(node2)

        # Check if scores are equal or both infinite
        any_infinite = self.scip.isInfinity(abs(score1)) or self.scip.isInfinity(abs(score2))
        if ( (not any_infinite and self.scip.isEQ(score1, score2)) or
            (self.scip.isInfinity(score1) and self.scip.isInfinity(score2)) or
            (self.scip.isInfinity(-score1) and self.scip.isInfinity(-score2))):

            # Prefer children over siblings over leaves
            nodetype1 = node1.getType()
            nodetype2 = node2.getType()

            # SCIP node types: CHILD = 0, SIBLING = 1, LEAF = 2
            if nodetype1 == SCIP_NODETYPE.CHILD and nodetype2 != SCIP_NODETYPE.CHILD:  # node1 is child, node2 is not
                return -1
            elif nodetype1 != SCIP_NODETYPE.CHILD and nodetype2 == SCIP_NODETYPE.CHILD:  # node2 is child, node1 is not
                return 1
            elif nodetype1 == SCIP_NODETYPE.SIBLING and nodetype2 != SCIP_NODETYPE.SIBLING:  # node1 is sibling, node2 is not
                return -1
            elif nodetype1 != SCIP_NODETYPE.SIBLING and nodetype2 == SCIP_NODETYPE.SIBLING:  # node2 is sibling, node1 is not
                return 1
            else:
                # Same node type, compare depths (prefer shallower nodes)
                depth1 = node1.getDepth()
                depth2 = node2.getDepth()
                if depth1 < depth2:
                    return -1
                elif depth1 > depth2:
                    return 1
                else:
                    return 0

        # Compare scores
        if score1 < score2:
            return -1
        else:
            return 1
        
def test_hybridestim_vs_default():
    """
    Test that the Python hybrid estimate node selector performs similarly
    to the default SCIP C implementation.
    """
    import os

    # Get the path to the 10teams instance
    instance_path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "10teams.mps")

    # Test with default SCIP hybrid estimate node selector
    m_default = Model()
    m_default.readProblem(instance_path)

    # Disable presolving, heuristics, and separation to focus on node selection
    m_default.setPresolve(SCIP_PARAMSETTING.OFF)
    m_default.setHeuristics(SCIP_PARAMSETTING.OFF)
    m_default.setSeparating(SCIP_PARAMSETTING.OFF)
    m_default.setParam("limits/nodes", 2000)
    m_default.setParam("nodeselection/hybridestim/stdpriority", 1_000_000)

    m_default.optimize()

    default_lp_iterations = m_default.getNLPIterations()
    default_nodes = m_default.getNNodes()
    default_obj = m_default.getObjVal() if m_default.getNSols() > 0 else None

    print(f"Default SCIP hybrid estimate node selector (C implementation):")
    print(f"  Nodes: {default_nodes}")
    print(f"  LP iterations: {default_lp_iterations}")
    print(f"  Objective: {default_obj}")

    # Test with Python implementation
    m_python = Model()
    m_python.readProblem(instance_path)

    # Disable presolving, heuristics, and separation to focus on node selection
    m_python.setPresolve(SCIP_PARAMSETTING.OFF)
    m_python.setHeuristics(SCIP_PARAMSETTING.OFF)
    m_python.setSeparating(SCIP_PARAMSETTING.OFF)
    m_python.setParam("limits/nodes", 2000)

    # Include our Python hybrid estimate node selector
    hybridestim_nodesel = HybridEstim(
        m_python,
    )
    m_python.includeNodesel(
        hybridestim_nodesel,
        "pyhybridestim",
        "Python hybrid best estimate / best bound search",
        stdpriority=1_000_000,
        memsavepriority=50
    )

    m_python.optimize()

    python_lp_iterations = m_python.getNLPIterations()
    python_nodes = m_python.getNNodes()
    python_obj = m_python.getObjVal() if m_python.getNSols() > 0 else None

    print(f"\nPython hybrid estimate node selector:")
    print(f"  Nodes: {python_nodes}")
    print(f"  LP iterations: {python_lp_iterations}")
    print(f"  Objective: {python_obj}")

    # Check if LP iterations are the same
    assert default_lp_iterations == python_lp_iterations, \
        "LP iterations differ between default and Python implementations!"


if __name__ == "__main__":
    test_hybridestim_vs_default()