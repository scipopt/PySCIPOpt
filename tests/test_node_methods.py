from helpers.utils import random_mip_1
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE, SCIP_RESULT

class MaxDepthTracker(Eventhdlr):
    def __init__(self):
        super().__init__()
        self.max_depth = -1

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        current_node = self.model.getCurrentNode()
        if current_node is not None:
            depth = current_node.getDepth()
            self.max_depth = max(self.max_depth, depth)
        return {'result': SCIP_RESULT.SUCCESS}

def test_getMaxDepth():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,
        small=True
    )

    print(f"Initial max depth: {m.getMaxDepth()}")
    assert m.getMaxDepth() == -1

    tracker = MaxDepthTracker()
    m.includeEventhdlr(tracker, "maxdepth_tracker", "Tracks maximum depth of nodes")

    m.optimize()

    max_depth = m.getMaxDepth()
    tracked_max_depth = tracker.max_depth
    nodes = m.getNNodes()

    print(f"Max depth after solving: {max_depth}")
    print(f"Tracked max depth: {tracked_max_depth}")
    print(f"Number of nodes explored: {nodes}")
    print(f"Optimization status: {m.getStatus()}")

    assert max_depth >= 0, f"Expected max_depth >= 0, got {max_depth}"

    if nodes > 1:
        assert max_depth >= 1, f"Expected max_depth >= 1 with {nodes} nodes, got {max_depth}"

    assert max_depth <= nodes, f"Max depth {max_depth} shouldn't exceed nodes {nodes}"

    # Verify that getMaxDepth() matches the actual maximum depth of all nodes
    assert max_depth == tracked_max_depth, f"getMaxDepth() returned {max_depth} but tracked max depth is {tracked_max_depth}"


def test_getPlungeDepth():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,
        small=True
    )

    initial_plunge = m.getPlungeDepth()
    print(f"Initial plunge depth: {initial_plunge}")
    assert initial_plunge == 0, f"Expected initial plunge depth to be 0, got {initial_plunge}"

    m.optimize()

    plunge_depth = m.getPlungeDepth()
    nodes = m.getNNodes()
    max_depth = m.getMaxDepth()

    print(f"Plunge depth after solving: {plunge_depth}")
    print(f"Number of nodes: {nodes}")
    print(f"Max depth: {max_depth}")

    assert plunge_depth >= 0, f"Expected plunge_depth >= 0, got {plunge_depth}"

    # If we explored multiple nodes and reached some depth, we likely did some plunging
    if nodes > 1 and max_depth > 0:
        assert plunge_depth >= 1, f"Expected plunge_depth >= 1 with {nodes} nodes and max_depth {max_depth}, got {plunge_depth}"


def test_getLowerbound():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,
        small=True
    )

    initial_lb = m.getLowerbound()
    print(f"Initial lower bound: {initial_lb}")

    m.optimize()

    lower_bound = m.getLowerbound()
    obj_val = m.getObjVal()

    print(f"Lower bound after solving: {lower_bound}")
    print(f"Status: {m.getStatus()}")

    assert initial_lb < lower_bound, f"Expected initial lower bound {initial_lb} to be less than final lower bound {lower_bound}"


def test_getCutoffbound():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,
        node_lim=10000,
        small=True
    )

    m.setIntParam("limits/solutions", 1)

    m.optimize()

    cutoff = m.getCutoffbound()
    obj_val = m.getObjVal() if m.getNSols() > 0 else None

    print(f"Cutoff bound after solving: {cutoff}")
    print(f"Objective value: {obj_val}")
    print(f"Status: {m.getStatus()}")

    assert abs(cutoff - obj_val) < 1e-6, f"Cutoff {cutoff} should equal optimal value {obj_val}"


def test_getNNodeLPIterations():
    m = random_mip_1(
        disable_sepa=False,
        disable_heur=True,
        disable_presolve=True,
        node_lim=30,
        small=True
    )

    initial_lp_iters = m.getNNodeLPIterations()
    print(f"Initial node LP iterations: {initial_lp_iters}")
    assert initial_lp_iters == 0, f"Expected 0 initial LP iterations, got {initial_lp_iters}"

    m.optimize()

    lp_iters = m.getNNodeLPIterations()
    total_lp_iters = m.getNLPIterations()
    nodes = m.getNNodes()

    print(f"Node LP iterations after solving: {lp_iters}")
    print(f"Total LP iterations: {total_lp_iters}")
    print(f"Number of nodes: {nodes}")

    assert lp_iters >= 0, f"Expected non-negative LP iterations, got {lp_iters}"
    assert lp_iters <= total_lp_iters, f"Node LP iterations {lp_iters} should not exceed total LP iterations {total_lp_iters}"

    if nodes > 0:
        assert lp_iters > 0, f"Expected positive LP iterations with {nodes} nodes explored"


def test_getNStrongbranchLPIterations():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,
        node_lim=20,
        small=True
    )

    initial_sb_iters = m.getNStrongbranchLPIterations()
    print(f"Initial strong branching LP iterations: {initial_sb_iters}")
    assert initial_sb_iters == 0, f"Expected 0 initial strong branching iterations, got {initial_sb_iters}"

    m.optimize()

    sb_iters = m.getNStrongbranchLPIterations()
    total_lp_iters = m.getNLPIterations()
    nodes = m.getNNodes()

    print(f"Strong branching LP iterations: {sb_iters}")
    print(f"Total LP iterations: {total_lp_iters}")
    print(f"Number of nodes: {nodes}")

    assert sb_iters >= 0, f"Expected non-negative strong branching iterations, got {sb_iters}"
    assert sb_iters <= total_lp_iters, f"Strong branching iterations {sb_iters} should not exceed total LP iterations {total_lp_iters}"
