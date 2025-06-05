from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING

"""
This is a test for the bipartite graph generation functionality.
To make the test more practical, we embed the function in a dummy branching rule. Such functionality would allow
users to then extract the feature set before any branching decision. This can be used to gather data for training etc 
and to to deploy actual branching rules trained on data from the graph representation.
"""


class DummyFeatureExtractingBranchRule(Branchrule):

    def __init__(self, scip, static=False, use_prev_states=True):
        self.scip = scip
        self.static = static
        self.use_prev_states = use_prev_states
        self.prev_col_features = None
        self.prev_row_features = None
        self.prev_edge_features = None

    def branchexeclp(self, allowaddcons):

        # Get the bipartite graph data
        if self.use_prev_states:
            prev_col_features = self.prev_col_features
            prev_edge_features = self.prev_edge_features
            prev_row_features = self.prev_row_features
        else:
            prev_col_features = None
            prev_edge_features = None
            prev_row_features = None
        col_features, edge_features, row_features, feature_maps = self.scip.getBipartiteGraphRepresentation(
            prev_col_features=prev_col_features, prev_edge_features=prev_edge_features,
            prev_row_features=prev_row_features, static_only=self.static
        )

        # Here is now where a decision could be based off the features. If no decision is made just return DIDNOTRUN

        return {"result": SCIP_RESULT.DIDNOTRUN}



def create_model():
    scip = Model()
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setLongintParam("limits/nodes", 250)
    scip.setParam("presolving/maxrestarts", 0)

    x0 = scip.addVar(lb=-2, ub=4)
    r1 = scip.addVar()
    r2 = scip.addVar()
    y0 = scip.addVar(lb=3)
    t = scip.addVar(lb=None)
    l = scip.addVar(vtype="I", lb=-9, ub=18)
    u = scip.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-12, ub=40))
        scip.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-52, ub=10))
        scip.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[200::2]))

    scip.addCons(r1 >= x0)
    scip.addCons(r2 >= -x0)
    scip.addCons(y0 == r1 + r2)
    scip.addCons(t + l + 7 * u <= 300)
    scip.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5 * more_vars[9])
    scip.addCons(more_vars[3] >= l + 2)
    scip.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    scip.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))

    scip.setObjective(t - quicksum(j * v for j, v in enumerate(more_vars[20:-40])))

    return scip


def test_bipartite_graph():
    scip = create_model()

    dummy_branch_rule = DummyFeatureExtractingBranchRule(scip)
    scip.includeBranchrule(dummy_branch_rule, "dummy branch rule", "custom feature extraction branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()


def test_bipartite_graph_static():
    scip = create_model()

    dummy_branch_rule = DummyFeatureExtractingBranchRule(scip, static=True)
    scip.includeBranchrule(dummy_branch_rule, "dummy branch rule", "custom feature extraction branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()

def test_bipartite_graph_use_prev():
    scip = create_model()

    dummy_branch_rule = DummyFeatureExtractingBranchRule(scip, use_prev_states=True)
    scip.includeBranchrule(dummy_branch_rule, "dummy branch rule", "custom feature extraction branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()

def test_bipartite_graph_static_use_prev():
    scip = create_model()

    dummy_branch_rule = DummyFeatureExtractingBranchRule(scip, static=True, use_prev_states=True)
    scip.includeBranchrule(dummy_branch_rule, "dummy branch rule", "custom feature extraction branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()