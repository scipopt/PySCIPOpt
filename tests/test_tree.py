import pytest

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING


class NodeEventHandler(Eventhdlr):

    def __init__(self):
        self.calls = []

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        self.calls.append('eventexec')
        assert event.getType() == SCIP_EVENTTYPE.NODEFOCUSED
        node = event.getNode()
        
        if node.getDepth() == 0:
            assert node.getParent() is None
            assert node.getParentBranchings() is None
            return

        variables, branchbounds, boundtypes = node.getParentBranchings()
        assert len(variables) == 1
        assert len(branchbounds) == 1
        assert len(boundtypes) == 1
        domain_changes = node.getDomchg()
        bound_changes = domain_changes.getBoundchgs()
        assert len(bound_changes) == 1


def test_tree():
    # create solver instance
    s = Model()
    s.setMaximize()
    s.hideOutput()
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    node_eventhdlr = NodeEventHandler()
    s.includeEventhdlr(node_eventhdlr, "NodeEventHandler", "python event handler to catch NODEFOCUSED")

    # add some variables
    n = 121
    x = [s.addVar("x{}".format(i), obj=1.0, vtype="INTEGER") for i in range(n)]

    # add some constraints
    for i in range(n):
      for j in range(i):
        dist = min(abs(i - j), abs(n - i - j))
        if dist in (1, 3, 4):
          s.addCons(x[i] + x[j] <= 1)
    # solve problem
    s.optimize()

    # print solution
    assert round(s.getObjVal()) == 36.0

    del s

    assert len(node_eventhdlr.calls) > 3