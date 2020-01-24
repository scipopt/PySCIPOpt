import pytest

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING

calls = []

class MyEvent(Eventhdlr):

    def eventinit(self):
        calls.append('eventinit')
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexit(self):
        calls.append('eventexit')
        self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexec(self, event):
        calls.append('eventexec')
        assert event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED
        assert event.getNode().getNumber() == 1
        assert event.getNode().getParent() is None


def test_event():
    # create solver instance
    s = Model()
    s.hideOutput()
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    eventhdlr = MyEvent()
    s.includeEventhdlr(eventhdlr, "TestFirstLPevent", "python event handler to catch FIRSTLPEVENT")

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)
    # solve problem
    s.optimize()

    # print solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0

    del s

    assert 'eventinit' in calls
    assert 'eventexit' in calls
    assert 'eventexec' in calls
    assert len(calls) == 3

if __name__ == "__main__":
    test_event()
