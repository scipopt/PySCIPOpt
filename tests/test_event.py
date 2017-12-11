import pytest

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING

class MyEvent(Eventhdlr):

    def init(self):
        print("in init")
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def exit(self):
        print("in exit")
        self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexec(self, event):
        print("MyEvent was triggered")

def test_event():
    # create solver instance
    s = Model()
    print("before creation")
    eventhdlr = MyEvent()
    print("before including")
    s.includeEventhdlr(eventhdlr, "Myevent", "custom event handler implemented in python")
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    print("before catching")
    s.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, eventhdlr)

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)
    print("before optimize")
    # solve problem
    s.optimize()

    # print solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0

if __name__ == "__main__":
    test_event()
