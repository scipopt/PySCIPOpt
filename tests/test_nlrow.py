import pytest

from pyscipopt import Model, Heur, SCIP_RESULT, quicksum, SCIP_PARAMSETTING, SCIP_HEURTIMING

class MyHeur(Heur):
    def heurexec(self, heurtiming, nodeinfeasible):
        self.model.interruptSolve()
        return {"result": SCIP_RESULT.DIDNOTFIND}

def test_nlrow():

    # create nonlinear model
    m = Model("nlrow")

    # add heuristic to interrupt solve: the methods we wanna test can only be called in solving stage
    heuristic = MyHeur()
    m.includeHeur(heuristic, "PyHeur", "heur to interrupt", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)

    # create variables
    x = m.addVar(name="x", lb=-3, ub=3, obj=-1)
    y = m.addVar(name="y", lb=-3, ub=3, obj=-1)

    # create constraints
    m.addCons(1*x + 2*y + 3 * x**2 + 4*y**2  + 5*x*y <= 6)
    m.addCons(7*x**2 + 8*y**2 == 9)
    m.addCons(10*x + 11*y <= 12)

    # optimize without presolving
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.optimize()

    # check whether NLP has been constructed and there are 3 nonlinear rows that match the above constraints
    assert m.isNLPConstructed()
    assert m.getNNlRows() == 3

    # collect nonlinear rows
    nlrows = m.getNlRows()
    
    # test getNNlRows
    assert len(nlrows) == m.getNNlRows()

    # to test printing of NLRows
    for row in nlrows:
        m.printNlRow(row)

    # the nlrow that corresponds to the linear (third) constraint is added before the nonlinear rows,
    # because Initsol of the linear conshdlr gets called first
    # therefore the ordering is: nlrows[0] is for constraint 3, nlrows[1] is for constraint 1,
    # nlrows[2] is for constraint 2

    # check first nonlinear row that represents constraint 3
    assert nlrows[0].getLhs() == -m.infinity()
    assert nlrows[0].getRhs() == 12

    linterms = nlrows[0].getLinearTerms()
    assert len(linterms) == 2
    assert str(linterms[0][0]) == "t_x"
    assert linterms[0][1] == 10
    assert str(linterms[1][0]) == "t_y"
    assert linterms[1][1] == 11

    linterms = nlrows[1].getLinearTerms()
    assert len(linterms) == 2
    assert str(linterms[0][0]) == "t_x"
    assert linterms[0][1] == 1
    assert str(linterms[1][0]) == "t_y"
    assert linterms[1][1] == 2

    # check third nonlinear row that represents constraint 2
    assert nlrows[2].getLhs() == 9
    assert nlrows[2].getRhs() == 9

    linterms = nlrows[2].getLinearTerms()
    assert len(linterms) == 0
