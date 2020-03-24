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

    # check first nonlinear row
    assert nlrows[0].getLhs() == -m.infinity()
    assert nlrows[0].getRhs() == 6

    linterms = nlrows[0].getLinearTerms()
    assert len(linterms) == 2
    assert str(linterms[0][0]) == "t_x"
    assert linterms[0][1] == 1
    assert str(linterms[1][0]) == "t_y"
    assert linterms[1][1] == 2

    quadterms = nlrows[0].getQuadraticTerms()
    assert len(quadterms) == 3
    assert str(quadterms[0][0]) == "t_x"
    assert str(quadterms[0][1]) == "t_x"
    assert quadterms[0][2] == 3
    assert str(quadterms[1][0]) == "t_y"
    assert str(quadterms[1][1]) == "t_y"
    assert quadterms[1][2] == 4
    assert str(quadterms[2][0]) == "t_x"
    assert str(quadterms[2][1]) == "t_y"
    assert quadterms[2][2] == 5

    # check second nonlinear row
    assert nlrows[1].getLhs() == 9
    assert nlrows[1].getRhs() == 9

    linterms = nlrows[1].getLinearTerms()
    assert len(linterms) == 0

    quadterms = nlrows[1].getQuadraticTerms()
    assert len(quadterms) == 2
    assert str(quadterms[0][0]) == "t_x"
    assert str(quadterms[0][1]) == "t_x"
    assert quadterms[0][2] == 7
    assert str(quadterms[1][0]) == "t_y"
    assert str(quadterms[1][1]) == "t_y"
    assert quadterms[1][2] == 8

    # check third nonlinear row
    assert nlrows[2].getLhs() == -m.infinity()
    assert nlrows[2].getRhs() == 12

    linterms = nlrows[2].getLinearTerms()
    assert len(linterms) == 2
    assert str(linterms[0][0]) == "t_x"
    assert linterms[0][1] == 10
    assert str(linterms[1][0]) == "t_y"
    assert linterms[1][1] == 11

    quadterms = nlrows[2].getQuadraticTerms()
    assert len(quadterms) == 0

if __name__ == "__main__":
    test_nlrow()
