"""
Tests the usage of sub solutions found in heuristics with copyLargeNeighborhoodSearch()
"""
import pytest
from pyscipopt import Model, Heur, SCIP_HEURTIMING, SCIP_RESULT


class MyHeur(Heur):
    def __init__(self, model: Model, fix_vars, fix_vals):
        super().__init__()
        self.original_model = model
        self.used = False
        self.fix_vars = fix_vars
        self.fix_vals = fix_vals

    def heurexec(self, heurtiming, nodeinfeasible):
        self.used = True
        # fix z to 2 and optimize the remaining problem
        m2 = self.original_model.copyLargeNeighborhoodSearch(self.fix_vars, self.fix_vals)
        m2.optimize()

        # translate the solution to the original problem
        sub_sol = m2.getBestSol()
        sol_translation = self.original_model.translateSubSol(m2, sub_sol, self)

        accepted = self.original_model.trySol(sol_translation)
        assert accepted
        m2.freeProb()
        return {"result": SCIP_RESULT.FOUNDSOL}


def test_sub_sol():
    m = Model("sub_sol_test")
    x = m.addVar(name="x", lb=0, ub=3, obj=1)
    y = m.addVar(name="y", lb=0, ub=3, obj=2)
    z = m.addVar(name="z", lb=0, ub=3, obj=3)

    m.addCons(4 <= x + y + z)

    # include the heuristic
    my_heur = MyHeur(m, fix_vars= [z], fix_vals = [2])
    m.includeHeur(my_heur, "name", "description", "Y", timingmask=SCIP_HEURTIMING.BEFOREPRESOL, usessubscip=True)

    #optimize
    m.optimize()
    # assert the heuristic did run
    assert my_heur.used

    heur_sol = [2, 0, 2]
    opt_sol = [3, 1, 0]

    found_solutions = []
    for sol in m.getSols():
        found_solutions.append([sol[x], sol[y], sol[z]])

    # both the sub_solution and the real optimum should be in the solution pool
    assert heur_sol in found_solutions
    assert opt_sol in found_solutions
