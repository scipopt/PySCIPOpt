import gc
import weakref

import pytest

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING, SCIP_LPSOLSTAT
from pyscipopt.scip import is_memory_freed

from helpers.utils import random_mip_1, is_optimized_mode

class MyHeur(Heur):

    def heurexec(self, heurtiming, nodeinfeasible):

        sol = self.model.createSol(self)
        vars = self.model.getVars()

        sol[vars[0]] = 5.0
        sol[vars[1]] = 0.0

        accepted = self.model.trySol(sol)

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

class SimpleRoundingHeuristic(Heur):

    def heurexec(self, heurtiming, nodeinfeasible):

        scip = self.model
        result = SCIP_RESULT.DIDNOTRUN

        # This heuristic does not run if the LP status is not optimal
        lpsolstat = scip.getLPSolstat()
        if lpsolstat != SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": result}

        # We haven't added handling of implicit integers to this heuristic
        if scip.getNImplVars() > 0:
            return {"result": result}

        # Get the current branching candidate, i.e., the current fractional variables with integer requirements
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = scip.getLPBranchCands()

        # Ignore if there are no branching candidates
        if ncands == 0:
            return {"result": result}

        # Create a solution that is initialised to the LP values
        sol = scip.createSol(self, initlp=True)

        # Now round the variables that can be rounded
        for i in range(ncands):
            old_sol_val = branch_cand_sols[i]
            scip_var = branch_cands[i]
            may_round_up = scip_var.varMayRound(direction="up")
            may_round_down = scip_var.varMayRound(direction="down")
            # If we can round in both directions then round in objective function direction
            if may_round_up and may_round_down:
                if scip_var.getObj() >= 0.0:
                    new_sol_val = scip.feasFloor(old_sol_val)
                else:
                    new_sol_val = scip.feasCeil(old_sol_val)
            elif may_round_down:
                new_sol_val = scip.feasFloor(old_sol_val)
            elif may_round_up:
                new_sol_val = scip.feasCeil(old_sol_val)
            else:
                # The variable cannot be rounded. The heuristic will fail.
                continue

            # Set the rounded new solution value
            scip.setSolVal(sol, scip_var, new_sol_val)

        # Now try the solution. Note: This will free the solution afterwards by default.
        stored = scip.trySol(sol)

        if stored:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

def test_heur():
    # create solver instance
    s = Model()
    heuristic = MyHeur()
    s.includeHeur(heuristic, "PyHeur", "custom heuristic implemented in python", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
    s.setPresolve(SCIP_PARAMSETTING.OFF)

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)

    # solve problem
    s.optimize()

    # print solution
    sol = s.getBestSol()
    assert sol != None
    assert round(sol[x]) == 5.0
    assert round(sol[y]) == 0.0

def test_heur_memory():
    if is_optimized_mode():
       pytest.skip()

    def inner():
        s = Model()
        heuristic = MyHeur()
        s.includeHeur(heuristic, "PyHeur", "custom heuristic implemented in python", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
        return weakref.proxy(heuristic)

    heur_prox = inner()
    gc.collect() # necessary?
    with pytest.raises(ReferenceError):
        heur_prox.name

    assert is_memory_freed()

def test_simple_round_heur():
    # create solver instance
    s = random_mip_1(disable_sepa=False, disable_huer=False, node_lim=1)
    heuristic = SimpleRoundingHeuristic()
    s.includeHeur(heuristic, "SimpleRounding", "simple rounding heuristic implemented in python", "Y",
                  timingmask=SCIP_HEURTIMING.DURINGLPLOOP)
    # solve problem
    s.optimize()
