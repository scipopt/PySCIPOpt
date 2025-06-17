###########
Heuristics
###########

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, SCIP_LPSOLSTAT

  scip = Model()

.. contents:: Contents

What is a Heuristic?
=====================

A (primal) heuristic is an algorithm for finding a feasible solution to an optimization problem at lower
computational costs than their exact counterparts but without any optimality guarantees.
The reason that heuristics are implemented in exact optimization solvers are two-fold. It is advantageous
for certain algorithms to have a good intermediate solution, and it is helpful for users that they can
halt the solving process and access the current best solution.

Simple Rounding Heuristic Example
=================================

In this example we show how to implement a simple rounding heuristic in SCIP. The rounding heuristic
will take all the fractional variables with integer requirements from the current relaxation solution,
and attempt to round them to their nearest integer values.

.. code-block:: python

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

To include the heuristic in the SCIP model one would use the following code:

.. code-block:: python

  heuristic = SimpleRoundingHeuristic()
  scip.includeHeur(heuristic, "SimpleRounding", "custom heuristic implemented in python", "Y",
                   timingmask=SCIP_HEURTIMING.DURINGLPLOOP)

.. note:: The ``timingmask`` is especially important when programming your own heuristic. See
  `here <https://www.scipopt.org/doc/html/HEUR.php>`_ for information on timing options and how the affect
  when the heuristic can be called. Note also that heuristics are, as other plugins, called in order of
  their priorities.

.. note:: When you create a SCIP solution object it is important that you eventually free the object.
  This is done by calling ``scip.freeSol(sol)``, although this is not necessary when the solution has been
  passed to ``scip.trySol(sol)`` with ``free=True`` (default behavior).

