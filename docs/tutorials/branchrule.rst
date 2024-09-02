###############
Branching Rules
###############

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, Branchrule, SCIP_RESULT

  scip = Model()

.. contents:: Contents

What is Branching
===================

Branching is when an optimization problem is split into smaller subproblems.
Traditionally this is done on an integer variable with a fractional LP solution, with
two child nodes being created with constraints ``x >= ceil(frac)`` and ``x <= floor(frac)``.
In SCIP, arbitrary amount of children nodes can be created, and the constraints added the
created nodes can also be arbitrary. This is not going to be used in the examples below, but this
should be kept in mind when considering your application of your branching rule.

Example Branching Rule
=======================

Here we will program a most infeasible branching rule. This rule selects the integer variable
whose LP solution is most fractional.

.. code-block:: python

  import numpy as np

  class MostInfBranchRule(Branchrule):

      def __init__(self, scip):
          self.scip = scip

      def branchexeclp(self, allowaddcons):

          # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
          # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
          # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
          branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

          # Find the variable that is most fractional
          best_cand_idx = 0
          best_dist = np.inf
          for i in range(npriocands):
              if abs(branch_cand_fracs[i] - 0.5) <= best_dist:
                  best_dist = abs(branch_cand_fracs[i] - 0.5)
                  best_cand_idx = i

          # Branch on the variable with the largest score
          down_child, eq_child, up_child = self.model.branchVarVal(
              branch_cands[best_cand_idx], branch_cand_sols[best_cand_idx])

          return {"result": SCIP_RESULT.BRANCHED}

Let's talk about some features of this branching rule. Currently we only explicitly programmed
a single function, which is called ``branchexeclp``. This is the function that gets called
when branching on an LP optimal solution. While this is the main case, it is not the only
case that SCIP handles. What if there was an LP error at the node, or you are given a set of external
candidates? For more information on this please read `this page <https://www.scipopt.org/doc/html/BRANCH.php>`_.

Now let's discuss what the function returned. We see that it returned a simple dictionary, with the
``"result"`` key and a ``SCIP_RESULT``. This is because inside the function the child nodes
have already been created, and the solver just needs to be made aware of this with the appropriate
code. In the case of branching when something goes wrong (and you have not made the children nodes),
just simply returned ``SCIP_RESULT:DIDNOTRUN``. This will then move on to the next branching rule with
the next highest priority.

.. note::

  Returning ``SCIP_RESULT:DIDNOTRUN`` for more complicated components of the branching rule
  line in ``branchexecps`` is completely encouraged. It is even strongly suggested if you are doing
  an LP specific branching rule.

Now we will finally see how to include the branching rule.

.. code-block:: python

  scip = Model()

  most_inf_branch_rule = MostInfBranchRule(scip)
  scip.includeBranchrule(most_inf_branch_rule, "mostinf", "custom most infeasible branching rule",
                         priority=10000000, maxdepth=-1, maxbounddist=1)

This function ``includeBranchrule`` takes the branching rule as an argument, the name (which will
be visible in the statistics), a description, the priority (for your rule to be called by default it must
be higher than the current highest, which can be quite large), the maxdepth of the branch and bound tree
for which the rule still works (-1 for unlimited), and the maxbounddist (We recommend using 1 and to see
SCIP documentation for an explanation).

Strong Branching Information
=============================

Now let's look at a more complicated example, namely one where we implement our own strong branching rule.
The aim of this example is to provide a basic understanding of what functions are necessary to use
strong branching or obtain some strong branching information.

.. note:: This example is not equivalent to the strong branching rule in SCIP. It ignores some of the
  more complicated interactions in a MIP solver for information found during strong branching.
  These include but are not strictly limited to:

  - What happens if a new primal solution is found and the bound is larger than the cutoff bound?
  - What happens if the bound for one of the children is above a cutoff bound?
  - If probing is enabled then one would need to handle new found bounds appropriately.

.. code-block:: python

  class StrongBranchingRule(Branchrule):

      def __init__(self, scip):
          self.scip = scip

      def branchexeclp(self, allowaddcons):

          branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

          # Initialise scores for each variable
          scores = [-self.scip.infinity() for _ in range(npriocands)]
          down_bounds = [None for _ in range(npriocands)]
          up_bounds = [None for _ in range(npriocands)]

          # Initialise placeholder values
          num_nodes = self.scip.getNNodes()
          lpobjval = self.scip.getLPObjVal()
          lperror = False
          best_cand_idx = 0

          # Start strong branching and iterate over the branching candidates
          self.scip.startStrongbranch()
          for i in range(npriocands):

              # Check the case that the variable has already been strong branched on at this node.
              # This case occurs when events happen in the node that should be handled immediately.
              # When processing the node again (because the event did not remove it), there's no need to duplicate work.
              if self.scip.getVarStrongbranchNode(branch_cands[i]) == num_nodes:
                  down, up, downvalid, upvalid, _, lastlpobjval = self.scip.getVarStrongbranchLast(branch_cands[i])
                  if downvalid:
                      down_bounds[i] = down
                  if upvalid:
                      up_bounds[i] = up
                  downgain = max([down - lastlpobjval, 0])
                  upgain = max([up - lastlpobjval, 0])
                  scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
                  continue

              # Strong branch!
              down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = self.scip.getVarStrongbranch(
                  branch_cands[i], 200, idempotent=False)

              # In the case of an LP error handle appropriately (for this example we just break the loop)
              if lperror:
                  break

              # In the case of both infeasible sub-problems cutoff the node
              if downinf and upinf:
                  return {"result": SCIP_RESULT.CUTOFF}

              # Calculate the gains for each up and down node that strong branching explored
              if not downinf and downvalid:
                  down_bounds[i] = down
                  downgain = max([down - lpobjval, 0])
              else:
                  downgain = 0
              if not upinf and upvalid:
                  up_bounds[i] = up
                  upgain = max([up - lpobjval, 0])
              else:
                  upgain = 0

              # Update the pseudo-costs
              lpsol = branch_cands[i].getLPSol()
              if not downinf and downvalid:
                  self.scip.updateVarPseudocost(branch_cands[i], -self.scip.frac(lpsol), downgain, 1)
              if not upinf and upvalid:
                  self.scip.updateVarPseudocost(branch_cands[i], 1 - self.scip.frac(lpsol), upgain, 1)

              scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
              if scores[i] > scores[best_cand_idx]:
                  best_cand_idx = i

          # End strong branching
          self.scip.endStrongbranch()

          # In the case of an LP error
          if lperror:
              return {"result": SCIP_RESULT.DIDNOTRUN}

          # Branch on the variable with the largest score
          down_child, eq_child, up_child = self.model.branchVarVal(
              branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())

          # Update the bounds of the down node and up node. Some cols might not exist due to pricing
          if self.scip.allColsInLP():
              if down_child is not None and down_bounds[best_cand_idx] is not None:
                  self.scip.updateNodeLowerbound(down_child, down_bounds[best_cand_idx])
              if up_child is not None and up_bounds[best_cand_idx] is not None:
                  self.scip.updateNodeLowerbound(up_child, up_bounds[best_cand_idx])

          return {"result": SCIP_RESULT.BRANCHED}

Let's look at some of the specific functions called during this branch rule. In SCIP we must call ``startStrongbranch``
before doing any actual strong branching (which is done with the call ``getVarStrongbranch``). When we're done
with strong branching we must then also call ``endStrongbranch``.